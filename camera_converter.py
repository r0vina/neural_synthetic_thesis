import numpy as np

#FRAME_LIST_PATH = "/home/r0vina/colmap_scene/workspace/text/frames.txt"
#RIG_CALIB_PATH  = "/home/r0vina/colmap_scene/workspace/text/rigs.txt"
#CAMERA_LIST_PATH = "/home/r0vina/colmap_scene/workspace/text/cameras.txt"

#FRAME_LIST_PATH = "/home/r0vina/colmap_scene/paintedjar/sparse_txt/frames.txt"
#RIG_CALIB_PATH  = "/home/r0vina/colmap_scene/paintedjar/sparse_txt/rigs.txt"
#CAMERA_LIST_PATH = "/home/r0vina/colmap_scene/paintedjar/sparse_txt/cameras.txt"

#FRAME_LIST_PATH = "/home/r0vina/colmap_scene/full_set/table/sparse_txt/frames.txt"
#RIG_CALIB_PATH  = "/home/r0vina/colmap_scene/full_set/table/sparse_txt/rigs.txt"
#CAMERA_LIST_PATH = "/home/r0vina/colmap_scene/full_set/table/sparse_txt/cameras.txt"

FRAME_LIST_PATH = "/home/r0vina/colmap_scene/full_set/clear/workspace/sparse_txt/frames.txt"
RIG_CALIB_PATH  = "/home/r0vina/colmap_scene/full_set/clear/workspace/sparse_txt/rigs.txt"
CAMERA_LIST_PATH = "/home/r0vina/colmap_scene/full_set/clear/workspace/sparse_txt/cameras.txt"


AXIS_FIX_ON = False
AXIS_FIX = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1],
], dtype=np.float64)

def noncomment_lines(path):
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            yield s

def parse_camera_list(path):
    cams = {}
    for s in noncomment_lines(path):
        parts = s.split()
        cam_id = int(parts[0])
        model = parts[1]
        w = int(parts[2]); h = int(parts[3])
        params = list(map(float, parts[4:]))
        cams[cam_id] = {"model": model, "w": w, "h": h, "params": params}
    return cams

def parse_frame_list_line(s):
    parts = s.split()
    frame_id = int(parts[0])
    rig_id = int(parts[1])
    qw, qx, qy, qz = map(float, parts[2:6])
    tx, ty, tz = map(float, parts[6:9])
    n = int(parts[9])
    triples = []
    idx = 10
    for _ in range(n):
        stype = parts[idx]; sid = int(parts[idx+1]); did = int(parts[idx+2])
        triples.append((stype, sid, did))
        idx += 3
    return frame_id, rig_id, (qw,qx,qy,qz), (tx,ty,tz), triples

def quat_to_rot(qw,qx,qy,qz):
    w,x,y,z = qw,qx,qy,qz
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w),   1-2*(x*x+y*y)],
    ], dtype=np.float64)

def rig_from_world(qw,qx,qy,qz,tx,ty,tz):
    R = quat_to_rot(qw,qx,qy,qz)
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = np.array([tx,ty,tz], dtype=np.float64)
    return T  # world->rig

def fov_x_deg(w, f_px):
    return float(np.degrees(2.0*np.arctan(w/(2.0*f_px))))

cams = parse_camera_list(CAMERA_LIST_PATH)

# We don't actually need rig calib here because your rig is ref camera with no extrinsics given.
# If you later have per-sensor pose blocks, that's where they'd get applied.

frames = [parse_frame_list_line(s) for s in noncomment_lines(FRAME_LIST_PATH)]

print("import mitsuba as mi")
print("mi.set_variant('cuda_ad_rgb')  # or 'llvm_ad_rgb'")
print("")
print("scene_dict = { 'type': 'scene',")

for frame_id, rig_id, (qw,qx,qy,qz), (tx,ty,tz), triples in frames[:2]:
    cam_sensor_ids = [sid for (stype, sid, did) in triples if stype.upper() == "CAMERA"]
    if not cam_sensor_ids:
        continue
    cam_id = cam_sensor_ids[0]
    cam = cams[cam_id]

    if cam["model"] != "SIMPLE_RADIAL":
        raise ValueError(f"Expected SIMPLE_RADIAL, got {cam['model']}")

    w, h = cam["w"], cam["h"]
    f, cx, cy, k1 = cam["params"]
    fovx = fov_x_deg(w, f)

    T_rig_from_world = rig_from_world(qw,qx,qy,qz,tx,ty,tz)
    T_world_from_rig = np.linalg.inv(T_rig_from_world)

    T_world_from_cam = T_world_from_rig  # rig == camera

    if AXIS_FIX_ON:
        T_world_from_cam = T_world_from_cam @ AXIS_FIX

    name = f"sensor_cam{cam_id:02d}_f{frame_id:04d}"
    print(f"  '{name}': {{")
    print("    'type': 'perspective',")
    print(f"    'fov': {fovx}, 'fov_axis': 'x',")
    print(f"    'to_world': mi.ScalarTransform4f({T_world_from_cam.tolist()}),")
    print("    'film': { 'type': 'hdrfilm', "
          f"'width': {w}, 'height': {h}, 'rfilter': {{'type':'gaussian'}} ,")
    print("    'sampler': { 'type':'independent', 'sample_count': 64 },")
    # Metadata note: k1 exists but isn't applied by this sensor.
    print(f"    # SIMPLE_RADIAL k1 = {k1} (undistort input images if you need exact alignment)")
    print("  },")
print("}")
print("")
print("scene = mi.load_dict(scene_dict)")
print("print('Sensors:', len(scene.sensors()))")
