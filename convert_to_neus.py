import os
import numpy as np

# You need COLMAP's read_write_model.py in your PYTHONPATH.
# Get it from the COLMAP repo/scripts if you don't already have it.
from read_write_model import read_model, qvec2rotmat

def K_from_colmap_camera(cam):
    """
    Build K from COLMAP camera model.
    Supports the common models; extend if yours differs.
    """
    model = cam.model
    p = cam.params

    if model in ["SIMPLE_PINHOLE"]:
        f, cx, cy = p
        fx = fy = f
    elif model in ["PINHOLE"]:
        fx, fy, cx, cy = p
    elif model in ["SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"]:
        f, cx, cy, _k = p
        fx = fy = f
    elif model in ["RADIAL"]:
        f, cx, cy, _k1, _k2 = p
        fx = fy = f
    elif model in ["OPENCV", "OPENCV_FISHEYE"]:
        fx, fy, cx, cy = p[:4]
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {model}")

    K = np.array([[fx, 0,  cx],
                  [0,  fy, cy],
                  [0,  0,  1]], dtype=np.float64)
    return K

def write_idr_npz(colmap_model_dir, image_dir, out_npz):
    cams, imgs, _pts = read_model(colmap_model_dir, ext=".bin")

    # Map COLMAP image records to your extracted frames by filename.
    # Assumes your case/image filenames match COLMAP's registered image names.
    frame_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".png")])

    data = {}

    for i, fn in enumerate(frame_files):
        # Find COLMAP image by name:
        colmap_img = None
        for _id, im in imgs.items():
            if im.name == fn:
                colmap_img = im
                break
        if colmap_img is None:
            raise RuntimeError(f"COLMAP has no registered image named {fn}")


        cam = cams[colmap_img.camera_id]
        print(cam)
        K = K_from_colmap_camera(cam)

        # COLMAP stores world-to-camera as (qvec, tvec)
        R = qvec2rotmat(colmap_img.qvec)
        t = colmap_img.tvec.reshape(3, 1)

        P = K @ np.hstack([R, t])  # 3x4

        world_mat = np.eye(4, dtype=np.float64)
        world_mat[:3, :4] = P

        scale_mat = np.eye(4, dtype=np.float64)  # start identity; preprocess will fill proper normalization

        data[f"world_mat_{i}"] = world_mat
        data[f"scale_mat_{i}"] = scale_mat
        print(i)

    np.savez(out_npz, **data)
    print(f"Wrote {out_npz} with {len(frame_files)} views")

if __name__ == "__main__":
    write_idr_npz(
        colmap_model_dir="../../colmap_scene/paintedjar/paintedworkspace/sparse/0",
        image_dir="../../colmap_scene/paintedjar/frames",
        out_npz="cameras_sphere_painted.npz"
    )
