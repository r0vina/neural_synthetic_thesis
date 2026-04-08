import mitsuba as mi
import drjit as dr
import os
from os.path import realpath, join
from tqdm import tqdm
from mitsuba.util import write_bitmap

mi.set_variant('cuda_ad_rgb')
SCENE_DIR = realpath('Assets/lamp_probe')
OUTPUT_DIR = realpath('output')
resx, resy = int(422), int(338)
integrator_type = 'direct_projective'  
config_name = "probe_scene"

render_resolution = (resx, resy)
n_upsampling_steps = 4
spp = 1
max_iterations = 30000
learning_rate = 0.05

sensor = {
    'type': 'perspective',
    'near_clip': 1,
    'far_clip': 1000,
    'fov': 42.2002,
    'to_world': mi.ScalarTransform4f.translate([-47.648800, -609.171021, 39.794601]) \
        @ mi.ScalarTransform4f.rotate(axis=[0, 0, 1], angle=169.35016914939735) \
        @ mi.ScalarTransform4f.rotate(axis=[0, 1, 0], angle=3.738234676621087) \
        @ mi.ScalarTransform4f.rotate(axis=[1, 0, 0], angle=106.93099643799677),

    'sampler': {
        'type': 'independent',
        'sample_count': 512  # Not really used
    },
    'film': {
        'type': 'hdrfilm',
        'width': resx,
        'height': resy,
        'pixel_format': 'rgb',
        'rfilter': {
            # Important: smooth reconstruction filter with a footprint larger than 1 pixel.
            'type': 'gaussian'
        }
    },
}

sensor_cam1 = {
    'type': 'perspective',
    'fov': 65.40863391955037, 
    'fov_axis': 'x',
    'to_world': mi.ScalarTransform4f([[0.994920106014901, -0.0824059370686718, 0.05782079369165012, 0.8723786476636446], [0.09835831522907623, 0.918074060082153, -0.3840125805615866, 2.6605641465531997], [-0.02143885427432941, 0.38774899321610823, 0.921515650321414, 0.41071385447252295], [0.0, 0.0, 0.0, 1.0]]),
    'film': { 'type': 'hdrfilm', 'width': int(6024/16), 'height': int(4020/16), 'rfilter': {'type':'gaussian'} },
    'sampler': { 'type':'independent', 'sample_count': 64 },
    # SIMPLE_RADIAL k1 = -0.07585071808070738 (undistort input images if you need exact alignment)
}

print(type(sensor_cam1))
sensors = []
sensors.append(mi.load_dict(sensor_cam1))


integrator = {
    'type': 'direct_projective',
    'sppc': 1,     # continuous samples
   # 'sppd': 16,    # direct discontinuous (silhouette) samples
    'sppi': 0,
    'hide_emitters': False  # because your sphere IS the emitter now
}

CONFIGS = {
    'probe_scene': {
        'reference': join(SCENE_DIR, 'hdr.hdr'),
        'scene': {
            'type': 'scene',
           # 'sensor': sensor,
            'integrator': integrator,
           # 'emitter': emitter,
       
            #Mirror ball object

            'mirror_ball' : {
                'type' : 'sphere',
                'to_world': mi.ScalarAffineTransform4f().translate([0,0, 0]) @ mi.ScalarAffineTransform4f().scale([1, 1, 1]),
                'emitter': {'type': 'area',
                   'radiance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]}},
                

            }

        },

        'optimize_keys' : ['mirror_ball.to_world']
    },
}



config = CONFIGS[config_name]


config.update({
    'render_resolution': render_resolution,
    'n_upsampling_steps': n_upsampling_steps,
    'spp': spp,
    'max_iterations': max_iterations,
    'learning_rate': learning_rate,
})




# Resolve scene
mi.file_resolver().append(SCENE_DIR)

# Load the scene
scene = mi.load_dict(config['scene'])

# Traverse scene parameters
params = mi.traverse(scene)

sensor = sensors[0]
crop_size = sensor.film().crop_size()

def l1_loss(image, ref):
    """L1 loss function."""
    return dr.mean(dr.abs(image - ref))

def load_ref_image(img_name, resolution, output_dir):
    b = mi.Bitmap(img_name)
    b = b.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, False)
    if dr.any(b.size() != resolution):
        b = b.resample(resolution)

    mi.util.write_bitmap(join(output_dir, 'out_ref.exr'), b)
    
    print('[i] Loaded reference image from:', config['reference'])
    return mi.TensorXf(b)



import time
start_time = time.time()
mi.set_log_level(mi.LogLevel.Warn)
iterations = config['max_iterations']
loss_values = []
spp = config['spp']


probe_mask_refs = ['mask_big.png']

image_ref =  [load_ref_image(join(SCENE_DIR, ref), crop_size, output_dir="outputs") for ref in probe_mask_refs]


opt = mi.ad.Adam(lr=config['learning_rate'])

opt["tx"] = mi.Float(1.27025); dr.enable_grad(opt["tx"])
opt["ty"] = mi.Float(-0.0308244); dr.enable_grad(opt["ty"])
opt["tz"] = mi.Float(7.42806); dr.enable_grad(opt["tz"])


opt["s"] = mi.Float(1.26776); dr.enable_grad(opt["s"])
shape_key = "mirror_ball.to_world"
keys = ["tx", "ty", "tz", "s"]

params.update(opt)


print(params)



for it in tqdm(range(iterations)):
    t0 = time.time()
    t = mi.Vector3f(opt["tx"], opt["ty"], opt["tz"])
    T = mi.Transform4f.translate(t)

    S =  mi.Transform4f.scale(mi.Vector3f(opt["s"], opt["s"], opt["s"]))
    params[shape_key] = T @ S
    params.update()
    
    total_loss = 0.0
    for i in range(len(probe_mask_refs)):
        # Perform a differentiable rendering of the scene  
       

        image = mi.render(scene, params, seed=it, spp= 4*spp, spp_grad=16, sensor=sensors[i])
        #write_bitmap(f'_jerf_out_{it}_{i}.png', image)
        #write_bitmap(f'_jerf_out_{it}_{i}_ref.png', image_ref[i])

        depth = image[..., 0]
        mask = dr.select(depth > 0, 1.0, 0.0)
       # write_bitmap(f'jerf_out_{it}_{i}.png', mask)
        
       # image = image * mask_ref[i]
       # image_ref[i] = image_ref[i] * mask_ref[i]
        # Scale-independent L2 function

        ref = image_ref[i][..., 0]
        ref_mask = dr.select(ref > 0.5, 1.0, 0.0)
        #write_bitmap(f'ref_jerf_out_{it}_{i}.png', ref_mask)
        eps = 1e-6
        inter = dr.sum(mask * ref_mask[i])
        denom = dr.sum(mask) + dr.sum(ref_mask[i])
        loss = 1.0 - (2.0 * inter + eps) / (denom + eps)


        loss = l1_loss(image, image_ref[i])       

        total_loss += loss

    dr.backward(total_loss)

    # Log progress
    elapsed_ms = 1000. * (time.time() - t0)
    current_loss = total_loss.array[0]
    loss_values.append(current_loss)
    
    if it % 100 == 0:
        print(f"\nIteration {it:02d}: loss = {total_loss}")

        write_bitmap(f'pose_jerf_out_{it}_{i}.png', image)
        write_bitmap(f'pose_jerf_out_{it}_{i}_ref.png', image_ref[i])
       # mi.util.write_bitmap(f"bowl_optimized_{it:04d}.png", image)
        for k in keys:
            #print(f"    {k}: {params[k]}")
            grad_norm = dr.mean(dr.abs(dr.grad(opt[k]))).numpy().item()
            print(f"  {k:55s} grad={grad_norm:.3e}")

    if it % 100 == 0:
        with open("empty_parameter_values.txt", "w") as f:
           f.write(str(opt["tx"])+str(opt["ty"]) +str(opt["tz"]) + str(opt["s"]))

    # Take a gradient step
    opt.step()
    params.update(opt)
    
    # Increase rendering quality toward the end of the optimization
    if it in (int(0.7 * iterations), int(0.9 * iterations)):
        spp *= 2
        opt.set_learning_rate(0.5 * opt.learning_rate())
        

end_time = time.time()
print(((end_time - start_time) * 1000) / iterations, ' ms per iteration on average')
print('[i] Final optimized parameters:')
for k in keys:
    print(f"    {k}: {params[k].numpy()}")

mi.set_log_level(mi.LogLevel.Info)

print('[i] Final optimized parameters:')
for k in keys:
    print(f"    {k}: {params[k].numpy()}")