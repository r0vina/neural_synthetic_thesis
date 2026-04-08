
import drjit as dr
import mitsuba
mitsuba.set_variant('cuda_ad_rgb')
import mitsuba as mi
from mitsuba import Float, TensorXf
from mitsuba import load_file
from mitsuba import traverse
from mitsuba import render
from mitsuba.ad import Adam
from mitsuba.util import write_bitmap
import os
from os.path import realpath, join
import numpy as np

from tqdm import tqdm

SCENE_DIR = realpath('scenes')
OUTPUT_DIR = realpath('output')
config_name = "probe_scene"
resx, resy = int(422*2), int(338*2)
integrator_type = 'path'  

render_resolution = (resx, resy)
n_upsampling_steps = 4
spp = 4
max_iterations = 3000
learning_rate = 0.002



emitter = {
    'type':'envmap',
    'id' : 'my_envmap',
    'filename': 'renders/environment_map.exr',
    'scale': 0.98,
    'to_world': mi.ScalarTransform4f(np.array(
        [
            [0, 0, -1, 0],
            [0, -1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1],
        ]
    ))

}

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

sensor2 = {
    'type': 'perspective',
    'near_clip': 1,
    'far_clip': 1000,
    'fov': 42.2000981,
    'to_world': mi.ScalarTransform4f.translate([245.406006,-585.906006, 68.965302]) \
        @ mi.ScalarTransform4f.rotate(axis=[0, 0, 1], angle=-164.10868198032958) \
        @ mi.ScalarTransform4f.rotate(axis=[0, 1, 0], angle=-2.1634963571039774) \
        @ mi.ScalarTransform4f.rotate(axis=[1, 0, 0], angle=98.93123982545688),

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

sensor3 = {
    'type': 'perspective',
    'near_clip': 1,
    'far_clip': 1000,
    'fov': 42.200115,
    'to_world': mi.ScalarTransform4f.translate([-280.697998, -567.486023, 54.453999]) \
        @ mi.ScalarTransform4f.rotate(axis=[0, 0, 1], angle=145.52021287111316) \
        @ mi.ScalarTransform4f.rotate(axis=[0, 1, 0], angle=-1.0390215033121508) \
        @ mi.ScalarTransform4f.rotate(axis=[1, 0, 0], angle=107.01258987782174),

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

sensor4 = {
    'type': 'perspective',
    'near_clip': 1,
    'far_clip': 1000,
    'fov': 42.2000981,
    'to_world': mi.ScalarTransform4f.translate([208.130997, -599.528992, 34.866699]) \
        @ mi.ScalarTransform4f.rotate(axis=[0, 0, 1], angle=-166.04270468533204) \
        @ mi.ScalarTransform4f.rotate(axis=[0, 1, 0], angle=-4.24835973450136) \
        @ mi.ScalarTransform4f.rotate(axis=[1, 0, 0], angle=106.32412730005196),

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
sensors = []
sensors.append(mi.load_dict(sensor))
sensors.append(mi.load_dict(sensor2))
sensors.append(mi.load_dict(sensor3))
sensors.append(mi.load_dict(sensor4))


# Integrator configuration

integrator = {
    'type': integrator_type,
    #'samples_per_pass': 256,
    #'aovs' : 'dd:depth',
    'max_depth': 3,
    'hide_emitters': False,
   # 'sppi' : 0
}


# integrator = {
#     'type': 'direct_projective',
#     'sppc': 1,     # continuous samples
#    # 'sppd': 16,    # direct discontinuous (silhouette) samples
#     'sppi': 0,
#     'hide_emitters': False  # because your sphere IS the emitter now
# }


# Scene configs

CONFIGS = {
    'probe_scene': {
        'reference': join(SCENE_DIR, 'bowl_L000.png'),
        'scene': {
            'type': 'scene',
           # 'sensor': sensor,
            'integrator': integrator,
            'emitter': emitter,
       
            #Mirror ball object

            'mirror_ball' : {
                'type' : 'sphere',
                'to_world': mi.ScalarAffineTransform4f().translate([0,0, 0]) @ mi.ScalarAffineTransform4f().scale([1, 1, 1]),
                #'emitter': {'type': 'area',
                  #  'radiance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]}},
                'bsdf': {'type': 'conductor'}

            }

        },

       # 'optimize_keys': ['my_envmap.data'],
        'optimize_keys' : ['mirror_ball.to_world']
    },
}



config = CONFIGS[config_name]

# Update config with optimization parameters

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
keys = list(config['optimize_keys'])
#params.keep(keys)
#print(params)

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
    
    print('[i] Loaded reference image from:', img_name)
    return mi.TensorXf(b)

import time
start_time = time.time()
mi.set_log_level(mi.LogLevel.Warn)
iterations = config['max_iterations']
loss_values = []
spp = config['spp']


empty_refs = ['empty_L000.png', 'empty_R011.png', 'empty_R000.png', 'empty_L002.png']
probe_refs = [
['img_HDR_L_000_2048.png', 'img_HDR_L_000_1024.png', 'img_HDR_L_000_512.png', 'img_HDR_L_000_256.png', 'img_HDR_L_000_128.png', 'img_HDR_L_000_64.png', 'img_HDR_L_000_32.png', 'img_HDR_L_000_16.png', 'img_HDR_L_000_8.png', 'img_HDR_L_000_4.png'],
['img_HDR_R_011_2048.png', 'img_HDR_R_011_1024.png', 'img_HDR_R_011_512.png', 'img_HDR_R_011_256.png', 'img_HDR_R_011_128.png', 'img_HDR_R_011_64.png', 'img_HDR_R_011_32.png', 'img_HDR_R_011_16.png', 'img_HDR_R_011_8.png', 'img_HDR_R_011_4.png'],
['img_HDR_L_002_2048.png', 'img_HDR_L_002_1024.png', 'img_HDR_L_002_512.png', 'img_HDR_L_002_256.png', 'img_HDR_L_002_128.png', 'img_HDR_L_002_64.png', 'img_HDR_L_002_32.png', 'img_HDR_L_002_16.png', 'img_HDR_L_002_8.png', 'img_HDR_L_002_4.png'],
['img_HDR_R_000_2048.png', 'img_HDR_R_000_1024.png', 'img_HDR_R_000_512.png', 'img_HDR_R_000_256.png', 'img_HDR_R_000_128.png', 'img_HDR_R_000_64.png', 'img_HDR_R_000_32.png', 'img_HDR_R_000_16.png', 'img_HDR_R_000_8.png', 'img_HDR_R_000_4.png'],

]

# refs[i][j] = TensorXf for camera i, exposure j
refs = []
for i in range(len(sensors)):
    refs_i = []
    for j in range(len(probe_refs[i])):
        refs_i.append(load_ref_image(join(SCENE_DIR, probe_refs[i][j]), crop_size, output_dir="outputs"))
    refs.append(refs_i)



probe_mask_refs = ['sam_L000.png', 'sam_R011.png', 'sam_L002.png', 'sam_R000.png']

#image_ref =  [load_ref_image(join(SCENE_DIR, ref), crop_size, output_dir="outputs") for ref in probe_mask_refs]
image_ref =  [load_ref_image(join(SCENE_DIR, ref), crop_size, output_dir="outputs") for ref in probe_refs[0]]

opt = mi.ad.Adam(lr=config['learning_rate'])#, params=params)

opt["tx"] = mi.Float(43.7513)# dr.enable_grad(opt["tx"])
opt["ty"] = mi.Float(-274.146)# dr.enable_grad(opt["ty"])
opt["tz"] = mi.Float(-112.828)# dr.enable_grad(opt["tz"])


opt["s"] = mi.Float(50.2399) #dr.enable_grad(opt["s"])
shape_key = "mirror_ball.to_world"
keys = ["my_envmap.data"]
params.update(opt)
#print("OPTT", mi.ScalarTransform4f.translate(opt["t"]))
t = mi.Vector3f(opt["tx"], opt["ty"], opt["tz"])
T = mi.Transform4f.translate(t)
#T = mi.Transform4f.translate(opt["t"])
S =  mi.Transform4f.scale(mi.Vector3f(opt["s"], opt["s"], opt["s"]))
params[shape_key] = T @ S
params.update()


params.keep(['my_envmap.data'])
params['my_envmap.data'] = dr.full(TensorXf, 0.5, (128,256,3))
opt['my_envmap.data'] = params['my_envmap.data']

# per-image exposures
for i in range(len(sensors)):
    for j in range(len(probe_refs[i])):
        key = f'log_e_{i}_{j}'
        opt[key] = mi.Float(0.0)         # exp(0)=1 initial
        dr.enable_grad(opt[key])

opt["log_e_k"] = mi.Float(0.0)

params.update()
params.update(opt)
print(params)

write_bitmap('ORIG_jerf_envmap_.png', params['my_envmap.data'])
def l1_loss(image, ref):
    """L1 loss function."""
    return dr.mean(dr.abs(image - ref))
def l2_loss(image: mi.TensorXf, ref: mi.TensorXf) -> mi.Float:
    # use dr.mean on the underlying array to avoid Tensor shape weirdness
    diff = image.array - ref.array
    return dr.mean(dr.sqr(diff))

def log_l2(image, ref, eps=1e-4):
    a = dr.log(image.array + eps)
    b = dr.log(ref.array + eps)
    return dr.mean(dr.sqr(a - b))

for it in tqdm(range(iterations)):
    t0 = time.time()

    total_loss = 0.0
    for i in range(len(sensors)):

        image = mi.render(scene, params=params, seed=it, spp=4*spp, spp_grad=16, sensor=sensors[i])


        for j in range(len(refs[i])):

            e = dr.exp(opt[f'log_e_{i}_{j}'])
            total_loss += l2_loss(image * e, refs[i][j])

            if it % 100 == 0:
                print(f"\nIteration {it:02d}: loss = {total_loss}")
                print(f"\Luminance {e}")
                e = dr.exp(opt[f'log_e_{i}_{j}'])
                write_bitmap('_jerf_envmap_%03i.exr' % it, params['my_envmap.data'])
                write_bitmap(f'_jerf_out_{it}_{i}_{j}.png', image*e)
                write_bitmap(f'_jerf_out_{it}_{i}_{j}_ref.png', refs[i][j])
                with open("luminance.txt", "w") as f:
                    f.write(str(opt[f"log_e_{i}_{j}"]))

                for k in keys:
        
                    grad_norm = dr.mean(dr.abs(dr.grad(opt[k]))).numpy().item()
                    print(f"  {k:55s} grad={grad_norm:.3e}")

    dr.backward(total_loss)

    # Log progress
    elapsed_ms = 1000. * (time.time() - t0)
    current_loss = total_loss.array[0]
    loss_values.append(current_loss)
    
    
     
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