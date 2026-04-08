
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

mi.set_variant('cuda_ad_rgb')
SCENE_DIR = realpath('Assets/lamp_probe')
OUTPUT_DIR = realpath('output')
resx, resy = int(422), int(338)
integrator_type = 'path'  
config_name = "probe_scene"

render_resolution = (resx, resy)
n_upsampling_steps = 4
spp = 8
max_iterations = 30000
learning_rate = 0.005

emitter = {
    'type':'envmap',
    'id' : 'my_envmap',
    'filename': 'png_lamp_envmap_400.exr',
    'scale': 0.98,
    'to_world': mi.ScalarTransform4f(np.array(
        [
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1],
        ]
    )),

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

sensor_cam1 = {
    'type': 'perspective',
    'fov': 65.40863391955037, 
    'fov_axis': 'x',
    'to_world': mi.ScalarTransform4f([[0.994920106014901, -0.0824059370686718, 0.05782079369165012, 0.8723786476636446], [0.09835831522907623, 0.918074060082153, -0.3840125805615866, 2.6605641465531997], [-0.02143885427432941, 0.38774899321610823, 0.921515650321414, 0.41071385447252295], [0.0, 0.0, 0.0, 1.0]]) @ mi.ScalarTransform4f.rotate(axis=[0, 0, 1], angle=180),
    'film': { 'type': 'hdrfilm', 'width': int(6024/16), 'height': int(4020/16), 'rfilter': {'type':'gaussian'} },
    'sampler': { 'type':'independent', 'sample_count': 64 },
    # SIMPLE_RADIAL k1 = -0.07585071808070738 (undistort input images if you need exact alignment)
}

sensor_cam2 = {
    'type': 'perspective',
    'fov': 65.40863391955037, 
    'fov_axis': 'x',
    'to_world': mi.ScalarTransform4f([[0.9949396119340481, -0.08223783228641567, 0.05772441030757292, 0.8727213102286602], [0.09816306329225892, 0.9181436868461468, -0.38389605795012416, 2.6586361916970844], [-0.021428523271696292, 0.387619799862438, 0.9215702410261508, 0.41072213989006284], [0.0, 0.0, 0.0, 1.0]]) @ mi.ScalarTransform4f.rotate(axis=[0, 0, 1], angle=180),
    'film': { 'type': 'hdrfilm', 'width' :int(1985/3), 'height': int(1085/3), 'rfilter': {'type':'gaussian'} },
    'sampler': { 'type':'independent', 'sample_count': 64 },
    # SIMPLE_RADIAL k1 = -0.07585071808070738 (undistort input images if you need exact alignment)
}


print(type(sensor_cam1))
sensors = []
sensors.append(mi.load_dict(sensor_cam2))


integrator = {
    'type': integrator_type,
    #'samples_per_pass': 256,
    #'aovs' : 'dd:depth',
    'max_depth': 3,
    'hide_emitters': False,
   # 'sppi' : 0
}


CONFIGS = {
    'probe_scene': {
        'reference': join(SCENE_DIR, '00001.png'),

        'scene': {
            'type': 'scene',
           # 'sensor': sensor,
            'integrator': integrator,
            'emitter': emitter,
            
            'tex': {

                'type': 'diffuse',

                'reflectance' : {
                    'type' : 'bitmap',
                    'filename' : 'Assets/lamp_probe/material_0.png'
                    #'value' : [1.0, 0.1, 0.2]
                },
            },


            'mirror_ball' : {
                'type' : 'sphere',
                'to_world': mi.ScalarAffineTransform4f().translate([0,0, 0]) @ mi.ScalarAffineTransform4f().scale([1, 1, 1]),
                'bsdf': {'type': 'conductor'}

            },

       
      
            'table' : {
                'type' : 'ply',
                #'to_world': mi.ScalarAffineTransform4f().translate([0,0, 0]) @ mi.ScalarAffineTransform4f().scale([1, 1, 1]),
                #'bsdf': {'type': 'conductor'},
                'filename': "Assets/lamp_probe/table.ply",

                #'emitter': {'type': 'area',
                #   'radiance': {'type': 'rgb', 'value': [1.0, 0.0, 0.0]}},
                'bsdf_id' : {
                    'type' : 'ref',
                    'id' : 'tex'
                }


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
def l2_loss(image: mi.TensorXf, ref: mi.TensorXf) -> mi.Float:
    # use dr.mean on the underlying array to avoid Tensor shape weirdness
    diff = image.array - ref.array
    return dr.mean(dr.sqr(diff))

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


probe_mask_refs = ['mask.png']
probe_refs = [['00001.png']]
image_ref =  [load_ref_image(join(SCENE_DIR, ref), crop_size, output_dir="outputs") for ref in probe_refs[0]]
mask_ref =  [load_ref_image(join(SCENE_DIR, ref), crop_size, output_dir="outputs") for ref in probe_mask_refs]


refs = []
for i in range(len(sensors)):
    refs_i = []
    for j in range(len(probe_refs[i])):
        refs_i.append(load_ref_image(join(SCENE_DIR, probe_refs[i][j]), crop_size, output_dir="outputs"))
    refs.append(refs_i)


opt = mi.ad.Adam(lr=config['learning_rate'])

opt["tx"] = mi.Float(1.30197); #dr.enable_grad(opt["tx"])
opt["ty"] = mi.Float(-0.197331); #dr.enable_grad(opt["ty"])
opt["tz"] = mi.Float(7.76516); #dr.enable_grad(opt["tz"])


# opt["tx"] = mi.Float(1.3839); #dr.enable_grad(opt["tx"])
# opt["ty"] = mi.Float(-0.35261); #dr.enable_grad(opt["ty"])
# opt["tz"] = mi.Float(7.07224); #dr.enable_grad(opt["tz"])

opt["s"] = mi.Float(1.20211); #dr.enable_grad(opt["s"])
shape_key = "mirror_ball.to_world"
keys = ['tex.reflectance.data']

params.update(opt)


print(params)
params.update(opt)
#print("OPTT", mi.ScalarTransform4f.translate(opt["t"]))
t = mi.Vector3f(opt["tx"], opt["ty"], opt["tz"])
T = mi.Transform4f.translate(t)
#T = mi.Transform4f.translate(opt["t"])
S =  mi.Transform4f.scale(mi.Vector3f(opt["s"], opt["s"], opt["s"]))
params[shape_key] = T @ S
params.update()



params.keep(['tex.reflectance.data', 'my_envmap.scale'])
params['tex.reflectance.data'] = dr.full(TensorXf, 0.5, (1024*2, 1024*2,3))
opt['tex.reflectance.data'] = params['tex.reflectance.data']
opt['my_envmap.scale'] = params['my_envmap.scale']

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


for it in tqdm(range(iterations)):
    t0 = time.time()

    total_loss = 0.0
    for i in range(len(sensors)):

        image = mi.render(scene, params=params, seed=it, spp=4*spp, spp_grad=16, sensor=sensors[i])


        for j in range(len(refs[i])):

            #e = dr.exp(opt[f'log_e_{i}_{j}'])
            total_loss += l1_loss(image, refs[i][j])

            if it % 100 == 0:
                print(f"\nIteration {it:02d}: loss = {total_loss}")
                #print(f"\Luminance {e}")
                #e = dr.exp(opt[f'log_e_{i}_{j}'])
                write_bitmap(f'table_tex_{it}_{i}_{j}.png', image)
                write_bitmap(f'table_tex_{it}_.png',params['tex.reflectance.data'])
               
                #write_bitmap(f'_lamp_{it}_{i}_{j}_ref.exr', refs[i][j])
                #with open("lamp_luminance.txt", "w") as f:
                 #   f.write(str(opt[f"log_e_{i}_{j}"]))

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

    #opt['tex.reflectance.data'] = dr.clip(opt['tex.reflectance.data'], 0.0, 1.0)

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