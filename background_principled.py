# Imports

import os
from os.path import realpath, join
import numpy as np
import matplotlib.pyplot as plt
import drjit as dr
import mitsuba as mi

# Mitsuba variant
mi.set_variant('cuda_ad_rgb')

# Scene configuration

SCENE_DIR = realpath('scenes')
OUTPUT_DIR = realpath('output')
resx, resy = int(844), int(676)
integrator_type = 'prbvolpath'  # 'volpathmis'
config_name = 'scene_with_bowl' # 'empty_scene'

# Optimization configuration

render_resolution = (resx, resy)
n_upsampling_steps = 4
spp = 2048
max_iterations = 3000
learning_rate = 5e-3

output_dir = realpath(join('.', 'mitsuba/outputs', config_name))
os.makedirs(output_dir, exist_ok=True)
print('[i] Results will be saved to:', output_dir)

# Emitter configuration

emitter = {
    'type':'envmap',
    'filename': join(SCENE_DIR, 'textures/environment_map.exr'),
    'scale': 1.0,
    'to_world': mi.ScalarTransform4f(np.array(
        [
            [0, 0, -1, 0],
            [0, -1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1],
        ]
    ))

}

# Integrator configuration

integrator = {
    'type': integrator_type,
    #'samples_per_pass': 256,
    'max_depth': 80,
    'hide_emitters': False,
}

# Sensor configuration

# Looking at the receiving plane, not looking through the lens
sensor_to_world = mi.ScalarTransform4f().look_at(
    target=[0, -20, 0],
    origin=[0, -4.65, 0],
    up=[0, 0, 1]
)

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


# Scene configs

CONFIGS = {
    'empty_scene': {
        'reference': join(SCENE_DIR, 'empty_L000.png'),
        'optimize_keys': ('white-bsdf.reflectance.value', 'grey-bsdf.reflectance.value', 'cloth-bsdf.reflectance.value', 'emitter.scale'),
        'scene': {
            'type': 'scene',
           # 'sensor': sensor,
            'integrator': integrator,
            'emitter': emitter,
            # Glass BSDF
         'white-bsdf': {
                'type': 'principled',
                'id': 'white-bsdf',
                'base_color': { 'type': 'rgb', 'value': (1.0, 1.0, 0.95121145) },
                'roughness' : 0.74126035,
                'anisotropic': 0.33597347,
                'metallic' : 1.0,
                'spec_trans': 0.001,
                'specular' : 0.001,
                'spec_tint' : 0.0001,
                'sheen' : 0.0,
                'sheen_tint' : 0.0,
                'clearcoat': 0.0,
                'clearcoat_gloss': 0.3,
                'spec_trans': 0.4

            },
            'grey-bsdf': {
                'type': 'principled',
                'id': 'grey-bsdf',
                'base_color': { 'type': 'rgb', 'value': (0.25754288, 0.2631278, 0.2569997) },
                'roughness' : 0.0789915,
                'anisotropic': 0.8722538,
                'metallic' : 0.000,
                'spec_trans': 0.001,
                'specular' : 0.56056285,
                'spec_tint' : 0.0001,
                'sheen' : 0.0,
                'sheen_tint' : 0.0,
                'clearcoat': 0.0,
                'clearcoat_gloss': 0.3,
                'spec_trans': 0.4

            },

            'cloth-bsdf': {
                'type': 'principled',
                'id': 'cloth-bsdf',
                'base_color': { 'type': 'rgb', 'value': (1.0, 1.0, 0.71882707) },
                'roughness' : 0.73760545,
                'anisotropic': 0.001,
                'metallic' : 0.000,
                'spec_trans': 0.001,
                'specular' : 1,
                'spec_tint' : 0.0001,
                'sheen' : 0.0,
                'sheen_tint' : 0.0,
                'clearcoat': 1.0,
                'clearcoat_gloss': 0.3,
                'spec_trans': 0.4

            },



            
            # Empty Scene
            # Grey backdrop
            'backdrop_gray': {
                'type': 'ply',
                'id': 'backdrop_gray',
                'filename': join(SCENE_DIR,'meshes/empty_scene_stl_deformed_with_glass_bowl-Backdrop_Gray.ply'),
                #'to_world': mi.ScalarTransform4f().rotate(axis=(1, 0, 0), angle=90),
                'bsdf': {'type': 'ref', 'id': 'grey-bsdf'},
            },


            # White backdrop
            'backdrop_white': {
                'type': 'ply',
                'id': 'backdrop_white',
                'filename': join(SCENE_DIR,'meshes/empty_scene_stl_deformed_with_glass_bowl-Backdrop_White.ply'),
                #'to_world': mi.ScalarTransform4f().rotate(axis=(1, 0, 0), angle=90),
                'bsdf': {'type': 'ref', 'id': 'white-bsdf'},
            },

            # Cloth layer
            'cloth' : {
                'type' : 'ply',
                'id' : 'cloth',
                'filename' : join(SCENE_DIR,'meshes/empty_scene_stl_deformed_with_glass_bowl-White_Cloth.ply'),
                'bsdf': {'type': 'ref', 'id': 'cloth-bsdf'},
            },

        },

          #'optimize_keys': ['glass_bowl_bottom.interior_medium.sigma_t.value.value', 'glass_bowl_bottom.interior_medium.albedo.value.value', 'glass-bsdf.eta', 'glass_bowl_top.interior_medium.sigma_t.value.value', 'glass_bowl_top.interior_medium.albedo.value.value']#, 'white-bsdf.reflectance.value', 'grey-bsdf.reflectance.value', 'cloth-bsdf.reflectance.value', 'emitter.scale'],
        'optimize_keys' : ['white-bsdf.roughness.value', 'grey-bsdf.roughness.value', 'cloth-bsdf.roughness.value', 'white-bsdf.metallic.value', 'grey-bsdf.metallic.value', 'cloth-bsdf.metallic.value', 'white-bsdf.specular', 'grey-bsdf.specular', 'cloth-bsdf.specular', 'white-bsdf.base_color.value', 'grey-bsdf.base_color.value', 'cloth-bsdf.base_color.value']

    },

    'scene_with_bowl': {
        'reference': join(SCENE_DIR, 'bowl_L000.png'),
        'scene': {
            'type': 'scene',
           # 'sensor': sensor,
            'integrator': integrator,
            'emitter': emitter,
            # Glass BSDF
                   'white-bsdf': {
                'type': 'principled',
                'id': 'white-bsdf',
                'base_color': { 'type': 'rgb', 'value': (1.0, 1.0, 0.95121145) },
                'roughness' : 0.74126035,
                'anisotropic': 0.33597347,
                'metallic' : 1.0,
                'spec_trans': 0.001,
                'specular' : 0.001,
                'spec_tint' : 0.0001,
                'sheen' : 0.0,
                'sheen_tint' : 0.0,
                'clearcoat': 0.0,
                'clearcoat_gloss': 0.3,
                'spec_trans': 0.4

            },
            'grey-bsdf': {
                'type': 'principled',
                'id': 'grey-bsdf',
                'base_color': { 'type': 'rgb', 'value': (0.25754288, 0.2631278, 0.2569997) },
                'roughness' : 0.0789915,
                'anisotropic': 0.8722538,
                'metallic' : 0.000,
                'spec_trans': 0.001,
                'specular' : 0.56056285,
                'spec_tint' : 0.0001,
                'sheen' : 0.0,
                'sheen_tint' : 0.0,
                'clearcoat': 0.0,
                'clearcoat_gloss': 0.3,
                'spec_trans': 0.4

            },

            'cloth-bsdf': {
                'type': 'principled',
                'id': 'cloth-bsdf',
                'base_color': { 'type': 'rgb', 'value': (1.0, 1.0, 0.71882707) },
                'roughness' : 0.73760545,
                'anisotropic': 0.001,
                'metallic' : 0.000,
                'spec_trans': 0.001,
                'specular' : 1,
                'spec_tint' : 0.0001,
                'sheen' : 0.0,
                'sheen_tint' : 0.0,
                'clearcoat': 1.0,
                'clearcoat_gloss': 0.3,
                'spec_trans': 0.4

            },


            'marker-bsdf': {
                'type': 'plastic',
                'id': 'marker-bsdf',
                'diffuse_reflectance': { 'type': 'rgb', 'value': (0.0001, 0.0001, 0.0001) },
            },

            'glass-bsdf':  {
                'type': 'dielectric',
                'int_ior': 1.504,
                'ext_ior': 1.0
            },
   
            
            # Empty Scene
            # Grey backdrop
            'backdrop_gray': {
                'type': 'ply',
                'id': 'backdrop_gray',
                'filename': join(SCENE_DIR,'meshes/empty_scene_stl_deformed_with_glass_bowl-Backdrop_Gray.ply'),
                #'to_world': mi.ScalarTransform4f().rotate(axis=(1, 0, 0), angle=90),
                'bsdf': {'type': 'ref', 'id': 'grey-bsdf'},
            },


            # White backdrop
            'backdrop_white': {
                'type': 'ply',
                'id': 'backdrop_white',
                'filename': join(SCENE_DIR,'meshes/empty_scene_stl_deformed_with_glass_bowl-Backdrop_White.ply'),
                #'to_world': mi.ScalarTransform4f().rotate(axis=(1, 0, 0), angle=90),
                'bsdf': {'type': 'ref', 'id': 'white-bsdf'},
            },

            # Cloth layer
            'cloth' : {
                'type' : 'ply',
                'id' : 'cloth',
                'filename' : join(SCENE_DIR,'meshes/empty_scene_stl_deformed_with_glass_bowl-White_Cloth.ply'),
                'bsdf': {'type': 'ref', 'id': 'cloth-bsdf'},
            },


            # Glass bowl
            'glass_bowl_bottom': {
                'type': 'ply',
                'id': 'glass_bowl_bottom',
                'filename': join(SCENE_DIR,'meshes/glass_bowl_bottom.ply'),
                'interior': {
                    'type': 'homogeneous',
                    'scale': 1,
                    'sigma_t': {'type' : 'rgb', 'value' : [0.02849365, 0.0211794, 0.0141112]},
                    'albedo': {
                        'type': 'rgb',
                        'value' : [0.0, 0.0, 0.06656517]
                        #'value': [0.8706511, 0.84687555, 0.9000115]
                    },
                },

                'bsdf': {'type': 'ref', 'id': 'glass-bsdf'},
            },

            'glass_bowl_top': {
                'type': 'ply',
                'id': 'glass_bowl_top',
                'filename': join(SCENE_DIR,'meshes/glass_bowl_top.ply'),
                'interior': {
                    'type': 'homogeneous',
                    'scale': 1,
                    'sigma_t': { 'type' : 'rgb', 'value': [0.02604071,0.01343422, 0.00806818] },
                    'albedo': {
                        'type': 'rgb',
                        'value':[0.0, 0.0, 0.0]
                    },
                },

                'bsdf': {'type': 'ref', 'id': 'glass-bsdf'},
            },


            # 'glass_bowl_bottom_markers': {
            #     'type': 'ply',
            #     'id': 'glass_bowl_bottom_markers',
            #     'filename': join(SCENE_DIR,'meshes/glass_bowl_bottom_markers.ply'),
            #     'bsdf': {'type': 'ref', 'id': 'marker-bsdf'},
            # },

            # 'glass_bowl_top_markers': {
            #     'type': 'ply',
            #     'id': 'glass_bowl_top_markers',
            #     'filename': join(SCENE_DIR,'meshes/glass_bowl_top_markers.ply'),
            #     'bsdf': {'type': 'ref', 'id': 'marker-bsdf'},
            # },


        },
        #'optimize_keys': ['glass_bowl_bottom.interior_medium.sigma_t.value.value', 'glass_bowl_bottom.interior_medium.albedo.value.value', 'glass-bsdf.eta', 'glass_bowl_top.interior_medium.sigma_t.value.value', 'glass_bowl_top.interior_medium.albedo.value.value']#, 'white-bsdf.reflectance.value', 'grey-bsdf.reflectance.value', 'cloth-bsdf.reflectance.value', 'emitter.scale'],
        'optimize_keys' : ['white-bsdf.roughness.value', 'grey-bsdf.roughness.value', 'cloth-bsdf.roughness.value', 'white-bsdf.metallic.value', 'grey-bsdf.metallic.value', 'cloth-bsdf.metallic.value', 'white-bsdf.base_color.value', 'grey-bsdf.base_color.value', 'cloth-bsdf.base_color.value']

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

# Load reference image(s)

# Make sure the reference image will have a resolution matching the sensor
sensor = sensors[0]
crop_size = sensor.film().crop_size()

# Reference and mask images
 
empty_refs = ['empty_L000.png', 'empty_R011.png', 'empty_R000.png', 'empty_L002.png']

bowl_refs = ['bowl_L000.png', 'bowl_R011.png', 'bowl_R000.png', 'bowl_L002.png']
mask_refs = ['bowl_L000_mask.png', 'bowl_R011_mask.png', 'bowl_R000_mask.png', 'bowl_L002_mask.png']

def load_ref_image(img_name, resolution, output_dir):
    b = mi.Bitmap(img_name)
    b = b.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, False)
    if dr.any(b.size() != resolution):
        b = b.resample(resolution)

    mi.util.write_bitmap(join(output_dir, 'out_ref.exr'), b)
    
    print('[i] Loaded reference image from:', config['reference'])
    return mi.TensorXf(b)



def load_mask_image(img_name, resolution, output_dir):
    b = mi.Bitmap(img_name)
    b = b.convert(mi.Bitmap.PixelFormat.RGB, mi.Bitmap.Float32, False)
    if dr.any(b.size() != resolution):
        b = b.resample(resolution)

    mi.util.write_bitmap(join(output_dir, 'out_mask.exr'), b)
    
    print('[i] Loaded mask image from:', img_name)
    return mi.TensorXf(b)



if config_name == 'empty_scene':
    image_ref = [load_ref_image(join(SCENE_DIR, ref), crop_size, output_dir=output_dir) for ref in empty_refs]
    rect_mask = np.zeros(image_ref[0].shape)
    rect_mask[50:280, 100:320, 0] = 1.0
    rect_mask[50:280, 100:320, 1] = 1.0
    rect_mask[50:280, 100:320, 2] = 1.0

    mask_ref = None

if config_name == 'scene_with_bowl':
    image_ref = [load_ref_image(join(SCENE_DIR, ref), crop_size, output_dir=output_dir) for ref in bowl_refs]
    mask_ref = [load_mask_image(join(SCENE_DIR, ref), crop_size, output_dir=output_dir) for ref in mask_refs]



# Enable gradients on selected parameters
params = mi.traverse(scene)

print(params)
keys = list(config['optimize_keys'])

if 'glass-bsdf.eta' in keys:
    keys.remove('glass-bsdf.eta')

params.keep(keys)

for k in keys:
    dr.enable_grad(params[k])

params.update()


# Loss functions

opt = mi.ad.Adam(lr=config['learning_rate'], params=params)

def best_gain_rgb(I, R):
    num = dr.sum(I * R, axis=0)
    den = dr.sum(I * I, axis=0) + 1e-12
    return num / den


def hybrid_loss(I_render, I_ref):
    #I_ref = srgb_to_linear(I_ref)
    s = dr.detach(best_gain_rgb(I_render, I_ref))
    diff = I_render * s - I_ref
    data = dr.mean(dr.sqrt(dr.sqr(diff) + 1e-6))
    # optional edge term
   # gR, gI = sobel(I_ref), sobel(I_render * s)
   #edge = dr.mean(dr.sqrt(dr.sqr(gI - gR) + 1e-6))
    return data + 0.2 ##edge


def scale_independent_loss(image, ref):
    """Brightness-independent L2 loss function."""
    scaled_image = image #/ dr.mean(dr.detach(image))
    scaled_ref = ref #/ dr.mean(ref)
    return dr.mean(dr.square(scaled_image - scaled_ref))


def l1_loss(image, ref):
    """L1 loss function."""
    return dr.mean(dr.abs(image - ref))



# Sanity check - print initial parameter values
print("Initial parameter values:")
for k in keys:
            print(f"    {k}: {params[k]}")




# Optimization loop

import time
start_time = time.time()
mi.set_log_level(mi.LogLevel.Warn)
iterations = config['max_iterations']
loss_values = []
spp = config['spp']


for it in range(iterations):
    t0 = time.time()
    
    total_loss = 0.0
    for i in range(len(sensors)):
        # Perform a differentiable rendering of the scene
        image = mi.render(scene, params, seed=it, spp= 2*spp, spp_grad=spp, sensor=sensors[i])
        mi.util.write_bitmap(f"final_bowl_disney_optimized_{it:04d}_{i}.png", image)
       # image = image * mask_ref[i]
       # image_ref[i] = image_ref[i] * mask_ref[i]
        # Scale-independent L2 function
        if config_name == 'empty_scene':
            loss = l1_loss(image * rect_mask, image_ref[i] * rect_mask)
        elif config_name == 'scene_with_bowl':
            loss = l1_loss(image * mask_ref[i], image_ref[i] * mask_ref[i])

       
        for k in keys:
            opt[k] = dr.clip(opt[k], 0.0, 1.0)
        params.update(opt)
        total_loss += loss
    # Back-propagate errors to input parameters and take an optimizer step
    dr.backward(total_loss)


    # Log progress
    elapsed_ms = 1000. * (time.time() - t0)
    current_loss = total_loss.array[0]
    loss_values.append(current_loss)
    
    if it % 20 == 0:
        print(f"\nIteration {it:02d}: loss = {total_loss.numpy():.6f}")
        mi.util.write_bitmap(f"final_bowl_disney_optimized_{it:04d}.png", image)
        for k in keys:
            print(f"    {k}: {params[k]}")
            grad_norm = dr.mean(dr.abs(dr.grad(opt[k]))).numpy().item()
            print(f"  {k:55s} grad={grad_norm:.3e}")
       # mi.util.write_bitmap(f"cbox_reference_{it:04d}.png", image_ref)
   # if it % 1000 == 0:
    #    with open("disney_empty_parameter_values.txt", "w") as f:
     #       for k in keys:
 #               f.write(f"{k}: {params[k].numpy()}\n")
#
    # Take a gradient step
    opt.step()
    
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
