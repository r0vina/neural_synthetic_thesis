import drjit
import mitsuba
mitsuba.set_variant('cuda_ad_rgb')
import mitsuba as mi
from mitsuba import Float, TensorXf
from mitsuba import load_file
from mitsuba import traverse
from mitsuba import render
from mitsuba.ad import Adam
from mitsuba.util import write_bitmap

# Load example scene
mi.file_resolver().append('bunny')
scene = load_file('bunny/bunny.xml')

# Find differentiable scene parameters
params = traverse(scene)

print(params)

#param_res = params['my_envmap.resolution']
param_ref = params['my_envmap.data']

# Discard all parameters except for one we want to differentiate
params.keep(['my_envmap.data'])

# Render a reference image (no derivatives used yet)
image_ref = render(scene, spp=16)
crop_size = scene.sensors()[0].film().crop_size()

#print(crop_size)
write_bitmap('out_ref.png', image_ref)

print(drjit.shape(param_ref))
# Change to a uniform white lighting environment
params['my_envmap.data'] = drjit.full(TensorXf, 0.5, (125,251,3))

params.update()

# Construct an Adam optimizer that will adjust the parameters 'params'
opt = Adam(lr = 0.05)
opt['my_envmap.data'] = params['my_envmap.data']
params.update(opt)



for it in range(100):
    # Perform a differentiable rendering of the scene
    image = render(scene, params, spp=16)
    write_bitmap('out_%03i.png' % it, image)
    write_bitmap('envmap_%03i.png' % it, params['my_envmap.data'])

    # Objective: MSE between 'image' and 'image_ref'
    ob_val = drjit.sum(drjit.sqr(image - image_ref)) / len(image)
    print(params['my_envmap.data'])

    # Back-propagate errors to input parameters
    drjit.backward(ob_val)

    # Optimizer: take a gradient step
    opt.step()
    params.update(opt)

    # Compare iterate against ground-truth value
    err_ref = drjit.sum(drjit.sqr(param_ref - params['my_envmap.data']))
    print(err_ref)
    print('Iteration %03i:') # error=%g' % (it, Float(err_ref)))

