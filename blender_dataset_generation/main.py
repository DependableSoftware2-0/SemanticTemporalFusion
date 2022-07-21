import blenderproc as bproc
import numpy as np
import random
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('scene', nargs='?', default="examples/resources/scene.obj", help="Path to the scene.obj file")
parser.add_argument('cc_material_path', nargs='?', default="resources/cctextures", help="Path to CCTextures folder, see the /scripts for the download script.")
parser.add_argument('output_dir', nargs='?', default="examples/basics/camera_sampling/output", help="Path to where the final files, will be saved")
args = parser.parse_args()

bproc.init()

# load the objects into the scene
#objs = bproc.loader.load_obj(args.scene)
objs = bproc.loader.load_blend(args.scene)

for obj in objs:
    print (obj.get_cp("category_id") , obj.get_name())


# Load all recommended cc materials, however don't load their textures yet
cc_materials = bproc.loader.load_ccmaterials(args.cc_material_path, preload=True)

# Go through all objects
for obj in objs:
    # For each material of the object
    for i in range(len(obj.get_materials())):
        # In 40% of all cases
        #if np.random.uniform(0, 1) <= 0.4:
        # Replace the material with a random one from cc materials
        obj.set_material(i, random.choice(cc_materials))

# Now load all textures of the materials that were assigned to at least one object
bproc.loader.load_ccmaterials(args.cc_material_path, fill_used_empty_materials=True)


# define a light and set its location and energy level
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([5, -5, 5])
light.set_energy(1000)

# Find point of interest, all cam poses should look towards it
poi = bproc.object.compute_poi(objs)
# Sample random camera location above objects
location = np.random.uniform([-0.5, -0.5, 0.2], [0.5, 0.5, 1])
print ("Position interst , ", poi , location)
# Compute rotation based on vector going from location towards poi
rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))

#Linear interpolation between 2 points based on the ratio t
uncertainty = 0.001
def lerp3D(v0, v1, t):
    #check if t value is under 1
    assert 0 <= t <= 1
    x0, y0, z0 = v0
    x1, y1, z1 = v1
    x = (1-t)*x0 + t*x1 + uncertainty*np.random.randn()
    y = (1-t)*y0 + t*y1 + uncertainty*np.random.randn()
    z = (1-t)*z0 + t*z1 + uncertainty*np.random.randn()
    return [x, y, z]


# Sample five camera poses
distance_range = np.linspace(0, 1, 5)
for i in distance_range[:3]:
    # Compute linear interpolation 
    new_location = lerp3D(location, poi, i)
    print ("Position interst , ", poi , new_location)
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(new_location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)


# activate normal and depth rendering
#bproc.renderer.enable_normals_output()
bproc.renderer.enable_depth_output(activate_antialiasing=False)

# render the whole pipeline
data = bproc.renderer.render()



# Render segmentation masks (per class and per instance)
data.update(bproc.renderer.render_segmap(map_by=["class"]))
#seg_data = bproc.renderer.render_segmap(map_by=["instance", "class", "name"])


# write the data to a .hdf5 container
bproc.writer.write_hdf5(args.output_dir, data)


# write the animations into .gif files
#bproc.writer.write_gif_animation(args.output_dir, data)


#bproc.writer.write_bop(os.path.join(args.output_dir, 'bop_data'),
#                       depths = data["depth"],
#                       colors = data["colors"],
#                       color_file_format = "PNG",
#                       m2mm = True,
#                       ignore_dist_thres = 10)
