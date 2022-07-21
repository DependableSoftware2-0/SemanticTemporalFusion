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


# Define a function that samples the initial pose of a given object above the ground
def sample_initial_pose(obj: bproc.types.MeshObject):
    obj.set_location(bproc.sampler.upper_region(objects_to_sample_on=room_planes[0:1],
                                                min_height=1, max_height=4, face_sample_range=[0.4, 0.6]))
    obj.set_rotation_euler(np.random.uniform([0, 0, 0], [np.pi * 2, np.pi * 2, np.pi * 2]))



# create room
room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]

# sample light color and strenght from ceiling
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')
# sample point light on shell
light_point = bproc.types.Light()

# Load all recommended cc materials, however don't load their textures yet
cc_materials = bproc.loader.load_ccmaterials(args.cc_material_path, preload=True)


       
# Go through all objects
for obj in objs:
    # For each material of the object
    for i in range(len(obj.get_materials())):
        # In 90% of all cases
        if np.random.uniform(0, 1) <= 0.9:
        # Replace the material with a random one from cc materials
            obj.set_material(i, random.choice(cc_materials))

random_cc_texture = np.random.choice(cc_materials)
for plane in room_planes:
    plane.replace_materials(random_cc_texture)


# set shading and physics properties and randomize PBR materials
for j, obj in enumerate(objs):
    obj.set_shading_mode('auto')
        
    mat = obj.get_materials()[0]
    mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
    mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))

# sample light color and strenght from ceiling
light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                   emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))    
light_plane.replace_materials(light_plane_material)

# sample point light on shell
light_point.set_energy(np.random.uniform(100, 1000))
light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                        elevation_min = 5, elevation_max = 89, uniform_volume = False)
light_point.set_location(location)


# Sample objects on the given surface
placed_objects = bproc.object.sample_poses_on_surface(objects_to_sample=objs,
                                         surface=room_planes[0],
                                         sample_pose_func=sample_initial_pose,
                                         min_distance=0.01,
                                         max_distance=0.2)

# BVH tree used for camera obstacle checks
bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(placed_objects)


# Now load all textures of the materials that were assigned to at least one object
bproc.loader.load_ccmaterials(args.cc_material_path, fill_used_empty_materials=True)

poses = 0
while poses < 2:

    # Sample location
    location = bproc.sampler.shell(center = [0, 0, 0],
                            radius_min = 0.61,
                            radius_max = 1.24,
                            elevation_min = 5,
                            elevation_max = 89,
                            uniform_volume = False)
    # Determine point of interest in scene as the object closest to the mean of a subset of objects
    poi = bproc.object.compute_poi(np.random.choice(placed_objects, size=10))
    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)

    # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
    if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
        # Persist camera pose

        # Sample five camera poses
        distance_range = np.linspace(0, 1, 5)
        for i in distance_range[:3]:
            # Compute linear interpolation 
            new_location = lerp3D(location, poi, i)
            print ("Position interst , ", poi , new_location)
            # Add homog cam pose based on location an rotation
            cam2world_matrix = bproc.math.build_transformation_mat(new_location, rotation_matrix)
            bproc.camera.add_camera_pose(cam2world_matrix)

        poses += 1  
# activate normal and depth rendering
#bproc.renderer.enable_normals_output()
#bproc.renderer.enable_depth_output(activate_antialiasing=False)

# render the whole pipeline
data = bproc.renderer.render()



# Render segmentation masks (per class and per instance)
data.update(bproc.renderer.render_segmap(map_by=["class"]))
#seg_data = bproc.renderer.render_segmap(map_by=["instance", "class", "name"])


# write the data to a .hdf5 container
bproc.writer.write_hdf5(args.output_dir, data, append_to_existing_output=True)
#bproc.writer.write_hdf5(args.output_dir, data)


# write the animations into .gif files
#bproc.writer.write_gif_animation(args.output_dir, data)


#bproc.writer.write_bop(os.path.join(args.output_dir, 'bop_data'),
#                       depths = data["depth"],
#                       colors = data["colors"],
#                       color_file_format = "PNG",
#                       m2mm = True,
#                       ignore_dist_thres = 10)
