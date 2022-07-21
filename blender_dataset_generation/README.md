

# BlenderProc dataset generation


### Commands to generate the images.
* for i in {1..100}; do blenderproc run robocup_dataset.py multiple_only_objects_robocup.blend cc_textures images_robocup/; done
* for i in {1..50}; do blenderproc run robocup_dataset.py multiple_only_objects_robocup.blend cc_textures images_robocup/; done
* for i in {1..30}; do blenderproc run robocup_dataset.py multiple_only_objects_robocup.blend cc_textures images_robocup/; done

To visualize the images 
* blenderproc vis hdf5 images_robocup/628.hdf5
* blenderproc vis hdf5 330.hdf5
* blenderproc vis hdf5 {330, 331,332,333}.hdf5

