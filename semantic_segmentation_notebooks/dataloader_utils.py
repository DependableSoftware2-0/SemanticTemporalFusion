import os
import torch
import shutil
import numpy as np

from PIL import Image, Resampling
from tqdm import tqdm
from urllib.request import urlretrieve
import h5py

import json
from pytransform3d.transform_manager import TransformManager


class RoboCupDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.files_directory = self.root

        self.filenames = self._read_split()  # read train/valid/test splits
        
        #changing self classes also update classes below in pre process
        self.classes = {0:0,  40:1, 88:2, 112:3, 136:2, 184:4, 208:5, 232:4} # naming both plastic tube and plastic
        self.blender_names = {0:'background',  40:'small allu', 88:'plastic tube', 112:'large allu', 136:'v plastic', 184:'large nut', 208:'bolt', 232:'small nut'}
        self.class_names = {0:'background',  1:'small allu', 2:'plastic tube', 3:'large allu', 4:'large nut', 5:'bolt'}

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        file_path = os.path.join(self.files_directory, filename)
        
        with h5py.File(file_path, 'r') as data: 
            image = np.array(data['colors'])
            mask = np.array(data['class_segmaps'])
            
        
        #trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(mask)

        sample = dict(image=image, mask=mask)
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        classes = {0:0,  40:1, 88:2, 112:3, 136:2, 184:4, 208:5, 232:4} # naming both plastic tube and plastic

        mask = mask.astype(np.float32)
        #Remove this and do it in before loading data and save as h5p5
        for c in classes:
            mask[mask==c] = classes[c]

        return mask

    def _read_split(self):
        
        filenames = [f for f in os.listdir(self.files_directory) if os.path.isfile(os.path.join(self.files_directory, f))]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames


class SimpleRoboCupDataset(RoboCupDataset):
    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)

        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.LINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Resampling.NEAREST))

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)

        return sample


class RoboCupDatasetWithPose(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test", "two_sequence", "three_sequence"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.files_directory = self.root

        self.filenames = self._read_split()  # read train/valid/test splits
        
        #changing self classes also update classes below in pre process
        self.classes = {0:0,  40:1, 88:2, 112:3, 136:2, 184:4, 208:5, 232:4} # naming both plastic tube and plastic
        self.blender_names = {0:'background',  40:'small allu', 88:'plastic tube', 112:'large allu', 136:'v plastic', 184:'large nut', 208:'bolt', 232:'small nut'}
        self.class_names = {0:'background',  1:'small allu', 2:'plastic tube', 3:'large allu', 4:'large nut', 5:'bolt'}
        #Saving Camera Matrix K and its inverse
        if self.mode ==  "train" or self.mode == "valid": 
            filename = self.filenames[0]
        elif self.mode == "two_sequence" or self.mode == "three_sequence":
            filename = self.filenames[0][0]
        print ("filename ", filename)
        file_path = os.path.join(self.files_directory, filename)

        with h5py.File(file_path, 'r') as data:
            transformation_matrix = json.loads(np.array(data['camera_pose']).item().decode()) 
            
        self.K = np.array(transformation_matrix['cam_K']).reshape(3,3)
        self.Kinv= np.linalg.inv(self.K)
        
        self.height = 512 #ToDo replace this with self.K height value. Curently for blenderproc dataset there is a mismatch
        self.width = 512 #ToDo replace this with self.K width value. Curently for blenderproc dataset there is a mismatch

        

    def __len__(self):
        return len(self.filenames)
    
    def print_filenames(self):
        return self.filenames

    
    @staticmethod
    def _give_rotation_translation(old_transformation_matrix,
                                 new_transformation_matrix):
        R = np.array(old_transformation_matrix['cam_R_w2c']).reshape(3,3)
        T = np.array(old_transformation_matrix['cam_t_w2c']).reshape(3,1)
        old_pose = np.hstack((R,T))
        old_pose = np.vstack((old_pose,[0., 0., 0., 1.]))
        
        R = np.array(new_transformation_matrix['cam_R_w2c']).reshape(3,3)
        T = np.array(new_transformation_matrix['cam_t_w2c']).reshape(3,1)
        new_pose = np.hstack((R,T))
        new_pose = np.vstack((new_pose,[0., 0., 0., 1.]))
        
        tm = TransformManager()
        tm.add_transform("world", "old_pose", old_pose)
        tm.add_transform("world", "new_pose", new_pose)
        old2new = tm.get_transform("old_pose", "new_pose")
        R = old2new[:3,:3]
        T = old2new[:3,3].reshape(3,1)
        return R, T
    
    def __getitem__(self, idx):

        if self.mode == "train" or self.mode == "valid": 
            filename = self.filenames[idx]
            file_path = os.path.join(self.files_directory, filename)

            with h5py.File(file_path, 'r') as data: 
                image = np.array(data['colors'])
                mask = np.array(data['class_segmaps'])


            #trimap = np.array(Image.open(mask_path))
            mask = self._preprocess_mask(mask)

            sample = dict(image=image, mask=mask)
            if self.transform is not None:
                sample = self.transform(**sample)
        elif self.mode == "two_sequence" or self.mode == "three_sequence":
            #For sequence
            # send all the n images
            # only last frame maske
            # all the poses 
            seq_filenames = self.filenames[idx]
            
            images = []
            maskes = []
            depths = []
            transformation_matrices = []
            for filename in seq_filenames:
                file_path = os.path.join(self.files_directory, filename)

                with h5py.File(file_path, 'r') as data: 
                    images.append(np.array(data['colors']))
                    maskes.append( self._preprocess_mask(np.array(data['class_segmaps'])))
                    depths.append(np.array(data['depth']))
                    transformation_matrices.append(json.loads(np.array(data['camera_pose']).item().decode()))

                #trimap = np.array(Image.open(mask_path))
                #maskes.append( self._preprocess_mask(mask))

            
            if self.mode == "two_sequence":
                try :
                    rotation_old_to_new_camera_frame, \
                    translation_old_to_new_camera_frame = self._give_rotation_translation(transformation_matrices[0], 
                                                                                      transformation_matrices[1])
                except: 
                    print ("Transformation error in sequence : ", seq_filenames)
                    return None
                sample = dict(image0 = images[0],
                              depth0 = depths[0],
                              image1 = images[1], 
                              mask0 = maskes[0], 
                              mask1 = maskes[1], 
                              rotation_old_to_new_camera_frame = rotation_old_to_new_camera_frame,
                              translation_old_to_new_camera_frame = translation_old_to_new_camera_frame)
            elif self.mode == "three_sequence":
                try :
                    rotation_0_to_new_1_frame, \
                    translation_0_to_1_camera_frame = self._give_rotation_translation(transformation_matrices[0], 
                                                                                      transformation_matrices[1])
                    rotation_1_to_new_2_frame, \
                    translation_1_to_2_camera_frame = self._give_rotation_translation(transformation_matrices[1], 
                                                                                      transformation_matrices[2])
                except: 
                    print ("Transformation error in sequence : ", seq_filenames)
                    return None
                sample = dict(image0 = images[0],
                              depth0 = depths[0],
                              image1 = images[1],
                              depth1 = depths[1],
                              image2 = images[2],
                              mask0 = maskes[0], 
                              mask1 = maskes[1],
                              mask2 = maskes[2],
                              rotation_0_to_new_1_frame = rotation_0_to_new_1_frame,
                              translation_0_to_1_camera_frame = translation_0_to_1_camera_frame, 
                              rotation_1_to_new_2_frame = rotation_1_to_new_2_frame,
                              translation_1_to_2_camera_frame=translation_1_to_2_camera_frame)
                raise NotImplemented

        else:
            raise NotImplementedError("check mode variable while instantiating")

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        classes = {0:0,  40:1, 88:2, 112:3, 136:2, 184:4, 208:5, 232:4} # naming both plastic tube and plastic

        mask = mask.astype(np.float32)
        #Remove this and do it in before loading data and save as h5p5
        for c in classes:
            mask[mask==c] = classes[c]

        return mask
    
    
    
    
    def _read_split(self):
        def _atoi(text):
            return int(text) if text.isdigit() else text

        def _natural_keys(text):
            '''
            alist.sort(key=natural_keys) sorts in human order
            http://nedbatchelder.com/blog/200712/human_sorting.html
            (See Toothy's implementation in the comments)
            '''
            return [ _atoi(c) for c in  text.split('.') ]

        
        filenames = [f for f in os.listdir(self.files_directory) if os.path.isfile(os.path.join(self.files_directory, f))]
        print ("Found {:d} files in the folder".format(len(filenames)))
        
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        elif self.mode == "three_sequence": 
            #Sorting files as per the numbering
            # Assumption that files are numbered '0.hdf5', '1.hdf5', '10.hdf5', '
            # Assumption there is 3 images in sequnce starting from 0 
            filenames.sort(key=_natural_keys)
            
            filenames = [[filenames[i+n] for n in range(3)] 
                         for i, x in enumerate(filenames) if i % 3 == 0]
        elif self.mode == "two_sequence": 
            #Sorting files as per the numbering
            # Assumption that files are numbered '0.hdf5', '1.hdf5', '10.hdf5', '
            filenames.sort(key=_natural_keys)
            
            # 3 camera poses can be divided into 2 frames each
            # for example 1 - 2- 3 images
            # can be divided in to [1,2] and [2,3] sequence
            filenames = [[filenames[i+j+n] for n in range(2)] 
                         for i, x in enumerate(filenames) if i % 3 == 0
                         for j in range(2)] 
                        
            print( "Found {:d} two image sequences ".format(len(filenames)))
        return filenames


class SequenceRoboCupDataset(RoboCupDatasetWithPose):
    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)
        
        
        # resize images
        # The robocup dataset intrinsic parameter has image size of 512 so fix to 512
        #image0 = np.array(Image.fromarray(sample["image0"]).resize((512, 512), Image.LINEAR))
        #image1 = np.array(Image.fromarray(sample["image1"]).resize((512, 512), Image.LINEAR))
        #depth0 = np.array(Image.fromarray(sample["depth0"]).resize((512, 512), Image.LINEAR))
        #mask0 = np.array(Image.fromarray(sample["mask0"]).resize((512, 512), Image.NEAREST))
        #mask1 = np.array(Image.fromarray(sample["mask1"]).resize((512, 512), Image.NEAREST))


        # convert to other format HWC -> CHW
        sample["image0"] = np.moveaxis(sample["image0"], -1, 0)
        sample["image1"] = np.moveaxis(sample["image1"], -1, 0)
        sample["mask0"] = np.expand_dims(sample["mask0"], 0)
        sample["mask1"] = np.expand_dims(sample["mask1"], 0)
        #ssample["depth0"] = depth0

        return sample

