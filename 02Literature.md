# Literature

### Why it is important
In [4] camera pose is used to improve depth prediction by creating a new data augmentation methodology and training the depth estimator with different novel poses while training.  In our method we dont retrain the model but use the pose, uncertainty prediction and the uncertainty fusion to improve the output.
It also provides another method of encoding each camera pose as an image and providing during training and testing. 
It also has provided a datloader for ScanNet and InteirorNet. 

In [5] they have used 2 deep learning branches and it fuses internal based on opticalflow. Its a video segmentation task. Tested on cityscapes and camvid datasets. It asks the accuracy vs efficency tradeoff. 

In [6] for the task of 6D pose estimation they have used fusion of the uncertainty in the pose estimation for improving the pose. This is a output layer fusion method. 

In [7] is our multiview - without poses, use both images while training, wihle testing . Uncertainty estimation, Dirichlet fusion . 



### Other comparison
Multimodal fusion vs Multi-view fusion

Multi-view with pose vs multi-view stereo fusion 



## References 

* [1] Ma, Lingni et al. “Multi-view deep learning for consistent semantic mapping with RGB-D cameras.” 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (2017): 598-605.
* [2] Zhi, Shuaifeng et al. “SceneCode: Monocular Dense Semantic Reconstruction Using Learned Encoded Scene Representations.” 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2019): 11768-11777.
* [3] H. Blum, A. Gawel, R. Siegwart and C. Cadena, "Modular Sensor Fusion for Semantic Segmentation," 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2018, pp. 3670-3677, doi: 10.1109/IROS.2018.8593786.
* [4] Zhao Y, Kong S, Fowlkes C. Camera pose matters: Improving depth prediction by mitigating pose distribution bias. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2021 (pp. 15759-15768).
* [5] Jain S, Wang X, Gonzalez JE. Accel: A corrective fusion network for efficient semantic segmentation on video. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2019 (pp. 8866-8875).
* [6] J. Richter-Klug, P. Mania, G. Kazhoyan, M. Beetz and U. Frese, "Improving Object Pose Estimation by Fusion With a Multimodal Prior – Utilizing Uncertainty-Based CNN Pipelines for Robotics," in IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 2282-2288, April 2022, doi: 10.1109/LRA.2022.3140450.
* [7]  Zongbo Han, Changqing Zhang, Huazhu Fu, Joey Tianyi Zhou "Trusted Multi-View Classification" ICLR 2021


## Datasets
| Paper  | Datasets  | Pose  |   |   |
|---|---|---|---|---|
| 1  |Standford 2D-3D-Semantic   |   |   |   |
|   | SceneNet RGB-D  |   |   |   |
|   | NyuV2  |   |   |   |
|   | InteriorNet  |   |   |   |
|   | Scannet  |   |   |   |

https://sites.google.com/view/awesome-slam-datasets/
copu of the sheet - https://docs.google.com/spreadsheets/d/1yTKz4F-4vhSpPWc9nDrm1ZvJgln0jLhInFbBdeQOZeQ/edit#gid=1823356432



## Datasets

* [Standford 2D-3D-Semantic] Iro Armeni, Alexander Sax, Amir R. Zamir, and Silvio Savarese. Joint 2D-3D-Semantic Data for Indoor Scene Understanding. arXiv preprint arXiv:1702.01105, 2017.
* [SceneNet RGB-D] John McCormac, Ankur Handa, Stefan Leutenegger, and 
Andrew J. Davison. SceneNet RGB-D: Can 5M Synthetic
Images Beat Generic ImageNet Pre-training on Indoor Seg-
mentation? In Proceedings of the International Conference
on Computer Vision (ICCV), 2017
  * Python code to dowload -> https://github.com/angeladai/3DMV 
* [NyuV2] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob
Fergus. Indoor Segmentation and Support Inference from
RGBD Images. In Proceedings of the European Conference
on Computer Vision (ECCV), 2012.
* [InteriorNet] an end-to-end pipeline to render an RGB-D-inertial benchmark for large scale interior scene understanding and mapping. Our dataset contains 20M images created by pipeline. 
    - [pytroch dataloader ]( https://github.com/yzhao520/CPP/tree/master/dataloader )
* [Scannet] 
    - [pytroch dataloader ]( https://github.com/yzhao520/CPP/tree/master/dataloader )

https://repository.up.ac.za/handle/2263/83957

RailEnV-PASMVS: a dataset for multi-view stereopsis training and reconstruction applications


### Outdoor
* [The SYNTHIA dataset (synthetic)] G. Ros, L. Sellart, J. Materzynska, D. Vazquez and A. M. Lopez, "The SYNTHIA dataset: A large collection of synthetic images for semantic segmentation of urban scenes", 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3234-3243, Jun. 2016.
* [Citiscpes Dataset (Real)]M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, et al., "The cityscapes dataset for semantic urban scene understanding", 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3213-3223, Jun. 2016. 
  * https://github.com/mcordts/cityscapesScripts
* [Replica Dataset](https://github.com/facebookresearch/Replica-Dataset)


#### Google searcher 
1. multi view + extrinsic +  dataset
2. multi view fusion + reliability
