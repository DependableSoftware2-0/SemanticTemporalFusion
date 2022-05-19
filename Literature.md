# Literature

| Paper  | Datasets  | Pose  |   |   |
|---|---|---|---|---|
| 1  |Standford 2D-3D-Semantic   |   |   |   |
|   | SceneNet RGB-D  |   |   |   |
|   | NyuV2  |   |   |   |


Multimodal fusion vs Multi-view fusion

Multi-view with pose vs multi-view stereo fusion 

### Why it is important
In [4] camera pose is used to improve depth prediction by creating a new data augmentation methodology and training the depth estimator with different novel poses while training.  In our method we dont retrain the model but use the pose, uncertainty prediction and the uncertainty fusion to improve the output.
It also provides another method of encoding each camera pose as an image and providing during training and testing. 

## References 

* [1] Ma, Lingni et al. “Multi-view deep learning for consistent semantic mapping with RGB-D cameras.” 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (2017): 598-605.
* [2] Zhi, Shuaifeng et al. “SceneCode: Monocular Dense Semantic Reconstruction Using Learned Encoded Scene Representations.” 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2019): 11768-11777.
* [3] H. Blum, A. Gawel, R. Siegwart and C. Cadena, "Modular Sensor Fusion for Semantic Segmentation," 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2018, pp. 3670-3677, doi: 10.1109/IROS.2018.8593786.
* [4] Zhao Y, Kong S, Fowlkes C. Camera pose matters: Improving depth prediction by mitigating pose distribution bias. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2021 (pp. 15759-15768).
* [5] 

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

### Outdoor
* [The SYNTHIA dataset (sythetic)] G. Ros, L. Sellart, J. Materzynska, D. Vazquez and A. M. Lopez, "The SYNTHIA dataset: A large collection of synthetic images for semantic segmentation of urban scenes", 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3234-3243, Jun. 2016.
* [Citiscpes Dataset (Real)]M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, et al., "The cityscapes dataset for semantic urban scene understanding", 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3213-3223, Jun. 2016.
