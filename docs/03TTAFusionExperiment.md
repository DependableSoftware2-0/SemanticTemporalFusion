# Test time Augmenttaion 

TTA is used for aleatoric uncertainty estimation[2], improves "improves the predictive performance"[1] and  robustness[?].

Mostly TTA are trained with cross entropy loss and fused by taking average [1] .

Can we train with evidential loss and fuse by the different mthods and the proposed ?



## Reference

1. "Greedy Policy Search: A Simple Baseline for Learnable Test-Time Augmentation" 
    (UAI 2020) by Dmitry Molchanov, Alexander Lyzhov, Yuliya Molchanova, Arsenii Ashukha, Dmitry Vetrov.
    * https://github.com/SamsungLabs/gps-augment

2. Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation with convolutional neural networks
Guotai Wang, Wenqi Li, Michael Aertsen, Jan Deprest, Sebastien Ourselin, Tom Vercauteren
