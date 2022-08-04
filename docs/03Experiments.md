## Experiments

### RQ1 Does external Semantic fusion with uncertainty improves prediction?
* Do adding uncertain evidence reduce uncertainty?
* Different kinds of [evidence](https://cse.sc.edu/~mgv/csce582sp21/links/mradEtAl_APIN2015.pdf)
    * Hard evidence
    * Virtual evidence
    * Likelihood evidence
    * soft evidence
* Not all unreliable evidence follow the principle of adding reduces uncertainty.
* So the first question to answer is the different uncertainty coming from the different UE methods can these be summed ?



### RQ1 Comparison of output semantic fusion 

### Does semantic fusion help in computation reduction?

In this experiment we are looking at the tradeoff between accuracy and computation.

### efficiency-accuracy trade off



### Datasets
* Want datasets with camera poses
* Blender dataset for Scannet using Blnderproc 
* Cutom blender dataset with robocup using blenderproc
* Blender dataset for YCB using blenderproc
* Citiscapes dataset - from video segmentation baseline Accel
    * https://github.com/mcordts/cityscapesScripts


### Status
1. Datasets:
    - German traffic sign dataset
    - Robocup blender 
    - Virtual Kitti 
2. Uncertainty
    - Evidential loss
3. Uncertainty fusion
    - Dirchlet fusion
    - Bayesian constructive fusion
    - Dirchlet destructive fusion
    - Sum fusion
    - Mean fusion

### Experiments
1. GTSRB dataset
    - Accuracy 93%
    - Fusion increase minorly but for all
    - When accuracy was less the fusion was not improving
    - Uncertainty increases for both correct and worng prediction 

2. Robocup dataset
3. Virtual kitti
    - Mobilenet small architecture
    - accuracy is around 70 %
    - Fusion reduces the accuracy
    - 
