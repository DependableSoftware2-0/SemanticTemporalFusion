
# Questions 
### Q4 Why should the community care?

* DNN performance is directly related to its capacity (cite papers with the pareto curve)
* Bigger the size of the network better the performance of the network.
* Autonomous systems especially robots are embedded devices and have limited resources.
* The DNN which are deployed on autonomous systems have constraints of the resources from the system.
* The limited resources have a limit on the size of the network and hence will produce sub-par performance.
* As the DNN outputs are used for making autonomous decisions the performance degradation in the DNN will lead
to catastropic failure in the system.
* Different methodology has been proposed to improve the performance of DNN in constrained enviroments:
   * Architecture modifications ?
   * Training modifications
   * Loss function modifications
   * information Fusion
   * output fusion 

![How deeplearning used for video ](https://static.wixstatic.com/media/226638_85aaca247fe14b4f8222ccba3e20e135~mv2.png/v1/fill/w_512,h_196,al_c,q_85,enc_auto/frames-fusion.png) https://www.ridgerun.com/video-based-ai
* Even though atonomnous systems are constrined by the hardware limitations they have an advantage that they
are embodied agents.
* The embodiement of the agent can be used to overcome the limitation.
* The embodiement of the agent has an advantage that the agents can move around and collect more information
Thus they dont need to make decision based on single source of data.
* The different sources of data now can fused to make a 
* In this work, we focus on the task of semantic segmentation with DNN. Our goal is to improve the performance of 
semantic segmentation with minimum increase of computation by exploiting the embodiment and the redundancy of information.
* We propose to use Multi-View geometry and semantic fusion
* Our goal is to 


### Reliability based introduction 
* DNN outputs are unreliable and cannot be trusted to make decisions in autonomous systems.
* Different methodology has been proposed to increase the reliability of the DNN including
   * Architecture modifications ?
   * Training modifications
   * Loss function modifications
   * Uncertainty estimation in addition to the output
   * Semantic Fusion 
* These methods can be classified in 2 categories: DNN internal improvements and DNN external improvements.
* Semantic fusion is an external reliability improvement methodology.
* All fusion methods improve reliability and make the output fault-tolerant of the intermittent faults in the output.
* In this work, we investigate does external semantic fusion with pose information and uncertain DNN output improves prediction/reliability.
* However, semantic fusion comes at higher computation cost. Multi-modal needs multiple DNN feature extractor.
* Higher computation is a problem for embedded applications like autonomous car, robots and in general environment.
* In this work, we focus on semantic fusion with minimum computation.
* This is achieved by muti-view geometry and exploiting the redundancy in temporal multi-view geometry.
* We also focus on the fusion of uncertain information 

### Q1 What did the community know before you did whatever you did?
* Semantic fusion is better ?
* Different types of fusion:
   * Multi-modal fusion
   * Multi-view fusion
       * Multiview batch vs multiview incremental semantic fusion 
       * Multiview early vs multiview latent vs multiview late fusion
* Different types of fusion based on where we do fusion [Survey paper on multimodal Fusion](https://hal-univ-evry.archives-ouvertes.fr/hal-02963619/file/Deep_Multimodal_Fusion_for_Semantic_Image_Segmentation__A_Survey.pdf):
   * Early fusion (data fusion)
   * Latent Fusion
   * Late fusion (decision fusion)
     * [Statistical fusion Baseline paper](https://arxiv.org/abs/1807.11249) 
     * [6D pose fusion ](https://ieeexplore.ieee.org/document/9670642)
   * Hybrid fusion 
* Type of fusion
   * Exact evidence fusion
   * Uncertain evidence fusion, [Improving Object Pose Estimation by Fusion With a Multimodal Prior – Utilizing Uncertainty](https://ieeexplore.ieee.org/document/9670642)
* Multimodal vs multiview semantic fusion
* Image segmentation vs Video segmentation 
   * [Baseline video segmentation with fusion - Accel](https://arxiv.org/pdf/1807.06667.pdf)

* efficiency-accuracy trade off

### Q2 What are the new things you learned after you did whatever you did?

### Q3 What exactly did you do?

* Exploiting temporal redundancy in the data to get ~robust and~ reliable output with minimum computation.
* Semantic fusion for the task of object picking with a manipulator having a tooltip camera.
* Incremental Semantic fusion
* Benchmarking uncertainty estimation methods  for the task of semantic fusion.
* Benchmarking uncertainty fusion methods for the task of semantic fusion. 

### Q5 What does the community still not know?



# Paper 

## Introduction
– Overview of Q1, Q2, Q3; plus
– Why should the community care?

## Related Work
– Q1
•
•
•
## Problem Formulation
## Algorithm/Methodology
## Evaluation
– Q2 & Q3
•
## Conclusion and Future Work
– Overview of Q1, Q2, and Q3; plus
– What does the community still not know?


