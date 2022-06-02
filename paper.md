
# Questions 
### Q4 Why should the community care?

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
* However, they all come at higher computation cost. Multi-modal needs multiple DNN feature extractor.
* Higher computation problem for embedded applications like autnomous car, robots and in general environment.
* In this work, we focus on semantic fusion with minimum computation.
* This is achieved by muti-view geometry and exploiting the redundancy in temporal multi-view geometry.

### Q1 What did the community know before you did whatever you did?
* Semantic fusion is better ?
* Different types of fusion:
   * Multi-modal fusion
   * Multi-view fusion
* Different types of fusion based on where we do fusion [Survey paper on multimodal Fusion](https://hal-univ-evry.archives-ouvertes.fr/hal-02963619/file/Deep_Multimodal_Fusion_for_Semantic_Image_Segmentation__A_Survey.pdf):
   * Early fusion (data fusion)
   * Latent Fusion
   * Late fusion (decision fusion)
     * [Statistical fusion Baseline paper](https://arxiv.org/abs/1807.11249) 
     * [6D pose fusion ]()
   * Hybrid fusion 
* Multimodal vs multiview semantic fusion
* Multiview batch vs multiview incremental semantic fusion 
* Multiview early vs multiview latent vs multiview late fusion
* Image segmentation vs Video segmentation 
   * [Baseline video segmentation with fusion - Accel](https://arxiv.org/pdf/1807.06667.pdf)

* efficiency-accuracy trade off

### Q2 What are the new things you learned after you did whatever you did?

### Q3 What exactly did you do?

* Exploiting temporal redundancy in the data to get ~robust and~ reliable output with minimum computation.
* Semantic fusion for the task of object picking with a manipulator having a tooltip camera.
* Incremental Semantic fusion
* Benchmarking uncertainty estimation methods with differnt uncertainty fusion method for the task of semantic fusion

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


