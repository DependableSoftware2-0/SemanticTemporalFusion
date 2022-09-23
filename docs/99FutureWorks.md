# Future works

1. CRF as RNN for fusion 
  - https://github.com/sadeepj/crfasrnn_pytorch/blob/master/crfasrnn/crfrnn.py
  - As per the code the n_iteration=5 can be replaced with 2 with our previous frame and next frame formulation
  - [Conditional Random Fields as Recurrent Neural Networks](https://arxiv.org/pdf/1502.03240.pdf)
  - Very important paper to understand CRF, how it can be converted to a rnn and the different methods to do it
  - Another implementation : https://github.com/migonch/crfseg/blob/master/crfseg/model.py
  - Waht is different about man field CRF
  - Convolution CRF
      - https://github.com/MarvinTeichmann/ConvCRF/blob/master/convcrf/convcrf.py
  
2. MRF conv for fusion 
  - Better than CRF as it requires iteration MRF conv doesnt require iterations 
  - https://liuziwei7.github.io/projects/DPN
  - No code available 
  - 

3. Other Fusions
  - **subjective logiv fusion**
  - https://github.com/joseoliveirajr/subjective-logic-library/blob/master/%5B2022-01-20%5D.ipynb
  - **Nosiy-or fusion** 
  - https://nikosuenderhauf.github.io/roboticvisionchallenges/assets/papers/IROS19/tian.pdf
  - Fuses multimodal uncertainty with noisy-or fusion 
  -   [outputs = 1 - (1 - mean["rgb"]) * (1 - mean["d"]) #[batch,11,512,512]](https://github.com/GT-RIPL/UNO-IC/blob/6a95f2c6bc52ad80bfb1da53fd046a3d4db310d0/segmentation/ptsemseg/core/core.py#L37)




4. Dataset
  - Kitti has odometry/ego motion
  - BOP datasets - some have poses, all virtual dataset has poses
  