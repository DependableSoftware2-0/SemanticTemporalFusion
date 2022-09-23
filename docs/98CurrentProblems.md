# Issues/Problems


1. The baseline is single frame.
    - The baseline cannot be single frame
    - Its obvious knowledge, fusion should work better than single frame
    - We need a strong baseline.
    - some other fusion method who claim to perform better on a task.

2. Real world dataset or task

3. Dataset is limited 
    - Semantic segmentation all papers are eith cityscapes, kitti or scannet
    - Need to do other dataset
    - For example = BOP datasets
4. Fusion method is not extensive
    - other subjective fusion
    - RNN, LSTM, 
    - Bayesian fusion
        * Multiplication likelihood formula
        * using the prior from confusion matrix
5. One usecase one message
    - Create new dataset and task
    - Given camera extrinsic, intrinsic and segment a video frame with fixed model size

6. Why fusion with evidential loss is not converging ?
    - Currently we are using cross entropy loss to fuse the 2 uncertainty. 

7. Uncertainty estimation methods?
    - other deterministic uncertainty estimation methods
    - Laplace approximation BDD
    - evidential, dropout
    - other deterministric uncertatiny estimation methods
   
8. Problem statments ? 
