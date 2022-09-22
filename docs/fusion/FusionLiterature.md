# Belief fusion


1. Dempster Shafer fusion 
    - [A Python library for performing calculations in the Dempster-Shafer theory of evidence.](https://github.com/reineking/pyds)
    - Support for normalized as well as unnormalized belief functions
    - Different Monte-Carlo algorithms for combining belief functions
    - Various methods related to the generalized Bayesian theorem
    - Measures of uncertainty
    - Methods for constructing belief functions from data
    - [Example of fusion with IRIS Dataset](http://bennycheung.github.io/dempster-shafer-theory-for-classification)
    - [https://www.aaai.org/AAAI22Papers/AAAI-11669.LiuW.pdf](Dirichlet fusion paper with code we are using)
        - Uses same evidetial loss function
        - with multiview datasets
        - g0od future work can be to show how fusion network performs better with these
        - Theory explanation and evidential explanation
        - https://github.com/hanmenghan/TMC/tree/main/TMC%20ICLR 
        - They loss function is combined (individual per frame loss + fusion o/p loss)
1a. Yagger fusion (Future work)
    - [https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9341308&casa_token=EfSK0q4lJB8AAAAA:0KPdDpxxeMnFkvFeessfV-gPnDp0cCvYkY5WB6ypFhnBE52P5VYuRFgNeHnA_jeQeYe0FIVgn80&tag=1](Yager and Damspster combining)

2. Fusion with the efprob we had 
    - Use the Confusion matrix for Prior and 
    - efrob with constructive and destrutive update
    - uses confusion matrix

3. Creating a network with fusion
    - the projected tensor and current prediction tensor 
    - can be fused with a convolution 

4. LSTM fusion (Future work)
    - Convd LSTM fusion 
    - https://github.com/sladewinter/ConvLSTM/blob/master/ConvLSTM.py
    - 
5. Subjective logic fusion
    - https://github.com/joseoliveirajr/subjective-logic-library/blob/master/%5B2022-01-20%5D.ipynb
    - Has implemented 3 fusion : cumulative, averaging and weighted 
    - from SubjectiveLogic.BeliefFusion import cumulative_fusion, averaging_fusion, weighted_fusion7
    - https://confcats_isif.s3.amazonaws.com/web-files/journals/entries/Categories%20of%20belief%20fusion.pdf
