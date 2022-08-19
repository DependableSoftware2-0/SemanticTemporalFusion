# Experiments Summary
* Thursday Aug 18
    - Debugging Robocup above 95 accuracy fusion degradation problem
    - While training do extra validation and check regions when fusion is bad
    - Exp 483333 = train robocup effificentnet 130 epochs with validation called every 10 epochs. Compare val and fusion output while training.
    - Fusion with evidential loss doesnt train 1D and 2D
    - Fusion with crossentropy loss 1D is not converging. 2D is fusing  - 483357
* Wednesday Aug 17
    - Training vkitti with efficient net . - Result bug after 9 hrs
    - Robocup Efficientnet - 130 epochs
    - Fusion behaving strange - 1 class is goes to 0% after fusion with evidential loss for fusion 
    - Mayve the case that after 95% accuracy the fusion deteriots

|       =        | val_1  | sum  | mean | ds_combine | 1D | 2D  |
|----------------|--------|------|------|------------|----|-----|
| Frequency IoU  | **0.9794** | 0.9740 | 0.9740 | 0.9741 |0.9341  | 0.9215 |
| Mean IoU       | **0.8487** | 0.8342 | 0.8342 | 0.8350 |0.5007  | 0.6825 |
| Pixel Accuracy | **0.9875** | 0.9858 | 0.9858 | 0.9859 | 0.9645 | 0.9598 |
| IoU            | **0.9810** | 0.9778 | 0.9778 | 0.9779 | 0.9393 | 0.8885 |

    - Training vkitti efficientnet 100 epochs with 10 epcoh for fusion with crossentropy - 483325
        - 1D gives better results
        - 2D fusion is less ??????  should we train more
        

|       =        | val_1  | sum  | mean | ds_combine | 1D | 2D  |
|----------------|--------|------|------|------------|----|-----|
| Frequency IoU  | 0.8986 | 0.8752 | 0.8752 | 0.8772 | **0.9147** | 0.8805 |
| Mean IoU       | 0.8938 | 0.8683 | 0.8683 | 0.8702 | **0.9099** | 0.8763 |
| Pixel Accuracy | 0.9426 | 0.9298 | 0.9298 | 0.9859 | **0.9536** | 0.9324 |
| IoU            | 0.9336 | 0.9223 | 0.9223 | 0.9779 | **0.9447** | 0.9338 |

    - Increasing vkitti fusion training time for 1D and 2D fusion - run 483330 : to check whether 2d fusion improves
    - Result : No Fusion detiorates for both fusion After 20 epochs 
    - Why we cant reproduce result from above

|       =        | val_1  | 1D | 2D  |
|----------------|--------|----|-----|
| Frequency IoU  | 0.8986 | 0.8236 | 0.8639 |
| Mean IoU       | 0.8938 | 0.8450 | 0.7783 |
| Pixel Accuracy | 0.9426 | 0.8925 | 0.9213 |
| IoU            | 0.9336 | 0.7948 | 0.9353 |

    
* Tuesday Aug 16
    - Single script for training(model,1D and 2D) and validation and plotting
    - Robocup Training and validation done
    - 1D has problem 
    - Plotting of prediction - looks good


* Monday Aug 15
    - Single script for training model,1D,2D and validating
* Friday Aug 13
    - Robocup Training solved
    - Result Robocup only single image training . Validation on sequence todo
 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     Validate metric           DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   FrequencyIoU/valid       0.7870792746543884
    valid/dataset_iou       0.7902564406394958
     valid/dice_los         0.20299935340881348
  valid/evidential_loss     0.8104310631752014
   valid/per_image_iou      0.7937226295471191
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


    - Vkitti with Regnext timm 
    - Results: Something worng with 1D fusion debug
    
|       =        | val_1  | sum  | mean | ds_combine | 1D | 2D  |
|----------------|--------|------|------|------------|----|-----|
| Frequency IoU  | 0.8478 | 0.8344 | 0.8344 | 0.8356 |0.7515  | **0.8518** |
| Mean IoU       | 0.8350 | 0.8211 | 0.8211 | 0.8223 |0.7374  | **0.8383** |
| Pixel Accuracy | 0.9113 | 0.9050 | 0.9050 | 0.9058 | 0.8409 | **0.9165** |
| IoU            | **0.8943** | 0.8861 | 0.8861 | 0.8876 | 0.5822 | 0.8932 |

|       =     | Pixel Accuracy  | IOU per class | Class Pixel Accuracy |
|-------------|-----------------|---------------|----------------------|
| Single Image | 0.8980  |  [0.6450, 0.8918, 0.9494, 0.8469, 0.7718, 0.9321, 0.8079] | [0.8848, 0.9225, 0.9595, 0.8804, 0.8196, 0.9557, 0.8633] |
|  DS_combine  | 0.8977   | [0.6051, 0.8868, 0.9289, 0.8497, 0.7710, 0.9234, 0.7917] | [0.7632, 0.9378, 0.9639, 0.9083, 0.8529, 0.9635, 0.8942]|
|mean          | 0.8970    |  [0.6019, 0.8856, 0.9276, 0.8490, 0.7702, 0.9225, 0.7911] | [0.7601, 0.9374, 0.9634, 0.9079, 0.8527, 0.9634, 0.8941] |
|sum           | 0.8970    |  [0.6019, 0.8856, 0.9276, 0.8490, 0.7702, 0.9225, 0.7911] | [0.7601, 0.9374, 0.9634, 0.9079, 0.8527, 0.9634, 0.8941] |
| 1D fusion    | 0.8251    | [0.4788, 0.3396, 0.9571, 0.8702, 0.7746, 0.9261, 0.8159] | [0.8339, 0.3464, 0.9729, 0.9329, 0.8255, 0.9476, 0.9166] |
| 2D fusion    | **0.9095**|  [0.6416, 0.8954, 0.9484, 0.8625, 0.7809, 0.9320, 0.8080] | [0.7733, 0.9335, 0.9647, 0.9355, 0.8776, 0.9667, 0.9155] |




    - Vkitti 1D 2D traiing  (vkitti483087.out , )

    - Vkitti training : encoders + normalization
 
|       =     | Pixel Accuracy  | IOU per class | Class Pixel Accuracy |
|-------------|-----------------|---------------|----------------------|
| Single Image | 0.8630  |  [0.5984, 0.9026, 0.9170, 0.8316, 0.7134, 0.9354, 0.7088] | [0.9128, 0.9179, 0.9263, 0.8562, 0.7422, 0.9616, 0.7239] |
|  DS_combine  | 0.8742   | [0.5860, 0.9032, 0.9108, 0.8417, 0.7348, 0.9292, 0.7387] | [0.8105, 0.9355, 0.9404, 0.8881, 0.7890, 0.9734, 0.7823] |
|mean          | 0.8737    |  [0.5840, 0.9022, 0.9102, 0.8412, 0.7345, 0.9287, 0.7384] | [0.8084, 0.9351, 0.9402, 0.8878, 0.7889, 0.9732, 0.7822] |
|sum           | 0.8737    |  [0.5840, 0.9022, 0.9102, 0.8412, 0.7345, 0.9287, 0.7384] | [0.8084, 0.9351, 0.9402, 0.8878, 0.7889, 0.9732, 0.7822] |
| 1D fusion    | 0.8620    |  [0.5442, 0.6685, **0.9341**, **0.8632**, 0.7651, **0.9376**, **0.7820**] | [0.7845, 0.6957, 0.9719, 0.9291, 0.8312, 0.9757, 0.8458] |
| 2D fusion    | **0.8982**|  [**0.6240**, **0.9102**, **0.9326**, 0.8577, **0.7711**, 0.9342, 0.7739] | [0.7776, 0.9428, 0.9604, 0.9258, 0.8758, 0.9786, 0.8262] |

        - Val 1 Mean Pixel Accuracy :    tensor(0.8630, device='cuda:0')
        - Mean Pixel Accuracy DS_combine tensor(**0.8742**, device='cuda:0')
        - Mean Pixel Accuracy mean       tensor(0.8737, device='cuda:0')
        - Mean Pixel Accuracy sum        tensor(0.8737, device='cuda:0')
        - Mean Pixel Accuracy 1D         tensor(0.8620, device='cuda:0')
        - Mean Pixel Accuracy 2D         tensor(0.8982, device='cuda:0')

        - Val 1 IoU Per class :    tensor([**0.5984**, 0.9026, **0.9170**, 0.8316, 0.7134, **0.9354**, 0.7088],
        - IoU Per class DS_combine tensor([0.5860, **0.9032**, 0.9108, **0.8417**, **0.7348**, 0.9292, **0.7387**],
        - IoU Per class mean       tensor([0.5840, 0.9022, 0.9102, 0.8412, 0.7345, 0.9287, 0.7384],
        - IoU Per class sum        tensor([0.5840, 0.9022, 0.9102, 0.8412, 0.7345, 0.9287, 0.7384],
        - IoU Per class 1d         tensor([0.5442, 0.6685, 0.9341, 0.8632, 0.7651, 0.9376, 0.7820],
        - IoU Per class 2D         tensor([0.6240, 0.9102, 0.9326, 0.8577, 0.7711, 0.9342, 0.7739],

        - Val 1 Class Pixel Accuracy :    tensor([0.9128, 0.9179, 0.9263, 0.8562, 0.7422, 0.9616, 0.7239],
        - Class Pixel Accuracy DS_combine tensor([0.8105, 0.9355, 0.9404, 0.8881, 0.7890, 0.9734, 0.7823],
        - Class Pixel Accuracy mean       tensor([0.8084, 0.9351, 0.9402, 0.8878, 0.7889, 0.9732, 0.7822],
        - Class Pixel Accuracy sum        tensor([0.8084, 0.9351, 0.9402, 0.8878, 0.7889, 0.9732, 0.7822],
        - Class Pixel Accuracy 1D         tensor([0.7845, 0.6957, 0.9719, 0.9291, 0.8312, 0.9757, 0.8458],
        - Class Pixel Accuracy 2D         tensor([0.7776, 0.9428, 0.9604, 0.9258, 0.8758, 0.9786, 0.8262],


|       =        | val_1  | sum  | mean | ds_combine | 1D | 2D  |
|----------------|--------|------|------|------------|----|-----|
| Frequency IoU  | 0.8260   | 0.8262 | 0.8262 | 0.8270 |0.8023  | **0.8456** |
| Mean IoU       |0.8010    | 0.8056 | 0.8056 | 0.8063 |0.7849  | **0.8290** |
| Pixel Accuracy | 0.8944   | 0.8973 | 0.8973 | 0.8979 | 0.8811 | **0.9121** |
| IoU            | 0.8560   | 0.8580 | 0.8580 | 0.8586 | **0.8692** | 0.8689 |

* Thursday Aug 11
    - Robocup Training
    - R
* Wednesday Aug 10
    - Validating all methods (1D, 2D convolution with cross entropy loss and 10 epochs)
    - Results - 1D convolution is low . need to verify again
    
|       =        | val_1  | sum  | mean | ds_combine | 1D | 2D  |
|----------------|--------|------|------|------------|----|-----|
| Frequency IoU  | 0.8994   | 0.8760 | 0.8760 | 0.8718 | **0.9019** | 0.9008 |
| Mean IoU       |0.8953   | 0.8698 | 0.8698 | 0.8718 | **0.9019** | 0.8965 |
| Pixel Accuracy | 0.9433   | 0.9305 | 0.9305 | 0.9318 | **0.9459** | 0.9456 |
| IoU            | 0.9298 | 0.9238 | 0.9238 | 0.9262 | **0.9410** | 0.9310 |
    
    
    - Training for 100 epochs. After 80 epochs the learning rate is high so the loss is alos high. 
    - The epoch cycle are 10, 20, 30, 40. So at 100 the learning rate will be low 
    - Next epoh will be 150 
    - Results
```
   FrequencyIoU/val_0       0.8994248509407043
   FrequencyIoU/val_1       0.8994248509407043
        val_iou/0           0.9301660060882568
        val_iou/1            0.929871141910553
```
    



* Tuesday 
    - 1D finetuning experiments 
        - Conv2D with 3x3 kernel and upsample and Conv2D 1kernel gave comparable results
        - Conv2D with 1x1 kernel and cross_entropy loss function gave slightly better resutlts
    - evidential loss
        |  metric           | value |
        |-------------------|-------|
        | val iou 0 iou     | .8539 |
        | val iou 1 iou     | .8562 |
        | val 1d fusion     | .8523 |
    - cross entropy loss
        |  metric           | value |
        |-------------------|-------|
        | val iou 0 iou     | .8539 |
        | val iou 1 iou     | .8562 |
        | val 1d fusion     | .8881 |

* Wednesday
    - Adding dice loss, frequencyiou for training, adding augmentation, for 80 peochs, also 1Dtraining for 6 epochs
    - Batch size 32 gave GPU error while we have used it continuously. maybe related to kornia augmentation. reduced to 16
    - The problem was with return 2 rad variables (dice loss ) in the return of the on_training_step 
    - Results:
        |  metric           | value |
        |-------------------|-------|
        | train dataset IOU | .84   |
        | train perimage iou| .85   |
        |-------------------|-------|
        | val iou           | .925  |
        | DS fusion         | .924  |
        | mean fusion       | .922  |
        | sum fusion        | .922  |
        |---Frequency IOU---|-------|
        | val               | .8947 |
        | DS fusion         | .8758 |
        | mean fusion       | .8738 |
        | sum fusion        | .8738 |
        | OneD fusion       | .9289 |
    - Experiment to compare 1D and 2D convolution 
    - Results
```
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Validate metric                 DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
             FrequencyIoU/valid/0           0.8860907554626465
             FrequencyIoU/valid/1           0.8861611485481262
        FrequencyIoU/valid/OneD_fusion      0.9062254428863525
        FrequencyIoU/valid/TwoD_fusion      0.8769735097885132
         FrequencyIoU/DS_combine            0.8712740540504456
            FrequencyIoU/mean               0.8695641160011292
            FrequencyIoU/sum                0.8695641160011292

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

          iou/valid/0_iou              0.9201490879058838
          iou/valid/1_iou              0.9200822114944458      
          iou/valid/OneD_fusion_iou         0.9374560117721558
          iou/valid/TwoD_fusion_iou         0.9278312921524048
          val_iou/DS_combine_fusion         0.9205654859542847
           val_iou/mean_fusion              0.9183549880981445
           val_iou/sum_fusion               0.9183549880981445

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

         MeanIoU/DS_combine     0.8629774451255798
          MeanIoU/mean          0.8613346815109253
           MeanIoU/sum          0.8613346815109253
          MeanIoU/val_0         0.8791393637657166
          MeanIoU/val_1         0.8791835904121399

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        PixelAccuracy/DS_combine    0.9271551370620728
           PixelAccuracy/mean       0.9260948896408081
            PixelAccuracy/sum       0.9260948896408081
           PixelAccuracy/val_0      0.9345839023590088
           PixelAccuracy/val_1      0.9346224069595337
        
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```
* Monday, Aug8 
    - Training with only flip augmentation in cluster. Validation accuracy is higher than train accuracy in the begining
    - Debug Script for training 1D fusion. After loading checkpoint validation is correct but after training val reduces even though we have freezed.

* Sunday, Aug 7
    - Training for 60 epochs 
    - Clipped dirchlet loss function .
    - Results:
        |  metric           | value |
        |-------------------|-------|
        | train dataset IOU | .84   |
        | train perimage iou| .84   |
        |-------------------|-------|
        | val iou           | .8539 |
        | DS fusion         | .8582 |
        | mean fusion       | .8563 |
        | sum fusion        | .8563 |


        

