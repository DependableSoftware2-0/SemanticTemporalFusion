# Experiments Sumary

* Wednesday Aug 10
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
    - Validating all methods (1D, 2D convolution with cross entropy loss and 10 epochs)



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


        

