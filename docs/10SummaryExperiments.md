# Experiments Sumary


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

* Monday
    - Debuging 1D fine tuning 

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
    - 
