## Pytorch implementation of RetinaNet
## OHEM Loss, Focal Loss, and YOLO Loss on top of FPN

## Introduction
This repository implements a pure pytorch [Focal-Loss for Object Detection](https://arxiv.org/pdf/1708.02002.pdf) paper. Aim of this repository try different loss functions and make a fair comparison in terms of performance/training -time/-GPU-memory. 

At the moment we support pytorch-1.2 and ubuntu with Anaconda distribution of python. Tested on a single machine with 8 GPUs, works on 4 GPUs as well.

This repository is a successive version of [FPN.pytorch.1.0](https://github.com/gurkirt/FPN.pytorch1.0). Both are quite different in terms of anchors used and input size. This repository uses anchors from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) while the other has SSD style anchors. Also, input image transformation and size are the same as [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) while others have fixed input size, e.g. 600x600.

We only evaluate object detection, there is no support for mask prediction or semantic segmentation. Our objective to reproduce RetinaNet paper in its entirety. Even though the original [RetnaNet](https://arxiv.org/pdf/1708.02002.pdf) did not have mask prediction capability but the latest version [RetinaMask](https://arxiv.org/pdf/1901.03353.pdf) has it. If you want mask prediction with RentinaNet please go to [RetinaMask repo](https://github.com/chengyangfu/retinamask).


## Architecture 
![RetinaNet Structure](/figures/retinaNet.png)

ResNet is used as a backbone network (a) to build the pyramid features (b). 
Each classification (c) and regression (d) subnet is made of 4 convolutional layers and finally a convolutional layer to predict the class scores and bounding box coordinated respectively.

Similar to the original paper, we freeze the batch normalisation layers of ResNet based backbone networks. Also, few initial layers are also frozen, see `fbn` flag in training arguments. 

## Loss functions 
### OHEM with multi-box loss function
We use multi-box loss function with online hard example mining (OHEM), similar to [SSD](https://arxiv.org/pdf/1512.02325.pdf).
A huge thanks to Max DeGroot, Ellis Brown for [Pytorch implementation](https://github.com/amdegroot/ssd.pytorch) of SSD and loss function.

### Focal loss
Same as in the original paper we use sigmoid focal loss, see [RetnaNet](https://arxiv.org/pdf/1708.02002.pdf). We use pure pytorch implementation of it.

### Yolo Loss
Multi-part loss function from [YOLO](https://pjreddie.com/darknet/yolo/) is also implemented here.

## Results
Here are the results on `coco` dataset.

Loss |depth | input | AP    | AP_50   | AP_75 | AP_S | AP_M | AP_L |
|----|-------|:----: |:----:| :-----:  | :---:| :---:| :---:| :---: |
| Focal-[paper](https://arxiv.org/pdf/1708.02002.pdf) | 50 |  600 |  34.3 | 53.2 | 36.9 | 16.2 | 37.4  | 47.4 |
| Focal-here | 50 |  600 |  34.3 | 52.5 | 36.1 | 16.4 | 37.9  | 48.0 |
| Yolo | 50 |  600 |  33.3 | 52.2 | 36.1 | 15.7 | 36.7  | 46.8 |
| OHEM | 50 |  600 |  34.7 | 52.5 | 37.0 | 16.9 | 37.9  | 48.9 |

## Details
- Input image size is `600`.
- Resulting feature map size on five pyramid levels is `[75, 38, 19, 10, 5]` on one side 
- Batch size is set to `16`, the learning rate of `0.01`.
- COCO would need 3-4 GPUs because the number of classes is 80, hence loss function requires more memory

## Installation
- We used anaconda 3.7 as python distribution
- You will need [Pytorch1.x](https://pytorch.org/get-started/locally/)
- visdom and tensorboardX if you want to use the visualisation of loss and evaluation
  - if you want to use them set visdom/tensorboard flag equal to true while training 
  - and configure the visdom port in arguments in  `train.py.`
- You will need to install [COCO-API](https://github.com/cocodataset/cocoapi) to use evalute script.

## TRAINING
Please follow dataset preparation [README](https://github.com/gurkirt/FPN.pytorch/tree/master/prep) from `prep` folder of this repo.
Once you have pre-processed the dataset, then you are ready to train your networks.

To train run the following command. 

```
python train.py --loss_type=focal
```

It will use all the visible GPUs. 
You can append `CUDA_VISIBLE_DEVICES=<gpuids-comma-separated>` at the beginning of the above command to mask certain GPUs. We used 8 GPU machine to run these experiments.

Please check the arguments in `train.py` to adjust the training process to your liking.

## Evaluation
Model is evaluated and saved after each `25K` iterations. 

mAP@0.5 is computed after every `25K` iterations and at the end.

Coco evaluation protocol is demonstraed  in `evaluate.py` 


```
python evaluate.py --loss_type=focal
```

## COCO-API Result
Here are results on COCO dataset using [COCO-API](https://github.com/cocodataset/cocoapi).
Results using `cocoapi` are slightly different than above table what you see during training time. 


#### TODO update the results below.
```
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.348
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.525
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.370
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.169
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.380
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.489
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.298
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.454
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.487
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.268
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.536
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.651

```

## TODO
- Improve memory footprint
    - Use clustering for batch making  so all the images of similar size
    - see if we can improve loss functions
- Implement multi-scale training
- Implement fast evaluation like in paper
