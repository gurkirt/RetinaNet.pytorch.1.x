## Pytorch implementation of RetinaNet
## OHEM Loss, Focal Loss, and YOLO Loss on top of FPN

## Introduction
This repository implements a pure pytorch [Focal-Loss for Object Detection](https://arxiv.org/pdf/1708.02002.pdf) paper. Aim of this repository try different loss functions and make a fair comparison in terms of performance/training -time/-GPU-memory. 

At the moment we support pytorch-1.2 and ubuntu with Anaconda distribution of python. Tested on a single machine with 8 GPUs, works on 4 GPUs as well.

This repository is a successive version of [FPN.pytorch.1.0](https://github.com/gurkirt/FPN.pytorch1.0). Both are quite different in terms of anchors used and input size. This repository uses anchors from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) while the other has SSD style anchors. Also, input image transformation and size are the same as [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) while others have fixed input size, e.g. 600x600.

We only evaluate object detection, there is no support for mask prediction or semantic segmentation. Our objective to reproduce RetinaNet paper in its entirety. Even though the original [RetnaNet](https://arxiv.org/pdf/1708.02002.pdf) did not have mask prediction capability but the latest version [RetinaMask](https://arxiv.org/pdf/1901.03353.pdf) has it. If you want mask prediction with RentinaNet please go to [RetinaMask repo](https://github.com/chengyangfu/retinamask).

# TODO
- Improve memory footprint
- - Use clustering for batch making  so all the images of similar size
- - see if we can improve loss functions
- Implement multi-scale training
- Implement fast evaluation like in paper
