# Self-Guided Adaptation

Please Follow [faster-rcnn repository](https://github.com/jwyang/faster-rcnn.pytorch) to set up the environment

### Prerequisites

* Python 2.7 or 3.6
* Pytorch 0.4.0 (**now it does not support 0.4.1 or higher**)
* CUDA 8.0 or higher

### Data Preparation

* **PASCAL_VOC 07+12**: Please refer [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) for constructing PASCAL VOC Datasets.
* **WaterColor**: Please refer [Cross Domain Detection ](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets). 
* **Night-time(Detrac-night)**: TBA.
* **Citysscape, FoggyCityscape**: Please refer [Cityscape](https://www.cityscapes-dataset.com/).

### All codes are written to fit for the format of PASCAL_VOC.

### Pretrained Model

We used ResNet101 pretrained on the ImageNet in our experiments. You can download the model from:

* ResNet101: [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)

#### Well-trained Domain Adaptation Object Detection Models

* Pascal VOC to WaterColor(Res101-based): [GoogleDrive](https://drive.google.com/open?id=1bDjEkJCjz2DHP90ATUQL5wVwD4Qmq2fF)

## Train
* SGA-H
```
 CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net_auto.py \
                    --dataset source_dataset --dataset_t target_dataset --net res101 \
                    --cuda
