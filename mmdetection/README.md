
# Introduction  
This repository contains the source code for locate eyelid tumors based on clinical images.

# Prerequisites
* Ubuntu: 18.04 lts
* Python 3.7.8
* Pytorch 1.6.0
* NVIDIA GPU + CUDA_10.0 CuDNN_7.5

This repository has been trained and tested on four NVIDIA RTX2080Ti. 

# Installation
Other packages are as follows:
* pytorch: 1.6.0 
* numpy: 1.19.1
* mmcv-full 1.2.4
# Install dependencies
pip install -r requirements.txt
# Usage
* The file "train.py" in /tools is used for our models training.
* The file "test.py" in /tools is used for testing.

The training and testing are executed as follows:

## Train a new model
python tools/train.py configs/pascal_voc/faster_rcnn_r50_fpn_2x_voc0712.py

## Test and inference
python tools/test.py configs/pascal_voc/faster_rcnn_r50_fpn_2x_voc0712.py work_dirs/faster_rcnn_r50_fpn_2x_voc0712/latest.pth --eval mAP
***

**Please feel free to contact us for any questions or comments: Jiewei Jiang, E-mail: jiangjw924@126.com or Zhongwen Li, E-mail: li.zhw@qq.com.**
