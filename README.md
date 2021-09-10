
# Introduction  
This repository contains the source code for developing an artificial intelligence (AI) system to discriminate between malignant and benign eyelid tumors based on clinical images.
This system has the potential to serve as a screening tool to detect malignant eyelid tumors at an early stage and then offer a timely referral for a positive case.

# Prerequisites
* Ubuntu: 18.04 lts
* Python 3.7.8
* Pytorch 1.6.0
* NVIDIA GPU + CUDA_10.0 CuDNN_7.5

This repository has been trained and tested on four NVIDIA RTX2080Ti. Configurations (e.g batch size, image patch size) may need to be changed on different platforms.

# Installation
Other packages are as follows:
* pytorch: 1.6.0 
* wheel:  0.34.2
* yaml:   0.2.5
* scipy:  1.5.2
* joblib: 0.16.0
* opencv-python: 4.3.0.38
* scikit-image: 0.17.2
* numpy: 1.19.1
* matplotlib：3.3.1
* sikit-learn：0.23.2
# Install dependencies
pip install -r requirements.txt
# Usage
* The file "eyetumor_training_V1.py" in /EyelidTumors-Source is used for our models training.
* The file "eyetumor_testing_V1.py" in /EyelidTumors-Source is used for testing.

The training and testing are executed as follows:

## Train DenseNet121 on GPU
python eyetumor_training_V1.py -a 'densenet121'

## Train ResNet50 on GPU
python eyetumor_training_V1.py -a 'resnet50'

## Train vgg16 on GPU
python eyetumor_training_V1.py -a 'vgg16'

## Train Inception-v3 on GPU
python eyetumor_training_V1.py -a 'inception_v3'

## Evaluate three models of DenseNet121, ResNet50, Vgg16 and Inception-v3 at the same time on GPU
python eyetumor_testing_V1.py
***

The representative samples for malignant and benign eyelid tumors are presented in /EyelidTumors-Source/sample.  
The expected output: print the classification probabilities for malignant and benign eyelid tumors.

**Please feel free to contact us for any questions or comments: Jiewei Jiang, E-mail: jiangjw924@126.com or Zhongwen Li, E-mail: li.zhw@qq.com.
