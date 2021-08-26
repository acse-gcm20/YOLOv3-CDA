# YOLOv3 Crater Detection Algorithm and Degradation State Classifier
## Giles Matthews

Repository for my ACSE-9 Independent Research Project. This project applied the YOLOv3 feature extraction architecture to the problem of impact crater detection from images of Mars. Two models were developed: 

1. A single-class detection algorithm.
2. A four-class simultaneous detection and degradation state classification algorithm. The four classes relate to those described in the Robbins database where Class 1 contains the most degraded craters and Class 4 contains the most pristine craters.

A full description of this project and its results can be found in the Report PDF.
## How to Use
---
It is recommended that this repository is used on the Google Colab platform in order to make use of the available GPU resources. Example workflows are available in the repository.

Use the following command to clone the repository into Google Colab:

```
! git clone https://github.com/acse-gcm20/YOLOv3-CDA
```

The following Google Drive folder contains stats files, pretrained weights files and the necessary dataset zip files:
https://drive.google.com/drive/folders/1Qh7VHt_dTZc8v0tYdSZlXo3cwTmxnYCI?usp=sharing

Two model configuration files are provided in the [models](https://github.com/acse-gcm20/YOLOv3-CDA/tree/master/models) directory:

1. [CDA.cfg](https://github.com/acse-gcm20/YOLOv3-CDA/blob/master/CDA.cfg) - single class crater detection model.
2. [classifier.cfg](https://github.com/acse-gcm20/YOLOv3-CDA/blob/master/classifier.cfg) - four class detection and classification model.

These have corressponding configuration (.data) and class (.names) files in the [config](https://github.com/acse-gcm20/YOLOv3-CDA/tree/master/config) and [data](https://github.com/acse-gcm20/YOLOv3-CDA/tree/master/data) directories respectively.

## Code 
---
### Code Base
The Code base was originally forked from the following PyTorch implementation of the YOLOv3 architecture: https://github.com/eriklindernoren/PyTorch-YOLOv3

The scripts from this code base can be found in the [pytorchyolo](https://github.com/acse-gcm20/YOLOv3-CDA/tree/master/pytorchyolo) directory. The majority of this code base remains unchanged.

### Additional Scripts
A number of additional scripts have been written. These contain utility functions to plot figures, organise datasets and pre-process images and can be found in the [src](https://github.com/acse-gcm20/YOLOv3-CDA/tree/master/src) directory.

## Datasets
---
The datasets used in this project are available in zip files in the provided Google Drive folder. These files were not directly included in the repo because of their size. To access the datasets in Google Colab, ensure the Google Drive folder has been mounted, then use the following command:

```bash
! unzip <path to zip file> -d <destination directory> 
```

