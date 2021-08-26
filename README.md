# YOLOv3 Crater Detection Algorithm and Degradation State Classifier
## Giles Matthews

Repository for my ACSE-9 Independent Research Project. This project applied the YOLOv3 feature extraction architecture to the problem of impact crater detection from images of Mars. Two models were developed: 

1. A single-class detection algorithm.
2. A four-class simultaneous detection and degradation state classification algorithm. The four classes relate to those described in the Robbins database where Class 1 contains the most degraded craters and Class 4 contains the most pristine craters.

A full description of this project and its results can be found in the Report PDF.
## How to Use
---
It is recommended that this repository is used on the Google Colab platform in order to make use of the available GPU resources. Example workflows are available in the repository.

Use the following commands to clone the repository into Google Colab and install the necessary dependencies:

```
! git clone https://github.com/acse-gcm20/YOLOv3-CDA
! pip install -r /content/YOLOv3-CDA/requirements.txt
```

The following Google Drive folder contains stats files, pretrained weights files and the necessary dataset zip files:
https://drive.google.com/drive/folders/1Qh7VHt_dTZc8v0tYdSZlXo3cwTmxnYCI?usp=sharing

Two model configuration files are provided in the [models](https://github.com/acse-gcm20/YOLOv3-CDA/tree/master/models) directory:

1. [CDA.cfg](https://github.com/acse-gcm20/YOLOv3-CDA/blob/master/CDA.cfg) - single class crater detection model.
2. [classifier.cfg](https://github.com/acse-gcm20/YOLOv3-CDA/blob/master/classifier.cfg) - four class detection and classification model.

These have corresponding configuration (.data) and class (.names) files in the [config](https://github.com/acse-gcm20/YOLOv3-CDA/tree/master/config) and [data](https://github.com/acse-gcm20/YOLOv3-CDA/tree/master/data) directories respectively.

The config file contains the number of classes; points to two text files (e.g. train.txt and valid.txt) which contain the paths to all of the images in each of the training and validation sets. And the path to the class file.

The model expects that the label for an image will be in a 'labels' folder alongside the 'images' folder, the labels must have the same name as the image but with the extension ```.txt```.

### **Training**

Command to begin training a model.
```python
from pytorchyolo.train import run
run(model, epochs, config_file)
```
```model``` - [string] Path to the model to train e.g. ```models/CDA.cfg```.

```epochs``` - [integer] Number of epochs to train for. 

```config_file``` - [string] Path to the config file.

Optional parameters include specifying pretrained weights and providing a stats file to append to.

### **Detecting**

```python
from pytorchyolo.detect import detect_directory
detect_directory(model, weights, directory, classes, output)
```

```model``` - [string] Path to the model to inference with e.g. ```models/CDA.cfg```.

```weights``` - [string] Path to the weights file to be used.

```directory``` - [string] Path to directory of images to detect craters from.

```classes``` - [list] List of strings with class names.

```output``` - [string] Path to desired output location.

### **Testing**

```python
from pytorchyolo.test import test
test(model, weights, config, test_file)
```

```model``` - [string] Path to the model to test e.g. ```models/CDA.cfg```.

```weights``` - [string] Path to the weights file to be used.

```config``` - [string] Path to config file.

```classes``` - [list] Path to text file containing paths of images in test set.

## Code 
---
### **Code Base**
The Code base was originally forked from the following PyTorch implementation of the YOLOv3 architecture: https://github.com/eriklindernoren/PyTorch-YOLOv3.

The scripts from this code base can be found in the [pytorchyolo](https://github.com/acse-gcm20/YOLOv3-CDA/tree/master/pytorchyolo) directory. The majority of this code base remains unchanged. Some editing has been performed on the train.py, test.py and detect.py scripts so that they can be more easily run from a Python environment rather than on a command line. Some code has also been added to these scripts to expose and save statistics relevant to the project.

### **Additional Scripts**
A number of additional scripts have been written. These contain utility functions to plot figures, organise datasets and pre-process images and can be found in the [src](https://github.com/acse-gcm20/YOLOv3-CDA/tree/master/src) directory.

### **Weights Files and Statistics**
Weights files (.pth) can be found in the /weights/ directory in the google drive, the corresponding statistics files (.txt) can be found in the /stats/ directory.

## **Datasets**
---
The datasets used in this project are available in zip files in the provided Google Drive folder. These files were not directly included in the repo because of their size. To access the datasets in Google Colab, ensure the Google Drive folder has been mounted, then use the following command:

```bash
! unzip <path to zip file> -d <destination directory> 
```

The nature of these datasets is described in the Report PDF.

There are three zip files that are used in this project:
1. benedix.zip
2. processed_Robbins.zip
3. small_Robbins.zip

When training the classifier, the sort_dataset.py script contains a Dataset class which is used to move the appropriate images and labels containing classified craters from the Robbins dataset into a seperate 'classifier' directory. This Class takes a ```threshold``` float [0, 1] parameter to set the desired object loss threshold and a ```clean``` boolean parameter to declare whether or not to use images which exclusively contain classified craters.

Dataset Name | .zip file | Threshold | Clean
-------------|-----------|-----------|-------
Full Robbins | processed_Robbins.zip | 1 | False
Half Robbins | small_Robbins.zip | 0.5 | False
Clean Robbins | processed_Robbins.zip | 1 | True
