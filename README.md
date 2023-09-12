# RIS_PROJECT - Object Detection

Duckietown was initially introduced by MIT and later adopted by many other universities who saw its potential and outreach effort. A cheap platform, in comparison to a car, for which one can teach an advanced autonomy class anywhere. In our efforts to bring another contribution to this class, my partner and I focused on the concept of object detection. Where we used architectures such as YOLOv4 and ROS together to make this happen.


## Getting Started

### Dependencies

* Tensorflow
* ROS Noetic
* Ubuntu 20.04.4 (focal)
* YOLO v4
* Opencv
* cuDNN (only for gpu model)
* CUDA  (only for gpu model)

### Executing program
* Set up and calibrate the duckiebot using the following [guide](https://docs.duckietown.org/daffy/opmanual_duckiebot/out/index.html)
* Take a look at the ROS Wiki website to create a [workspace](https://wiki.ros.org/ROS/Tutorials/CreatingPackage)
* Download the repository
```
cd catkin_ws/src
```
```
git clone --recursive git@github.com:vsancnaj/RIS_PROJECT.git
```
* Build
```
catkin_make -DCMAKE_BUILD_TYPE=Release
```
* Modify “ros.yaml” with the correct camera topic
* "my_object_detect.yaml" to configure the model files and detected classes
* Modify “darknet_ros.launch” with the correct YAML file (“my_object_detect.yaml”)
* Run
```
catkin_make
```
```
source devel/setup.bash
```
```
roslaunch darknet_ros darknet_ros.launch
```
---

## Yolov4 using Darknet

In this project, we used Alexey's darknet git repository and duckietown object detection git repository. Since the overall file size is too big, uploading files on github is not allowed, so we uploaded modified or created files only in this repository. You should ask TA for the whole project file. Darknet supports both cpu and gpu model. Down below is detailed code and explanation of implementing yolo using darknet in your own local computer. However, we used google colab, which allows the user to use Nvidia gpu for the limited amount of time, for our project and custom data training since we both don't have gpu and it will take long time for the model training. Therefore, we recommend to work on a google colab by following our project code if you have cpu only model. We wrote detailed explanation of each steps in jupyter notebook file named YOLO_v4.

### Build project

#### installing dependencies in ubuntu

If you are planning to run a project in google colab, you can skip all codes down below. There is colab python file named YOLO v4, and you can follow the codes and explanations there. 

* CMake >= 3.8
```
sudo apt install cmake 
```
* Opencv
```
sudo apt install libopencv-dev python3-opencv   
```
* OpenMP (for cpu model)
```
sudo apt install libomp-dev  
```
* Other dependencies 
```
sudo apt install make git g++    
```
* please install CUDA and cuDNN if you are using gpu model


#### Build Darknet yolov4 model

* clone git repository for darknet
```
git clone https://github.com/AlexeyAB/darknet.git
```
* Change makefile for cpu model
```
GPU=0
CUDNN=0
CUDNN_HALF=0
OPENCV=1
AVX=1
OPENMP=1
LIBSO=0 
ZED_CAMERA=0
ZED_CAMERA_v2_8=0 
```
* Change makefile for gpu model
```
GPU=1
CUDNN=1
CUDNN_HALF=1
OPENCV=1
AVX=0
OPENMP=0
LIBSO=1  
ZED_CAMERA=0 
ZED_CAMERA_v2_8=0 
```
* Make
```
cd darknet
make
```
If you succeed in building you will find darknet execution file in build path.
* download weight
```
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
```
* test on coco data
```
./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights data/dog.jpg
```
#### Custom data training
* install git lfs
```
git lfs install
git lfs track "*.xml"
git add .gitattributes
```
* git clone duckietown-objdet repository
```
cd darknet/data
git clone https://github.com/duckietown/duckietown-objdet
```
##### custom data xml to yolo annotation 

In this step, you need to create separate python file and run file after copying below codes.

* import libraries
``` 
import os 
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
```
Set each directory for the xml and image data files in duckietown data and read name of each files. 

* find class name and how many class indices we have
```
num = len(xml_fl)
obj_name_list = []
for i in range(num):
  with open(xml_path+xml_fl[i],'rb') as f:
    data = f.read()
  soup = BeautifulSoup(data, 'xml')
  obj = soup.find_all('object')
  for j in range(len(obj)):
    n = obj[j].find('name')
    obj_name_list = np.append(obj_name_list, n.text)
unique_obj_name_list = np.unique(obj_name_list)
unique_obj_name_list, len(unique_obj_name_list)
```
*read pascal voc xml file and convert it into yolo txt file


The xml file has both general information, file name and image size, and detailed information, including object name and bounding box size on each object. However, yolo annotation has the same number of lines with the number of objects in each image and each line has 5 different pieces of informations: class number, x coordinate, y coordinate, width, height. Except class numbers, and 4 different numbers should be normalized between 0 and 1. 
Below is how to convert xml data to yolo annotation.
```
bw = xmax - xmin
bh = ymax - ymin
x_center_coordinate = (xmin +(bw/2))/iw)
y_center_coordinate = (ymin +(bh/2))/ih)
yolo_width = bw/iw
yolo_height = bh/ih
```
Below is python code.
```
iw = 640
ih = 480
for i in range(num):
  with open(xml_path+xml_fl[i], 'r') as f:
    data = f.read()
  root = ET.XML(data)
  soup = BeautifulSoup(data,'xml')
  obj = soup.find_all('object')
  objn = len(obj)
  fn = os.path.splitext(xml_fl[i])[0]
  if len(np.argwhere(wo_img_fl == fn)) == 1:
    with open(txt_path+fn+'.txt', 'w') as f1:
      for j in range(5,objn+5):
        f1.write(str(int(np.argwhere(unique_obj_name_list == root[j][0].text)))+' ')
        bw = int(root[j][4][2].text) - int(root[j][4][0].text)
        bh = int(root[j][4][3].text) - int(root[j][4][1].text)
        f1.write(str(float((int(root[j][4][0].text)+(bw/2))/iw))+' ')
        f1.write(str(float((int(root[j][4][1].text)+(bh/2))/ih))+' ')
        f1.write(str(bw/iw)+' ')
        f1.write(str(bh/ih))
        f1.write('\n')
```
#### create new files and modify darknet

* create obj.data folder

It contains directory information for the train, valid data, name of each class, and model weight.
```
classes=7
train=data/train.txt
valid=data/test.txt
names=data/obj.names
backup=backup/
```
* create obj.names folder

It has name of each class in a corresponding order.
```
Duckie
Duckiebot
Intersection sign
QR code
Signal sign
Stop sign
Traffic light
```
* create txt file

Create train.txt file and test.txt file by saving name of each data with the corresponding directory by separating custom data in 9:1 ratio.
* modify cfg file

cfg file decides the model architecture and hyper-parameter of the model
```
cd /content/gdrive/MyDrive/yolov4/darknet/cfg

#!sed -i 's/batch=1/batch=64/g' yolov4-custom.cfg
#!sed -i 's/subdivisions=1/subdivisions=16/g' yolov4-custom.cfg
!sed -i 's/width=608/width=416/g' yolov4-custom.cfg
!sed -i 's/height=608/height=416/g' yolov4-custom.cfg
!sed -i 's/max_batches = 500500/max_batches = 14000/g' yolov4-custom.cfg
!sed -i 's/steps=400000,450000/steps=11200,12600/g' yolov4-custom.cfg
!sed -i 's/classes=80/classes=7/g' yolov4-custom.cfg
!sed -i 's/filters=256/filters=36/g' yolov4-custom.cfg

cd ..
```
* download pretrained weight
```
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
```
* train the model
```
cd /content/gdrive/MyDrive/yolov4/darknet
chmod +x ./darknet
./darknet detector train data/obj.data cfg/yolov4-custom.cfg yolov4.conv.137 -dont_show -ext_output -map
```
* performance check
```
./darknet detector map data/obj.data cfg/yolov4-custom.cfg /content/gdrive/MyDrive/yolov4/darknet/backup/yolov4-custom_last.weights -points 0
```
* test the model with image data

Prediction image is created and it automatically saved in the darknet folder. If you want to know how to show the image directly, please check python code in jupyter notebook file named YOLO_v4. 
```
./darknet detector test data/obj.data cfg/yolov4-custom-test.cfg /content/gdrive/MyDrive/yolov4/darknet/backup/yolov4-custom_last.weights /content/gdrive/MyDrive/yolov4/darknet/data/obj/B_BR_Duckbar_frame00052.jpg -thresh 0.3 
```

## Authors

* Valentina Sanchez

* Kim Eunhye

* Under supervision of: Prof. Amr Alanwar Abdelhafez

## Resources

* https://roboticsknowledgebase.com/wiki/machine-learning/ros-yolo-gpu/
* https://github.com/leggedrobotics/darknet_ros
* https://github.com/AlexeyAB/darknet
* https://robocademy.com/2020/05/01/a-gentle-introduction-to-yolo-v4-for-object-detection-in-ubuntu-20-04/
* https://medium.com/geekculture/train-a-custom-yolov4-object-detector-on-linux-49b9114b9dc8
