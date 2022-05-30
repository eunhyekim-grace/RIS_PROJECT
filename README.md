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

Forked version of Alexey's darknet git repository and duckietown object detection git repository has been used in this project. We used google colaboratory for the custom data training in order to train our model fast using gpu. Google colab allows the user to use Nvidia gpu for the limited amount of time. 

### Build project

#### installing dependencies in ubuntu

If you are planning to run a project in google colab, you can skip all codes down below. There is colab file named YOLO v4, and you can follow the codes and explanations there. 

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
if you succeed in building you will find darknet in build path
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
you should create run below codes in python or in jupyter notebook
* import libraries
``` 
import os 
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
```
set each directory for the xml and image data files in duckietown data and read name of each files. 
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
The xml file has both general information, like file name and image size, and detailed information, like object name and bounding box size, on each object. However, yolo annotation has the same number of lines with the number of objects in each image and each line has 5 different pieces of informations: class number, x coordinate, y coordinate, width, height. Except class numbers, 4 different numbers should be normalized between 0 and 1. 
below is how to convert xml data to yolo annotation
```
bw = xmax - xmin
bh = ymax - ymin
x_center_coordinate = (xmin +(bw/2))/iw)
y_center_coordinate = (ymin +(bh/2))/ih)
yolo_width = bw/iw
yolo_height = bh/ih
```
below is python code 
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
* create obj folder
```
with open(obj_path+'obj.data','w')as f:
  f.write('classes = 7\n')
  f.write('train=data/train.txt\n')
  f.write('valid=data/test.txt\n')
  f.write('names=data/obj.names\n')
  f.write('backup=backup/')
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
