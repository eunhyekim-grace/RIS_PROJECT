# RIS_PROJECT - Object Detection

Duckietown was initially introduced by MIT and later adopted by many other universities who saw its potential and outreach effort. A cheap platform, in comparison to a car, for which one can teach an advanced autonomy class anywhere. In our efforts to bring another contribution to this class, my partner and I focused on the concept of object detection. Where we used architectures such as YOLOv4 and ROS together to make this happen.


## Getting Started

### Dependencies

* Tensorflow
* ROS Noetic
* Ubuntu 20.04.4 (focal)
* YOLO v4

### Executing program
* Set up and calibrate the duckiebot using the following [guide](https://docs.duckietown.org/daffy/opmanual_duckiebot/out/index.html)
* Take a look at the ROS Wiki website to create a [workspace](https://wiki.ros.org/ROS/Tutorials/CreatingPackage)
* Download the repository
```
cd catkin_ws/src
```
```
git clone --recursive THIS GITHUB
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


## Authors

* Valentina Sanchez

* Kim Eunhye

* Under supervision of: Prof. Amr Alanwar Abdelhafez

## Resources

* https://roboticsknowledgebase.com/wiki/machine-learning/ros-yolo-gpu/
* https://github.com/leggedrobotics/darknet_ros
