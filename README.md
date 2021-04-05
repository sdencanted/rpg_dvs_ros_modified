rpg_dvs_ros
===========

# Disclaimer

The github repo is originally cloned from [rpg-dvs_ros](https://github.com/uzh-rpg/rpg_dvs_ros) which is contributed by the authors: Elias Mueggler, Basil Huber, Luca Longinotti, Tobi Delbruck. Code has been modified for the purpose of my current project.



# Driver Installation

NOTE: substitute any mention of kinetic in the following instruction with the name of your current ROS distribution.

1. Install ROS dependencies:
*   `$ sudo apt-get install ros-melodic-camera-info-manager`
*   `$ sudo apt-get install ros-melodic-image-view`

2. Install libcaer (add required repositories as per [iniVation documentation](https://inivation.gitlab.io/dv/dv-docs/docs/getting-started.html#ubuntu-linux) first):
*   `$ sudo apt-get install libcaer-dev`

3. Install catkin tools:
*   `$ sudo apt-get install python-catkin-tools`

4. Create a catkin workspace (if you have not done it yet):
*   `$ cd`
*   `$ mkdir -p catkin_ws/src`
*   `$ cd catkin_ws`
*   `$ catkin config --init --mkdirs --extend /opt/ros/melodic --merge-devel --cmake-args -DCMAKE_BUILD_TYPE=Release`

5. Clone the `catkin_simple` package (https://github.com/catkin/catkin_simple), which will be used to build the DVS/DAVIS driver packages:
*   `$ cd ~/catkin_ws/src`
*   `$ git clone https://github.com/catkin/catkin_simple.git`

6. Clone this repository:
*   `$ cd ~/catkin_ws/src`
*   `$ git clone https://github.com/uzh-rpg/rpg_dvs_ros.git`

7. Build the packages:
* `$ catkin build dvs_ros_driver`  (if you are using the DVS128)
* `$ catkin build davis_ros_driver`  (if you are using the DAVIS)
* `$ catkin build dvxplorer_ros_driver`  (if you are using the DVXplorer)

8. You can test the installation by running a provided launch file. It starts the driver (DVS or DAVIS) and the renderer (an image viewer).
    1. First, build the renderer:
        * `$ catkin build dvs_renderer`
    2. Set up the environment:
        * `$ source ~/catkin_ws/devel/setup.bash` or if you use the zsh shell instead `$ source ~/catkin_ws/devel/setup.zsh`
    3. Then, launch the example:
        * `$ roslaunch dvs_renderer dvs_mono.launch`  (if you are using the DVS128)
        * `$ roslaunch dvs_renderer davis_mono.launch` (if you are using the DAVIS)
        * `$ roslaunch dvs_renderer dvxplorer_mono.launch` (if you are using the DVXplorer)
    You should get an image like this (in case of the DAVIS):

        ![dvs_rendering_screenshot_19 04 2017](https://cloud.githubusercontent.com/assets/8024432/25172262/b96baaa0-24f0-11e7-9c3e-e33f6d398a4a.png)

9. **If you do not have a DAVIS, you can still use this driver to read recorded files**, such as those of [The Event Camera Dataset and Simulator](http://rpg.ifi.uzh.ch/davis_data.html).
   **Example**: <a name="ExampleEventCameraDataset"></a>
    1. Download a squence of the dataset, such as [slider_depth.bag](http://rpg.ifi.uzh.ch/datasets/davis/slider_depth.bag)
    2. Open a terminal and launch the roscore:
     * `$ roscore`
    3. In another terminal, play the bag:
     * `$ rosbag play -l path-to-file/slider_depth.bag`
    4. In another terminal, launch the DVS/DAVIS renderer:
     * `$ roslaunch dvs_renderer renderer_mono.launch`
    You should see a movie with images like this:

        ![slider_depth_renderer](https://cloud.githubusercontent.com/assets/8024432/25312371/9afd4180-2817-11e7-9e33-cdaa8af1e6ed.png)

10. Optional: in the case of a **live stream** from the DAVIS (i.e., not a recorded file) you may adjust the DVS/DAVIS parameters to your needs using the dynamic reconfigure GUI. Run
    * `$ rosrun rqt_reconfigure rqt_reconfigure`
   and a window will appear. Select the `davis_ros_driver` (on the left panel) and you should get the following GUI that allows you to modify the parameters of the sensor.

   ![davis_ros_driver_rqt_reconfigure](https://cloud.githubusercontent.com/assets/8024432/25172274/c1267b8a-24f0-11e7-8130-af551a8a958d.png)

   A guide on how to modify the parameters in the bottom half of the GUI (biases) can be found here: https://inivation.github.io/inivation-docs/Advanced%20configurations/User_guide_-_Biasing.html


# Calibration

For intrinsic or stereo calibration of the DVS and DAVIS, please have a look at the following [document](dvs_calibration/README.md).

