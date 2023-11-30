
cmake_minimum_required(VERSION 3.5)
project(dvxplorer_ros_driver)

add_compile_options(-Wall -Wextra -Wpedantic -Werror)
# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_ros REQUIRED)
find_package(ament_cmake_auto REQUIRED)


set(ROS2_DEPENDENCIES
  "rclcpp"
  "rclcpp_components"
  "sensor_msgs"
  "dvs_msgs"
  "builtin_interfaces"
  "std_msgs"
  "camera_info_manager"
)

foreach(pkg ${ROS2_DEPENDENCIES})
  find_package(${pkg} REQUIRED)
endforeach()

ament_auto_find_build_dependencies(REQUIRED ${ROS2_DEPENDENCIES})
# find_package(catkin_simple REQUIRED)
find_package(libcaer REQUIRED)

# let cmake find OpenMP and set some variables
find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    message(STATUS "OPENMP FOUND")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(STATUS "OPENMP NOT FOUND")
endif()

# CUDA
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G") # enable cuda-gdb
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDA REQUIRED)
message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")
message(STATUS "Find CUDA libraries: ${CUDA_LIBRARIES}")

# catkin_simple()
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -march=native")

# -march=armv8.2-a -fno-finite-math-only
find_package(Boost REQUIRED COMPONENTS system thread)


ament_auto_add_library(dvxplorer_ros_driver
src/driver_ros2.cpp
src/rm3100_spi_userspace.c)

rclcpp_components_register_nodes(dvxplorer_ros_driver "dvxplorer_ros_driver::DvxplorerRosDriver")

target_link_libraries(dvxplorer_ros_driver
    ${catkin_LIBRARIES}
    ${Boost_LIBRARIES}
    libcaer::caer
    ${OpenMP_LIBS}
    m
    gpiod
)

# make the executable
# cs_add_executable(dvxplorer_ros_driver
# src/driver_node.cpp
# src/driver.cpp
# )
ament_auto_add_executable(dvxplorer_ros_driver_node src/driver_node_ros2.cpp)
target_link_libraries(dvxplorer_ros_driver_node dvxplorer_ros_driver)

# the node must go into the paroject specific lib directory or else
# the launch file will not find it
install(TARGETS
    dvxplorer_ros_driver_node
    DESTINATION lib/${PROJECT_NAME}/)

# the shared library goes into the global lib dir so it can
# be used as a composable node by other projects
install(TARGETS
    dvxplorer_ros_driver
    DESTINATION lib
)

install(DIRECTORY
    launch
    DESTINATION share/${PROJECT_NAME}/
    FILES_MATCHING PATTERN "*.py")

ament_export_dependencies(rosidl_default_runtime)
# link the executable to the necessary libs
ament_package()
