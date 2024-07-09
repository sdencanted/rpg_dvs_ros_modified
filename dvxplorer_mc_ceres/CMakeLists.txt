cmake_minimum_required(VERSION 2.8.3)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 87)
# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G") # enable cuda-gdb
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-lineinfo") # enable line info
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_87 -gencode  arch=compute_87,code=sm_87)
project(dvxplorer_mc_ceres
LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -O3")

# catkin simple
find_package(catkin_simple REQUIRED)
add_definitions(-DUSING_ROS_1)

# CUDA
find_package(CUDA 11.4 REQUIRED)
message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")
message(STATUS "Find CUDA libraries: ${CUDA_LIBRARIES}")
include_directories("${CUDA_INCLUDE_DIRS}")

# Ceres
find_package(Ceres REQUIRED COMPONENTS EigenSparse)

# jetson-utils
find_package(jetson-utils REQUIRED)
include_directories(/usr/lib/aarch64-linux-gnu/tegra)
link_directories(/usr/lib/aarch64-linux-gnu/tegra)

#turbojpeg
find_package(JPEG REQUIRED)

#eigen
find_package (Eigen3 3.0 REQUIRED NO_MODULE)

catkin_simple()
# make the executable
# cs_add_executable(dvxplorer_mc_ceres
# src/driver_node.cpp
# src/driver.cpp
# src/motion_compensation_float.cu 
# )

# cs_add_executable(dvxplorer_mc_ceres_evk4
#   src/driver_node_evk4.cpp
#   src/driver_evk4.cpp
#   src/motion_compensation_float.cu 
# )
cs_add_executable(dvxplorer_mc_ceres_evk4_lbfgsb
  src/driver_node_evk4_lbfgsb.cpp
  src/driver_evk4_lbfgsb.cpp
  src/motion_compensation.cu 
  src/mc_gradient_lbfgsb.cu 
)

# make the nodelet into a library
# cs_add_library(dvxplorer_mc_ceres_nodelet
#   src/driver_nodelet.cpp
#   src/driver.cpp
#   src/motion_compensation_float.cu 
# )

# link the executable to the necessary libs
# target_link_libraries(dvxplorer_mc_ceres
#   ${catkin_LIBRARIES}
#   Ceres::ceres
#   ${CUDA_LIBRARIES}
#   jetson-utils
#   turbojpeg
# )
# target_link_libraries(dvxplorer_mc_ceres_evk4
#   ${catkin_LIBRARIES}
#   Ceres::ceres
#   ${CUDA_LIBRARIES}
#   jetson-utils
#   turbojpeg
# )
target_link_libraries(dvxplorer_mc_ceres_evk4_lbfgsb
  ${catkin_LIBRARIES}
  ${CUDA_LIBRARIES}
  jetson-utils
  turbojpeg
  Eigen3::Eigen
)

# link the executable to the necessary libs
# target_link_libraries(dvxplorer_mc_ceres_nodelet
#   ${catkin_LIBRARIES}
#   Ceres::ceres
#   ${CUDA_LIBRARIES}
#   jetson-utils
#   turbojpeg
# )


# catkin_install_python(PROGRAMS scripts/subEvent.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# Install the nodelet library
cs_install()

# Install other support files for installation
install(FILES dvxplorer_mc_ceres_nodelet.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

# Export package
cs_export()
