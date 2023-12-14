cmake_minimum_required(VERSION 3.5)
set(CMAKE_VERBOSE_MAKEFILE ON)

# CUDA
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 87)
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G") # enable cuda-gdb
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
project(dvxplorer_motion_compensator
    VERSION 0.1
    LANGUAGES C CXX CUDA)

# -Werror  -Wpedantic
add_compile_options(-Wall -Wextra)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_ros REQUIRED)
find_package(ament_cmake_auto REQUIRED)

set(ROS2_DEPENDENCIES
    "rclcpp"
    "rclcpp_components"
    "dvs_msgs"
    "builtin_interfaces"
    "sensor_msgs"
    "geometry_msgs"
    "std_msgs"
    "tf2"
    "tf2_ros"
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

find_package(CUDA 11.4 REQUIRED)
message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")
message(STATUS "Find CUDA libraries: ${CUDA_LIBRARIES}")
include_directories("${CUDA_INCLUDE_DIRS}")

# jetson-utils
find_package(jetson-utils REQUIRED)
link_directories(/usr/lib/aarch64-linux-gnu/tegra)

# catkin_simple()
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -march=native")

# -march=armv8.2-a -fno-finite-math-only
find_package(Boost REQUIRED COMPONENTS system thread)

# ament_auto_add_library(motion_compensator
# src/motion_compensator.cpp)
add_library(motion_compensator
    src/motion_compensator.cpp  src/cuda_compensator.cu)
target_include_directories(motion_compensator
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
rclcpp_components_register_nodes(motion_compensator "dvxplorer_motion_compensator::DvxplorerMotionCompensator")

target_link_libraries(motion_compensator
    # ${catkin_LIBRARIES}
    ${Boost_LIBRARIES}
    ${CUDA_LIBRARIES}
    jetson-utils
)

ament_target_dependencies(motion_compensator ${ROS2_DEPENDENCIES})
ament_export_targets(motion_compensatorTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(${ROS2_DEPENDENCIES})

# the shared library goes into the global lib dir so it can
# be used as a composable node by other projects
# install(TARGETS
# dvxplorer_motion_compensator
# DESTINATION lib
# )
install(
    DIRECTORY include/
    DESTINATION include
)
install(
    TARGETS motion_compensator
    EXPORT motion_compensatorTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include)

install(DIRECTORY
    launch
    DESTINATION share/${PROJECT_NAME}/
    FILES_MATCHING PATTERN "*.py")

ament_auto_add_executable(dvxplorer_motion_compensator_node src/motion_compensator_node.cpp)
target_link_libraries(dvxplorer_motion_compensator_node motion_compensator)

# the node must go into the paroject specific lib directory or else
# the launch file will not find it
install(TARGETS
    dvxplorer_motion_compensator_node
    DESTINATION lib/${PROJECT_NAME}/)

ament_export_dependencies(ament_cmake)
ament_export_dependencies(${ROS2_DEPENDENCIES})
ament_export_dependencies(class_loader)

ament_export_include_directories(include)
ament_export_libraries(motion_compensator)

# link the executable to the necessary libs
ament_package()
