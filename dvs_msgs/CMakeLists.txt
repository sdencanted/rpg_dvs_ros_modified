# cmake_minimum_required(VERSION 2.8.3)
cmake_minimum_required(VERSION 3.5)
project(dvs_msgs)

# search for everything we need to build the messages, dont forget the message_generation
find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# search for all msg files
FILE(GLOB messages_to_build RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/msg/*.msg")

rosidl_generate_interfaces(${PROJECT_NAME}
    ${messages_to_build}
    DEPENDENCIES builtin_interfaces std_msgs
    ADD_LINTER_TESTS)

ament_export_dependencies(rosidl_default_runtime)

ament_package()

# # build the header files from the msg files, and notify catkin about the dependencies
# generate_messages(
# DEPENDENCIES
# std_msgs
# )

# # export the dependencis of this package for who ever depends on us
# catkin_package(
# CATKIN_DEPENDS message_runtime std_msgs
# )

# include_directories(
# ${catkin_INCLUDE_DIRS}
# )
