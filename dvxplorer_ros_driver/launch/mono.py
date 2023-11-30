from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='dvxplorer_ros_driver',
            namespace='dvxplorer',
            executable='dvxplorer_ros_driver_node',
            name='dvxplorer_ros_driver'
        )
    ])