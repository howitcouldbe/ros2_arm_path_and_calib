# launch/simulation_hand_eye.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    
    # 虚拟相机节点（在Gazebo中模拟真实相机）
    virtual_camera = Node(
        package='camera_info_manager',
        executable='camera_info_manager_node',
        name='virtual_camera_info_pub',
        parameters=[{
            'camera_name': 'camera_link',
            'camera_info_url': 'package://z_arm_2442c0_hand_eye_calib/config/camera_info.yaml'
        }]
    )
    
    # 手眼标定节点
    hand_eye_calibration = Node(
        package='hand_eye_calibration',
        executable='hand_eye_calibration_node',
        name='hand_eye_calibration',
        parameters=[{
            'simulation_mode': True,
            'camera_topic': '/camera/image_raw',
            'robot_base_frame': 'base_link',
            'end_effector_frame': 'end_effector_link',
            'camera_frame': 'camera_link'
        }]
    )
    
    return LaunchDescription([
        virtual_camera,
        hand_eye_calibration
    ])