from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'z_arm_2442c0_hand_eye_calib'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # launch
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sunshine',
    maintainer_email='1095134606@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'external_camera_bridge = src.ExternalCameraBridge:main',
            'calibration_board_detector = src.CalibrationBoardDetector:main',
            'hand_eye_calibration_node = src.HandEyeCalibrationNode:main',
        ],
    },
)
