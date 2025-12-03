#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge # 用于在 ROS 图像消息和 OpenCV 图像格式之间进行转换
import numpy as np
import cv2

#模拟相机设备
class ExternalCameraBridge(Node):
    def __init__(self):
        super().__init__('external_camera_bridge')
        
        # 发布图像和相机信息
        self.image_publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.camera_info_publisher_ = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
        
        self.bridge = CvBridge()
        
        # 初始化相机信息（根据你的相机参数调整）
        self.camera_info = self.create_camera_info()
        # 添加计数器跟踪回调次数
        self.callback_count = 0
        
        # 定时器，模拟图像获取
        self.timer = self.create_timer(0.1, self.publish_image_callback)  # 10Hz
        
        self.get_logger().info('External Camera Bridge started')
    
    def create_camera_info(self):
        """创建相机内参信息"""
        camera_info = CameraInfo()
        camera_info.width = 640
        camera_info.height = 480
        
        # 相机内参矩阵 K [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        camera_info.k = [
            500.0, 0.0, 320.0,  # fx, 0, cx
            0.0, 500.0, 240.0,  # 0, fy, cy
            0.0, 0.0, 1.0       # 0, 0, 1
        ]
        
        # 旋转矩阵 R (单位矩阵)
        camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        
        # 投影矩阵 P [fx, 0, cx, Tx, 0, fy, cy, Ty, 0, 0, 1, Tz]
        camera_info.p = [
            500.0, 0.0, 320.0, 0.0,  # fx, 0, cx, Tx
            0.0, 500.0, 240.0, 0.0,  # 0, fy, cy, Ty
            0.0, 0.0, 1.0, 0.0       # 0, 0, 1, Tz
        ]
        
        camera_info.distortion_model = "plumb_bob"
        # 畸变参数 [k1, k2, t1, t2, k3]
        camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        return camera_info
    
    def publish_image_callback(self):
        self.get_logger().info('Publishing image...'+str(self.callback_count))
        """模拟发布图像数据"""
        
        # 图像尺寸
        width, height = 640, 480
        test_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 棋盘格参数（世界坐标系中固定）
        board_cols, board_rows = 9, 6  # 9x6 内角点
        square_size = 0.025  # 每个方格25mm（实际物理尺寸）
        
        # 创建棋盘格的3D世界坐标点（固定不变）
        objp = np.zeros((board_cols * board_rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
        objp *= square_size  # 转换为实际尺寸

        # 相机内参（保持一致）
        camera_matrix = np.array([[500.0, 0.0, 320.0],
                                [0.0, 500.0, 240.0],
                                [0.0, 0.0, 1.0]], dtype=np.float32)
        dist_coeffs = np.zeros((5, 1))

        # 模拟相机位姿变化
        self.callback_count += 1
        angle = np.radians(self.callback_count % 360)
        
        # 相机的旋转（欧拉角）
        rvec = np.array([
            0.5 * np.sin(angle * 0.7),      # 绕X轴旋转
            0.3 * np.cos(angle * 0.5),      # 绕Y轴旋转
            0.2 * np.sin(angle * 0.3)       # 绕Z轴旋转
        ], dtype=np.float32)
        
        # 相机的平移
        tvec = np.array([
            -0.03 * np.sin(angle * 0.4),      # X方向移动
            -0.02 * np.cos(angle * 0.6),      # Y方向移动
            0.3 + 0.2 * np.sin(angle * 0.2) # Z方向移动（保持一定距离）
        ], dtype=np.float32)
        
        # 将3D世界点投影到2D图像平面
        img_points, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        img_points = img_points.reshape(-1, 2)

        # 绘制完整的棋盘格
        for row in range(board_rows - 1):
            for col in range(board_cols - 1):
                # 计算四个角点的索引
                top_left = row * board_cols + col
                top_right = row * board_cols + col + 1
                bottom_left = (row + 1) * board_cols + col
                bottom_right = (row + 1) * board_cols + col + 1
                
                # 获取四个角点的坐标
                points = []
                valid = True
                
                for idx in [top_left, top_right, bottom_right, bottom_left]:
                    if 0 <= idx < len(img_points):
                        x, y = int(img_points[idx][0]), int(img_points[idx][1])
                        if 0 <= x < width and 0 <= y < height:
                            points.append([x, y])
                        else:
                            valid = False
                            break
                    else:
                        valid = False
                        break
                
                # 如果所有点都在图像范围内，绘制方格
                if valid and len(points) == 4:
                    points = np.array(points, np.int32)
                    # 根据棋盘格模式确定颜色（黑白相间）
                    if (row + col) % 2 == 0:
                        color = (50, 50, 50)  # 深灰色
                    else:
                        color = (255, 255, 255)  # 白
                    
                    cv2.fillPoly(test_image, [points], color)


        # 转换为ROS图像消息
        img_msg = self.bridge.cv2_to_imgmsg(test_image, encoding="bgr8")
        timestamp = self.get_clock().now().to_msg()
        img_msg.header.stamp = timestamp
        img_msg.header.frame_id = "camera_1_link"
        
        # 发布图像
        self.image_publisher_.publish(img_msg)
        
        # 发布相机信息
        self.camera_info.header.stamp = timestamp
        self.camera_info.header.frame_id = "camera_1_link"
        self.camera_info_publisher_.publish(self.camera_info)
def main(args=None):
    rclpy.init(args=args)
    node = ExternalCameraBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()