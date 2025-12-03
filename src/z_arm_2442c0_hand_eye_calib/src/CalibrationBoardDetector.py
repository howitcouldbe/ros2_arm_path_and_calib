#!/usr/bin/env python3
# calibration_board_detector.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading
import time

# 标定板检测节点,收集手眼标定的图像数据
class CalibrationBoardDetector(Node):
    def __init__(self):
        super().__init__('calibration_board_detector')
        self.subscription = self.create_subscription(
            Image,
            '/calib_camera_sensor/image_raw',
            self.image_callback,
            10)
        
        
        self.transform_publisher = self.create_publisher(TransformStamped, '/calibration_board/transform', 10)
            
        self.bridge = CvBridge()
        self.callback_count = 0
        self.flag=False
        
        # 定义棋盘格尺寸（内角点数量）
        self.board_size = (7, 6)  # 7*6 内角点
        self.square_size = 0.05  # 每个方格边长 50mm
        
        # 存储检测到的角点
        self.corners = []
        # 没有实际的相机标定板，这里使用假设的棋盘格
        self.object_points = self.create_object_points()
        
        self.get_logger().info('Calibration Board Detector started')
        # 输入yes以接收图片，否则不接收
        threading.Thread(target=self.user_input_thread, daemon=True).start()
    
    def user_input_thread(self):
        while True:
            input_str = input("是否接收图片 (yes/no): ")
            if input_str.lower() == 'yes':
                self.flag=True
            else:
                self.flag=False
    def create_object_points(self):
        """创建棋盘格的3D点坐标（世界坐标系）"""
        objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp
    
    def image_callback(self, msg):
        """处理图像，检测棋盘格角点"""
        try:
            if not self.flag:
                return
            self.flag=False
            # 将ROS图像消息转换为OpenCV图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imshow("Image window", cv_image)
            cv2.imwrite(f"imgs/image_{str(time.time())}.jpg", cv_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            self.callback_count += 1
            
            # 转换为灰度图
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # 查找棋盘格角点
            # 或者使用更宽松的检测
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

            ret, corners = cv2.findChessboardCorners(gray, self.board_size, None, flags)
            self.get_logger().info('ret='+str(ret))
            if ret:
                # 精细化角点位置
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # 绘制角点
                cv2.drawChessboardCorners(cv_image, self.board_size, corners, ret)
                
                # 计算标定板的姿态
                self.estimate_pose(corners, gray.shape[::-1])
                
                self.get_logger().info('Chessboard detected')
            else:
                self.get_logger().info('Chessboard not detected')
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
    
    def estimate_pose(self, corners, image_size):
        """估计标定板相对于相机的姿态"""
        # 相机内参（使用你之前定义的参数）
        self.get_logger().info('Estimating pose...')
        camera_matrix = np.array([[554.383, 0.0, 320.5],
                              [0.0, 554.383, 240.5],
                              [0.0, 0.0, 1.0]])
        dist_coeffs = np.zeros((5, 1))  # 无畸变
        
        # 求解PnP问题，计算标定板相对于相机的位姿
        ret, rvec, tvec = cv2.solvePnP(self.object_points, corners, camera_matrix, dist_coeffs)
        
        if ret:
            # 将旋转向量转换为旋转矩阵
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            # 输出结果 AX=XB，需要求解X，这里的[[rotation_matrix],[tvec.flatten]]就是B
            self.get_logger().info(f'Translation: {tvec.flatten()}')
            self.get_logger().info(f'Rotation Matrix:\n{rotation_matrix}')
            # 发布标定板变换
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = 'camera_link'
            transform.child_frame_id = 'calibration_board_link'
            transform.transform.translation.x = float(tvec[0][0])
            transform.transform.translation.y = float(tvec[1][0])
            transform.transform.translation.z = float(tvec[2][0])
            r = R.from_matrix(rotation_matrix)
            quat = r.as_quat()
            transform.transform.rotation.x = float(quat[0])
            transform.transform.rotation.y = float(quat[1])
            transform.transform.rotation.z = float(quat[2])
            transform.transform.rotation.w = float(quat[3])
            self.transform_publisher.publish(transform)
            self.get_logger().info('Transformation published: '+str(transform))
    

def main(args=None):
    rclpy.init(args=args)
    detector = CalibrationBoardDetector()
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()