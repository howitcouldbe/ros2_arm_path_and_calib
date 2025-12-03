import logging
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import json
from geometry_msgs.msg import TransformStamped
from builtin_interfaces.msg import Time
import yaml
import cv2

# 手眼标定节点
class HandEyeCalibrationNode():
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            # 文件handler
            file_handler = logging.FileHandler('hand_eye_calibration.log')
            file_handler.setLevel(logging.DEBUG)
            
            # 控制台handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 创建formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # 添加handler到logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

        self.board_size = (7, 6)  # 7*6 内角点
        self.square_size = 0.03  # 每个方格边长 30mm
        self.object_points = self.create_object_points()
        

        # 尝试加载已保存的标定结果
        self.calibration_matrix = self.load_calibration_yaml()
        if self.calibration_matrix is not None:
            self.logger.info('Loaded existing calibration matrix')
        else:
            self.logger.info('No existing calibration found, will perform new calibration')
        
        # 存储观测数据
        self.robot_transforms = []  # 机器人末端相对于基座的变换
        self.camera_transforms = []  # 相机观察到的标定板变换
        
        self.logger.info('Hand Eye Calibration Node started')
    def save_calibration_yaml(self, x_matrix):
        """保存标定结果到YAML文件（ROS标准格式）"""
        try:
            translation = x_matrix[:3, 3].tolist()
            rotation_matrix = x_matrix[:3, :3].tolist()
            
            # 转换为四元数
            r = R.from_matrix(rotation_matrix)
            quaternion = r.as_quat().tolist()
            
            # 创建YAML数据结构
            yaml_data = {
                'hand_eye_calibration': {
                    'transform': {
                        'translation': {
                            'x': translation[0],
                            'y': translation[1],
                            'z': translation[2]
                        },
                        'rotation': {
                            'x': quaternion[0],
                            'y': quaternion[1],
                            'z': quaternion[2],
                            'w': quaternion[3]
                        }
                    },
                    'timestamp': str(Time()),
                    'sample_count': len(self.robot_transforms)
                }
            }
            
            # 保存到YAML文件
            filename = 'hand_eye_calibration.yaml'
            with open(filename, 'w') as f:
                yaml.dump(yaml_data, f, default_flow_style=False)
            
            self.logger.info(f'Calibration result saved to {filename}')
            
        except Exception as e:
            self.logger.error(f'Error saving calibration result to YAML: {e}')

    def load_calibration_yaml(self):
        """从YAML文件加载标定结果"""
        try:
            filename = 'hand_eye_calibration.yaml'
            
            # 检查文件是否存在
            
            if not os.path.exists(filename):
                self.logger.info(f'Calibration file {filename} not found')
                return None
            
            # 从YAML文件加载数据
            with open(filename, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            # 提取变换数据
            calibration_data = yaml_data['hand_eye_calibration']['transform']
            
            # 从四元数创建旋转矩阵
            quat = [
                calibration_data['rotation']['x'],
                calibration_data['rotation']['y'],
                calibration_data['rotation']['z'],
                calibration_data['rotation']['w']
            ]
            
            r = R.from_quat(quat)
            rotation_matrix = r.as_matrix()
            
            # 构造4x4变换矩阵
            x_matrix = np.eye(4)
            x_matrix[:3, :3] = rotation_matrix
            x_matrix[0, 3] = calibration_data['translation']['x']
            x_matrix[1, 3] = calibration_data['translation']['y']
            x_matrix[2, 3] = calibration_data['translation']['z']
            
            self.logger.info(f'Calibration matrix loaded from {filename}')
            
            # 记录加载的信息
            timestamp = yaml_data['hand_eye_calibration'].get('timestamp', 'unknown')
            sample_count = yaml_data['hand_eye_calibration'].get('sample_count', 'unknown')
            self.logger.info(f'Loaded calibration from {timestamp} with {sample_count} samples')
            
            return x_matrix
            
        except FileNotFoundError:
            self.logger.info(f'Calibration file {filename} not found, will create new calibration')
            return None
        except KeyError as e:
            self.logger.warning(f'Invalid calibration file format, missing key: {e}')
            return None
        except Exception as e:
            self.logger.error(f'Error loading calibration from YAML: {e}')
            return None
    def perform_calibration(self):
        # 根据图片计算标定板相对于相机的变换矩阵
        imgs = self.load_img()
        self.calculate_board_pose(imgs)
        # 加载数据
        self.robot_transforms, _ = self.load_calibration_data()
        
        self.logger.info('Performing hand-eye calibration...')
        # AX=XB
        # 1. 将变换转换为矩阵形式
        a_matrices = self.transforms_to_matrices(self.robot_transforms)
        b_matrices = self.transforms_to_matrices(self.camera_transforms)
        # 执行手眼标定算法，如Tsai-Lenz方法
        x_matrix = self.tsai_lenz_calibration(a_matrices, b_matrices)
        # 提取结果
        translation = x_matrix[:3, 3]
        rotation_matrix = x_matrix[:3, :3]
        # 转换为欧拉角
        r = R.from_matrix(rotation_matrix)
        euler_angles = r.as_euler('xyz', degrees=False)

         # 输出结果
        result = f"""
        Hand-Eye Calibration Result:
        Translation: [{translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}] meters 
        Rotation Matrix:
        {rotation_matrix} 
        Euler Angles (XYZ): [{euler_angles[0]:.6f}, {euler_angles[1]:.6f}, {euler_angles[2]:.6f}] radians
        """
        
        self.logger.info(result)
        self.logger.info('Calibration completed!')
        self.validate_calibration(x_matrix)
        self.save_calibration_yaml(x_matrix)
    
    def calculate_board_pose(self, imgs):
        # 转换为灰度图
        for cv_image in imgs:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # 查找棋盘格角点
            # 或者使用更宽松的检测
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

            ret, corners = cv2.findChessboardCorners(gray, self.board_size, None, flags)
            self.logger.info('ret='+str(ret))
            if ret:
                # 精细化角点位置
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # 绘制角点
                cv2.drawChessboardCorners(cv_image, self.board_size, corners, ret)
                
                # 计算标定板的姿态
                self.camera_transforms.append(self.estimate_pose(corners))
    def load_img(self):
        imgs = []
        img_path = "./imgs"
        for file_name in os.listdir(img_path):
            if file_name.endswith('.jpg') :
                imgs.append(cv2.imread(os.path.join(img_path, file_name)))
        return imgs

    def create_object_points(self):
        """创建棋盘格的3D点坐标（世界坐标系）"""
        # x = -0.009330
        # y = 0.753438
        # z = 0.757194
        # roll = 1.575357
        # pitch = -0.000001
        # yaw = -1.584065
        objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp
    def estimate_pose(self, corners):
        """估计标定板相对于相机的姿态"""
        # 相机内参
        self.logger.info('Estimating pose...')
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
            self.logger.info(f'Translation: {tvec.flatten()}')
            self.logger.info(f'Rotation Matrix:\n{rotation_matrix}')
            # 发布标定板变换
            transform = TransformStamped()
            transform.header.stamp = Time()
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
            return transform
        return None
        
    def validate_calibration(self, x_matrix):
        """验证标定结果的准确性"""
        self.logger.info('Validating calibration results...')
        
        a_matrices = self.transforms_to_matrices(self.robot_transforms)
        b_matrices = self.transforms_to_matrices(self.camera_transforms)
        
        errors = []
        
        # 对每组数据验证 AX = XB
        for i in range(len(a_matrices)):
            A = a_matrices[i]
            B = b_matrices[i]
            X = x_matrix
            
            # 计算 AX 和 XB
            ax = A @ X
            xb = X @ B
            
            # 计算误差
            error_matrix = ax - xb
            error_norm = np.linalg.norm(error_matrix)
            errors.append(error_norm)
            
            self.logger.info(f'Sample {i}: Error norm = {error_norm:.6f}')
        
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        self.logger.info(f'Validation Results:')
        self.logger.info(f'Mean error: {mean_error:.6f}')
        self.logger.info(f'Std error: {std_error:.6f}')
        
        if mean_error < 0.01:  # 阈值可以根据需要调整
            self.logger.info('Calibration validation: PASSED')
        else:
            self.logger.error('Calibration validation: FAILED - High error')
        
        return mean_error < 0.01
    def transforms_to_matrices(self, transforms):
        """将TransformStamped列表转换为4x4矩阵列表"""
        matrices = []
        for transform in transforms:
            # 创建4x4齐次变换矩阵
            matrix = np.eye(4)
            
            # 设置平移部分
            matrix[0, 3] = transform.transform.translation.x
            matrix[1, 3] = transform.transform.translation.y
            matrix[2, 3] = transform.transform.translation.z
            
            # 设置旋转部分（从四元数转换为旋转矩阵）
            quat = [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ]
            r = R.from_quat(quat)
            matrix[:3, :3] = r.as_matrix()
            
            matrices.append(matrix)
        
        return matrices
    
    def tsai_lenz_calibration(self, a_matrices, b_matrices):
        """使用Tsai-Lenz方法进行手眼标定"""
        """Tsai-Lenz算法不是梯度下降，也不是简单的均值计算
            算法特点：
            非迭代优化：Tsai-Lenz是一个非迭代的解析方法，不是梯度下降
            线性最小二乘：通过线性代数方法直接求解最优解
            全局解：一次性求得全局最优解，不需要逐步逼近"""
        n = len(a_matrices)
        if n != len(b_matrices):
            raise ValueError("A和B矩阵数量必须相等")
        
        # 计算相邻变换之间的差值
        M_A = []
        M_B = []
        
        for i in range(n - 1):
            # 计算A_i^-1 * A_{i+1}
            a_diff = np.linalg.inv(a_matrices[i]) @ a_matrices[i + 1]
            # 计算B_i^-1 * B_{i+1}  
            b_diff = np.linalg.inv(b_matrices[i]) @ b_matrices[i + 1]
            
            M_A.append(a_diff)
            M_B.append(b_diff)
        
        # 提取旋转部分并计算alpha和beta向量
        alpha = []
        beta = []
        
        for i in range(len(M_A)):
            # 从变换矩阵中提取旋转矩阵
            RA = M_A[i][:3, :3]
            RB = M_B[i][:3, :3]
            
            # 计算旋转矩阵的对数映射（旋转向量）
            ra = R.from_matrix(RA)
            rb = R.from_matrix(RB)
            
            omega_a = ra.as_rotvec()  # 旋转向量
            omega_b = rb.as_rotvec()  # 旋转向量
            
            # 构造alpha和beta向量
            alpha.append(omega_a)
            beta.append(omega_b)
        
        # 使用最小二乘法求解旋转部分
        # 构造线性方程组 C * rx = 0
        C = []
        for i in range(len(alpha)):
            # 使用向量叉积构造矩阵
            ai = alpha[i]
            bi = beta[i]
            
            # 构造 3x3 矩阵 [ai]× - [bi]×
            ai_cross = np.array([
                [0, -ai[2], ai[1]],
                [ai[2], 0, -ai[0]],
                [-ai[1], ai[0], 0]
            ])
            
            bi_cross = np.array([
                [0, -bi[2], bi[1]],
                [bi[2], 0, -bi[0]],
                [-bi[1], bi[0], 0]
            ])
            
            # 将3x3矩阵展平为行向量
            c_row = (ai_cross - bi_cross).flatten()
            C.append(c_row)
        
        C = np.array(C)
        
        # 使用SVD求解齐次线性方程组
        try:
            _, _, vt = np.linalg.svd(C)
            # 从9维解中提取有意义的信息
            solution = vt[-1, :]
            
            # 简单修复：使用solution的范数作为旋转向量的长度
            if np.linalg.norm(solution) > 1e-10:
                # 归一化
                normalized_solution = solution / np.linalg.norm(solution)
                # 取前3个元素作为旋转向量（这可能需要根据具体实现调整）
                rotvec = normalized_solution[:3] * np.pi  # 缩放到合理范围
            else:
                rotvec = np.array([0, 0, 0])
            
            # 构造旋转矩阵
            rx = R.from_rotvec(rotvec).as_matrix()
            
        except np.linalg.LinAlgError:
            self.logger.error("旋转矩阵计算失败，使用单位矩阵")
            rx = np.eye(3)
            return
        

        
        D = []
        e = []
        # (I - RA) * tx = RA * tb - ta
        for i in range(len(M_A)):
            RA = M_A[i][:3, :3]
            ta = M_A[i][:3, 3]
            tb = M_B[i][:3, 3]
            
            # 构造系数矩阵 (I - RA)
            d_block = np.eye(3) - RA
            D.append(d_block)
            
            # 构造常数向量 (RA * tb - ta)
            e_block = RA @ tb - ta
            e.append(e_block)
        
        # 将所有方程组合成一个大的线性系统
        d_matrix = np.vstack(D)
        e_vector = np.hstack(e)
        
        # 使用最小二乘法求解平移
        try:
            tx, residuals, rank, s = np.linalg.lstsq(d_matrix, e_vector, rcond=None)
        except np.linalg.LinAlgError:
            self.logger.warning("平移向量计算失败，使用零向量")
            tx = np.zeros(3)
        
        # 构造最终的4x4变换矩阵X
        X = np.eye(4)
        X[:3, :3] = rx
        X[:3, 3] = tx
        
        return X

    def load_calibration_data(self):
        """从JSON文件加载标定数据"""
        try:
            
            filename = 'calibration_data.json'
            
            # 检查文件是否存在
            if not os.path.exists(filename):
                print(f'Calibration data file {filename} not found')
                return None, None
            
            # 从JSON文件加载数据
            with open(filename, 'r') as f:
                calibration_data = json.load(f)
            
            # 提取数据
            robot_transforms_data = calibration_data.get('robot_transforms', [])
            camera_transforms_data = calibration_data.get('camera_transforms', [])
            
            robot_transforms = self.data_to_transforms(robot_transforms_data)
            camera_transforms = self.data_to_transforms(camera_transforms_data)
            self.logger.info(f'Loaded {len(robot_transforms_data)} robot transforms and {len(camera_transforms_data)} camera transforms from {filename}')
            
            return robot_transforms, camera_transforms
            
        except FileNotFoundError:
            self.logger.info(f'Calibration data file {filename} not found')
            return None, None
        except json.JSONDecodeError as e:
            self.logger.error(f'Error decoding JSON from {filename}: {e}')
            return None, None
        except Exception as e:
            self.logger.error(f'Error loading calibration data from {filename}: {e}')
            return None, None
    
    def data_to_transforms(self, transform_data_list):
        """将保存的数据转换为TransformStamped对象列表"""
        
        transforms = []
        
        for data in transform_data_list:
            transform = TransformStamped()
            
            # 设置基本字段（这里使用默认值，因为在保存时没有保存完整header信息）
            transform.header.stamp = Time()
            transform.header.frame_id = "base_link"  # 或根据需要设置
            
            # 设置平移
            transform.transform.translation.x = float(data['translation'][0])
            transform.transform.translation.y = float(data['translation'][1])
            transform.transform.translation.z = float(data['translation'][2])
            
            # 设置旋转
            transform.transform.rotation.x = float(data['rotation'][0])
            transform.transform.rotation.y = float(data['rotation'][1])
            transform.transform.rotation.z = float(data['rotation'][2])
            transform.transform.rotation.w = float(data['rotation'][3])
            
            transforms.append(transform)
        
        return transforms



def main():
    node = HandEyeCalibrationNode()
    node.perform_calibration()

if __name__ == '__main__':
    main()