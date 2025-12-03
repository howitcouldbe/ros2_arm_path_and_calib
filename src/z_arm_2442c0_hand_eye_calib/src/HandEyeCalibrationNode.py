#!/usr/bin/env python3
# hand_eye_calibration_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml

# 手眼标定节点
class HandEyeCalibrationNode(Node):
    def __init__(self):
        super().__init__('hand_eye_calibration_node')

        # 尝试加载已保存的标定结果
        self.calibration_matrix = self.load_calibration_yaml()
        if self.calibration_matrix is not None:
            self.get_logger().info('Loaded existing calibration matrix')
            # self.use_calibration_for_transform(self.robot_transforms)  # 测试使用
        else:
            self.get_logger().info('No existing calibration found, will perform new calibration')
        
        # TF监听器，用于获取机器人变换
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.transform_subscription = self.create_subscription(
            TransformStamped,
            '/calibration_board/transform',
            self.save_sample_callback,
            10
        )
        
        # 订阅关节状态
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10)
        
        # 存储观测数据
        self.robot_transforms = []  # 机器人末端相对于基座的变换
        self.camera_transforms = []  # 相机观察到的标定板变换
        
        # 标定触发服务或定时器
        self.calibration_timer = self.create_timer(5.0, self.collect_sample)
        
        self.sample_count = 0
        self.max_samples = 30  # 收集10组数据进行标定
        
        self.get_logger().info('Hand Eye Calibration Node started')
    def use_calibration_for_transform(self, object_in_camera_matrix):
        """使用标定矩阵将相机坐标系中的点转换到机器人坐标系"""
        if self.calibration_matrix is None:
            self.get_logger().warn('No calibration matrix available!')
            return None
        
        # object_in_camera_matrix: 物体在相机坐标系中的位姿
        # camera_in_base_matrix: 相机在机器人基座坐标系中的位姿（需要另外传入）
        # 返回: 物体在机器人基座坐标系中的位姿
        # 注意：这里的X是相机相对于末端的变换
        end_effector_in_base = self.get_end_effector_pose()
        if end_effector_in_base is None:
            self.get_logger().warn('Failed to get end effector pose!')
            return None
        return end_effector_in_base @ self.calibration_matrix @ object_in_camera_matrix
    
    def get_end_effector_pose(self):
        """
        获取末端执行器相对于基座的当前位姿
        """
        try:
            # 查询末端执行器相对于基座的变换
            transform = self.tf_buffer.lookup_transform(
                'base_link',           # target_frame (基座坐标系)
                'tool0' or 'ee_link',  # source_frame (末端执行器坐标系)
                rclpy.time.Time()      # 获取最新的变换
            )
            
            # 转换为4x4矩阵
            matrix = np.eye(4)
            matrix[0, 3] = transform.transform.translation.x
            matrix[1, 3] = transform.transform.translation.y
            matrix[2, 3] = transform.transform.translation.z
            
            quat = [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ]
            r = R.from_quat(quat)
            matrix[:3, :3] = r.as_matrix()
            
            return matrix
        except Exception as e:
            self.get_logger().warn(f'Failed to get end effector pose: {e}')
            return None
    def save_sample_callback(self, msg):
        """保存一个观测数据"""
        self.camera_transforms.append(msg)
         # 获取机器人末端相对于基座的变换
        robot_transform = self.tf_buffer.lookup_transform(
            'base_link',      # target frame
            'camera_1_link',  # source frame
            rclpy.time.Time())
        self.robot_transforms.append(robot_transform)
        self.sample_count += 1
    def joint_callback(self, msg):
        """处理关节状态更新"""
        # 这里可以根据关节角度计算正向运动学
        # 但由于你已有TF系统，可以直接从TF获取
        pass
    
    def collect_sample(self):
        """收集一组样本数据"""
        if self.sample_count >= self.max_samples:
            self.calibration_matrix = self.load_calibration_yaml()
            if self.calibration_matrix is not None:
                self.get_logger().info('Loaded existing calibration matrix')
                self.save_calibration_data(self.robot_transforms,self.camera_transforms)  # 测试使用
                self.validate_calibration(self.calibration_matrix)  # 测试使用
            else:
                self.get_logger().info('No existing calibration found, will perform new calibration')
                self.perform_calibration()
                self.sample_count = 0
                self.camera_transforms = []
                self.robot_transforms = []
            return
            
        try:
            # self.camera_transforms.append(camera_transform)
            self.get_logger().info(f'Collected sample {self.sample_count}/{self.max_samples}')
            
        except Exception as e:
            self.get_logger().warn(f'Failed to collect sample: {e}')
    
    def simulate_camera_transform(self):
        """模拟相机获取的标定板变换（实际应用中应从标定板检测获得）"""
        rng = np.random.default_rng(42)
        # 这只是一个模拟，实际应该从标定板检测节点获取
        transform = TransformStamped()
        transform.transform.translation.x = rng.uniform(-0.1, 0.1)
        transform.transform.translation.y = rng.uniform(-0.1, 0.1)
        transform.transform.translation.z = rng.uniform(0.5, 1.0)
        
        # 随机旋转
        r = R.from_euler('xyz', [rng.uniform(-np.pi, np.pi) for _ in range(3)])
        quat = r.as_quat()  # x, y, z, w
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]
        
        return transform
    
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
                    'timestamp': str(self.get_clock().now().to_msg().sec),
                    'sample_count': len(self.robot_transforms)
                }
            }
            
            # 保存到YAML文件
            filename = 'hand_eye_calibration.yaml'
            with open(filename, 'w') as f:
                yaml.dump(yaml_data, f, default_flow_style=False)
            
            self.get_logger().info(f'Calibration result saved to {filename}')
            
        except Exception as e:
            self.get_logger().error(f'Error saving calibration result to YAML: {e}')

    def load_calibration_yaml(self):
        """从YAML文件加载标定结果"""
        try:
            filename = 'hand_eye_calibration.yaml'
            
            # 检查文件是否存在
            import os
            if not os.path.exists(filename):
                self.get_logger().info(f'Calibration file {filename} not found')
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
            
            self.get_logger().info(f'Calibration matrix loaded from {filename}')
            
            # 记录加载的信息
            timestamp = yaml_data['hand_eye_calibration'].get('timestamp', 'unknown')
            sample_count = yaml_data['hand_eye_calibration'].get('sample_count', 'unknown')
            self.get_logger().info(f'Loaded calibration from {timestamp} with {sample_count} samples')
            
            return x_matrix
            
        except FileNotFoundError:
            self.get_logger().info(f'Calibration file {filename} not found, will create new calibration')
            return None
        except KeyError as e:
            self.get_logger().warn(f'Invalid calibration file format, missing key: {e}')
            return None
        except Exception as e:
            self.get_logger().error(f'Error loading calibration from YAML: {e}')
            return None
    def perform_calibration(self):
        """执行手眼标定算法"""
        if len(self.robot_transforms) < 3:
            self.get_logger().warn('标定样本不足，无法进行标定'+str(len(self.robot_transforms)))
            self.sample_count = len(self.robot_transforms)
            return

        if len(self.robot_transforms) != len(self.camera_transforms):
            self.get_logger().warn('机器人和相机数据数量不匹配'+str(len(self.robot_transforms))+' vs '+str(len(self.camera_transforms)))
            return
        
        self.get_logger().info('Performing hand-eye calibration...')
        self.save_calibration_data(self.robot_transforms,self.camera_transforms)
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
        
        # 简化版输出结果
        
        self.get_logger().info(result)
        self.get_logger().info('Calibration completed!')
        self.validate_calibration(x_matrix)
        self.save_calibration_yaml(x_matrix)
    
    def save_calibration_data(self,robot_transforms,camera_transforms):
        """保存标定数据到JSON文件"""
        try:
            import json
            
            # 创建简化数据结构
            calibration_data = {
                'timestamp': self.get_clock().now().to_msg().sec,
                'sample_count': len(robot_transforms),
                'robot_transforms': [],
                'camera_transforms': []
            }
            
            # 保存机器人变换数据
            for transform in robot_transforms:
                if isinstance(transform, TransformStamped):
                    robot_data = {
                        'translation': [
                            float(transform.transform.translation.x),
                            float(transform.transform.translation.y),
                            float(transform.transform.translation.z)
                        ],
                        'rotation': [
                            float(transform.transform.rotation.x),
                            float(transform.transform.rotation.y),
                            float(transform.transform.rotation.z),
                            float(transform.transform.rotation.w)
                        ]
                    }
                    calibration_data['robot_transforms'].append(robot_data)
            
            # 保存相机变换数据
            for transform in camera_transforms:
                if isinstance(transform, TransformStamped):
                    camera_data = {
                        'translation': [
                            float(transform.transform.translation.x),
                            float(transform.transform.translation.y),
                            float(transform.transform.translation.z)
                        ],
                        'rotation': [
                            float(transform.transform.rotation.x),
                            float(transform.transform.rotation.y),
                            float(transform.transform.rotation.z),
                            float(transform.transform.rotation.w)
                        ]
                    }
                    calibration_data['camera_transforms'].append(camera_data)
            
            # 保存到JSON文件
            filename = 'calibration_data.json'
            with open(filename, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            self.get_logger().info(f'Calibration data saved to {filename} with {len(robot_transforms)} samples')
            
        except Exception as e:
            self.get_logger().error(f'Error saving calibration data to JSON: {e}')
    def validate_calibration(self, x_matrix):
        """验证标定结果的准确性"""
        self.get_logger().info('Validating calibration results...')
        
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
            
            self.get_logger().info(f'Sample {i}: Error norm = {error_norm:.6f}')
        
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        self.get_logger().info(f'Validation Results:')
        self.get_logger().info(f'Mean error: {mean_error:.6f}')
        self.get_logger().info(f'Std error: {std_error:.6f}')
        
        if mean_error < 0.01:  # 阈值可以根据需要调整
            self.get_logger().info('Calibration validation: PASSED')
        else:
            self.get_logger().warn('Calibration validation: FAILED - High error')
        
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
            self.get_logger().warn("旋转矩阵计算失败，使用单位矩阵")
            rx = np.eye(3)
        

        
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
            self.get_logger().warn("平移向量计算失败，使用零向量")
            tx = np.zeros(3)
        
        # 构造最终的4x4变换矩阵X
        X = np.eye(4)
        X[:3, :3] = rx
        X[:3, 3] = tx
        
        return X

def main(args=None):
    rclpy.init(args=args)
    node = HandEyeCalibrationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

def validate_transform():
    robot_data,camera_data = load_calibration_data()
    robot_transforms = data_to_transforms(robot_data)
    camera_transforms = data_to_transforms(camera_data)
    """验证标定结果的准确性"""
    self.get_logger().info('Validating calibration results...')
    
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
        
        self.get_logger().info(f'Sample {i}: Error norm = {error_norm:.6f}')
    
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    print(f'Validation Results:')
    print(f'Mean error: {mean_error:.6f}')
    print(f'Std error: {std_error:.6f}')
    
    if mean_error < 0.01:  # 阈值可以根据需要调整
        print('Calibration validation: PASSED')
    else:
        print('Calibration validation: FAILED - High error')
    
    return mean_error < 0.01

def data_to_transforms(self, transform_data_list):
    """将保存的数据转换为TransformStamped对象列表"""
    from geometry_msgs.msg import TransformStamped
    import builtin_interfaces.msg
    
    transforms = []
    
    for data in transform_data_list:
        transform = TransformStamped()
        
        # 设置基本字段（这里使用默认值，因为在保存时没有保存完整header信息）
        transform.header.stamp = self.get_clock().now().to_msg()
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

def load_calibration_data():
    """从JSON文件加载标定数据"""
    try:
        import json
        
        filename = 'calibration_data.json'
        
        # 检查文件是否存在
        import os
        if not os.path.exists(filename):
            print(f'Calibration data file {filename} not found')
            return None, None
        
        # 从JSON文件加载数据
        with open(filename, 'r') as f:
            calibration_data = json.load(f)
        
        # 提取数据
        robot_transforms_data = calibration_data.get('robot_transforms', [])
        camera_transforms_data = calibration_data.get('camera_transforms', [])
        
        # 转换为TransformStamped对象列表
        robot_transforms = self.data_to_transforms(robot_transforms_data)
        camera_transforms = self.data_to_transforms(camera_transforms_data)
        
        self.get_logger().info(f'Loaded {len(robot_transforms)} robot transforms and {len(camera_transforms)} camera transforms from {filename}')
        
        return robot_transforms, camera_transforms
        
    except FileNotFoundError:
        self.get_logger().info(f'Calibration data file {filename} not found')
        return None, None
    except json.JSONDecodeError as e:
        self.get_logger().error(f'Error decoding JSON from {filename}: {e}')
        return None, None
    except Exception as e:
        self.get_logger().error(f'Error loading calibration data from {filename}: {e}')
        return None, None
    
def calculate_calibration_matrix(A, B, X):
    """计算AX和XB之间的误差"""
    AX = A @ X
    XB = X @ B
    error_matrix = AX - XB
    error_norm = np.linalg.norm(error_matrix)
    return error_norm
if __name__ == '__main__':
    main()