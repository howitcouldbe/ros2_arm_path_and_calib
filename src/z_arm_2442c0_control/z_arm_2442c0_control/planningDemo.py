import rclpy
from rclpy.node import Node
from moveit_msgs.msg import MotionPlanRequest, Constraints, OrientationConstraint
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R
from .VisualizationManager import VisualizationManager
from .RobotStateProvider import RobotStateProvider
from .CollisionManager import CollisionManager
from .TrajectoryExecutor import TrajectoryExecutor
from .MotionPlanner import MotionPlanner
from .ConstraintBuilder import ConstraintBuilder
import time
import numpy as np
import math

class PilzSequenceClient(Node):
    """"""
    
    def __init__(self):
        super().__init__('standard_move_group_client')
        
        # 依赖注入各个组件
        self.state_provider = RobotStateProvider(self)
        self.visualization = VisualizationManager(self)
        self.collision_manager = CollisionManager(self)
        self.trajectory_executor = TrajectoryExecutor(self)
        self.motion_planner = MotionPlanner(self)
        self.constraint_builder = None
        self.group_name = "camera_1_group"
        self.group_to_ee_map = {
            "camera_1_group": "camera_1_link",
            "camera_2_group": "camera_2_link", 
            "camera_3_group": "camera_3_link"
        }
    
    def initialize(self):
        """初始化系统"""
        while not self.state_provider.is_valid():
            self.get_logger().warn("等待完整的 joint_states 到达...")
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info('等待服务器...')
        self.motion_planner.wait_for_server()
        self.collision_manager.add_box_obstacle("box1")
        time.sleep(1)
    
    #region 发送运动序列
    def send_sequence(self):
        self.initialize()
        
        while True:
            try:
                group_input = input("请输入规划组代号（1:camera_1_group, 2:camera_2_group, 3:camera_3_group）：")
                if group_input in ['1', '2', '3']:
                    if group_input == '1':
                        self.group_name = "camera_1_group"
                    elif group_input == '2':
                        self.group_name = "camera_2_group"
                    else:
                        self.group_name = "camera_3_group"
                else:
                    print("无效的规划组代号，请重新输入")
                    continue
                
                x, y, z = map(float, input("请输入x,y,z坐标（用空格分隔）：").split())
                roll, pitch, yaw = map(float, input("请输入末端绕x轴，Y轴，Z轴旋转的角度（用空格分隔）：").split())
            except ValueError:
                print("输入错误，请重新输入")
                continue

            # 执行运动序列
            success = self.execute_motion_sequence(x, y, z, roll, pitch, yaw)
            if not success:
                self.get_logger().error("运动序列执行失败")
    #endregion
    
    #region 三段式运动序列
    def execute_motion_sequence(self, x, y, z, roll, pitch, yaw):
        """执行完整的运动序列"""
        # 1. 创建目标可视化
        self.visualization.create_goal_marker(x, y, z, radius=0.001)
        
        # 2. 创建目标约束区域
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [0.001]  # 1mm半径
        
        # 3. 第一段：粗略运动
        self.constraint_builder = ConstraintBuilder(self.group_name, self.group_to_ee_map)
        trajectory, _, request = self.get_trajectory_by_ompl(x, y, z, roll, pitch, yaw, primitive)
        if trajectory is None:
            return False
            
        # 4. 执行并等待第一段完成
        final_point = self.trajectory_executor.publish_trajectory(trajectory)
        done = self.trajectory_executor.wait_until_done(
            trajectory.joint_trajectory.joint_names, 
            final_point.positions, 
            tol=0.01  # 粗略容差
        )
        if not done:
            self.get_logger().error("第一段轨迹执行未完成或超时")
            return False
        self.get_logger().info("第一段轨迹执行完成")
        
        # 5. 第二段：精确微调
        request.start_state = self.state_provider.get_current_robot_state()
        request.start_state.is_diff = False
        request.goal_constraints[0] = Constraints()  # 清空旧约束
        request = self.constraint_builder.add_joint_constraints_to_prevent_large_rotations(request, max_rotation=math.pi)

        time.sleep(10)

        self.get_logger().info("开始第二段精确微调...,PTP规划")
        success = self.execute_precise_adjustment(request, planner_id='PTP', velocity_factor=0.5, acceleration_factor=0.5)
        if not success:
            return False
        time.sleep(10)
        self.get_logger().info("开始第三段精确微调...,LIN规划")
        success = self.execute_precise_adjustment(request, planner_id='LIN', velocity_factor=0.05, acceleration_factor=0.05)
        if not success:
            return False
        return success
    #endregion
    
    #region 精确微调
    def execute_precise_adjustment(self, request,planner_id,velocity_factor=0.5, acceleration_factor=0.5):
        """执行精确微调"""
        
        # 创建高精度目标
        goal_msg = self.motion_planner.create_goal(
            request, 
            planner_id=planner_id, 
            pipeline_id='pilz_industrial_motion_planner', 
            velocity_factor=velocity_factor,    # 低速确保精度
            acceleration_factor=acceleration_factor,
            plan_only=False, 
            look_around=False, 
            replan=False  # 关闭重规划避免不确定行为
        )

        self.get_logger().info('发送规划请求...')
        result = self.motion_planner.execute_plan(goal_msg)
        
        if result is None:
            self.get_logger().error("目标被拒绝")
            return False
            
        if result.error_code.val == result.error_code.SUCCESS:
            self.get_logger().info(f'执行完成，状态码：{result.error_code.val}')
            return True
        else:
            self.get_logger().warn(f'规划失败，状态码：{result.error_code.val}')
            return False
    #endregion
    
    # region 获取轨迹
    def get_trajectory_by_ompl(self, x, y, z, roll, pitch, yaw, primitive):
        """通过OMPL获取轨迹"""
        r = R.from_euler('xyz', [np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw)])
        qx, qy, qz, qw = r.as_quat()
        
        pose = self.set_goal_pose(x, y, z, qx, qy, qz, qw)
        position_constraint = self.constraint_builder.create_position_constraint(primitive, pose)
        
        request = MotionPlanRequest()
        request.group_name = self.group_name
        request.goal_constraints.append(Constraints())
        request.goal_constraints[0].position_constraints.append(position_constraint)
        request.start_state.is_diff = True
        request.start_state = self.state_provider.get_current_robot_state()
        
        goal_msg = self.motion_planner.create_goal(
            request, 'RRTstar', 'ompl', 1.0, 1.0, True, False, True)

        result = self.motion_planner.execute_plan(goal_msg)
        
        if result is None or result.error_code.val != 1:
            self.get_logger().error("OMPL 规划失败")
            return None, None, None
            
        return result.planned_trajectory, pose, request
    # endregion
    
    # region 设置目标位置
    def set_goal_pose(self, x, y, z, qx, qy, qz, qw):
        pose = Pose()
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        return pose
    # endregion

    
def main(args=None):
    rclpy.init(args=args)
    node = PilzSequenceClient()
    node.send_sequence()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
