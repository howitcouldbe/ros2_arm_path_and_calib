import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from moveit_configs_utils import MoveItConfigsBuilder
import numpy as np
import time
from moveit_msgs.msg import JointConstraint
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint, RobotState
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import BoundingVolume
from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import Marker
from moveit_msgs.msg import CollisionObject
from moveit_msgs.msg import PlanningScene
from moveit_msgs.msg import PlanningOptions
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from moveit_msgs.msg import TrajectoryConstraints
from control_msgs.msg import JointTrajectoryControllerState
import math



class PilzSequenceClient(Node):
    def __init__(self):
        super().__init__('standard_move_group_client')
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self.expected_joint_names = ['fixed_lift_joint', 'lift_revo_joint', 'double_revo_1_joint','double_revo_2_joint']  # 替换为你的 group 的关节名
        self.group_name = "camera_1_group"
        self.current_joint_positions = None
        self.joint_state_names = None
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.marker_pub = self.create_publisher(Marker, "/visualization_marker", 10)
        self.trajectory_pub = self.create_publisher(JointTrajectory, '/camera_controller/joint_trajectory', 10)
    
    def joint_state_callback(self, msg):
        self.joint_state_names = msg.name
        self.current_joint_positions = msg.position
    
    def is_joint_state_valid(self):
        return self.current_joint_positions is not None and len(self.current_joint_positions) >= len(self.expected_joint_names)

    def send_sequence(self):
        # 等待完整状态
        while not self.is_joint_state_valid():
            self.get_logger().warn("等待完整的 joint_states 到达...")
            rclpy.spin_once(self, timeout_sec=0.1)


        self.get_logger().info('等待服务器...')
        self._action_client.wait_for_server()
        self.add_box_obstacle("box1")
        time.sleep(1)

        # 创建一个球体
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [0.1]  # 半径为 5cm

        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.type = Marker.SPHERE

        while True:
            #外部输入获取x，y，z的坐标
            x, y, z = map(float, input("请输入x,y,z坐标（用空格分隔）：").split())
            roll,pitch,yaw = 0.0,0.0,0.0
            roll,pitch,yaw = map(float, input("请输入末端绕x轴，Y轴，Z轴旋转的角度（用空格分隔）：").split()) # roll 绕x轴旋转 pitch 绕y轴旋转 yaw 绕z轴旋转
            
            # region 创建目标区域
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = z
            marker.scale.x = 0.02
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            self.marker_pub.publish(marker)
            # endregion
            # region 调用ompl获取轨迹
            trajectory,pose,request_ = self.get_trajectory_by_ompl(x,y,z,roll,pitch,yaw,primitive)
            if trajectory is None:
                self.get_logger().error("获取轨迹失败")
                continue
            joint_trajectory = trajectory.joint_trajectory
            self.trajectory_pub.publish(joint_trajectory)
            final_point = joint_trajectory.points[-1]
            self.wait_until_trajectory_done(joint_trajectory.joint_names, final_point.positions)
            # endregion

            # region 终点附近的精确微调
            print(len(request_.goal_constraints))
            orientation_constraint = OrientationConstraint()
            orientation_constraint.orientation = pose.orientation  # 目标姿态
            orientation_constraint.link_name = "camera_1_link"
            orientation_constraint.header.frame_id = "base_link"
            orientation_constraint.absolute_x_axis_tolerance = 0.01
            orientation_constraint.absolute_y_axis_tolerance = 0.01
            orientation_constraint.absolute_z_axis_tolerance = 0.01
            request_.goal_constraints[0].orientation_constraints.append(orientation_constraint)
            goal_msg = self.create_goal(request_, 'PTP', 'pilz_industrial_motion_planner', 0.5, 0.5, plan_only=False, look_around=False, replan=False)
            if self.check_ik_reachable(pose):
                self.get_logger().info('发送规划请求...，第二段精确移动')
                future = self._action_client.send_goal_async(goal_msg)
                rclpy.spin_until_future_complete(self, future)
                goal_handle = future.result()
                if not goal_handle.accepted:
                    self.get_logger().error("第二段目标被拒绝")
                    return

                result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(self, result_future)
                result = result_future.result().result
                self.get_logger().info(f'第二段执行完成，状态码：{result.error_code.val}')
            else:
                self.get_logger().info('but not reachable')
            # endregion
    
    def wait_until_trajectory_done(self, joint_names, target_positions, tol=0.01, timeout=30.0):
        start_time = time.time()
        done = False

        def callback(msg):
            nonlocal done
            print("状态 joint_names:", msg.joint_names)
            print("轨迹 joint_names:", joint_names)

            actual = dict(zip(msg.joint_names, msg.actual.positions))
            print("当前状态：", actual)
            print("目标位置：", target_positions)
            print("误差：", [abs(actual[name] - target) for name, target in zip(joint_names, target_positions)])

            done = all(abs(actual[name] - target) < tol for name, target in zip(joint_names, target_positions))

        sub = self.create_subscription(JointTrajectoryControllerState, '/camera_controller/state', callback, 10)

        while not done and time.time() - start_time < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info("done"+str(done)+"-----"+str(time.time() - start_time))
        self.destroy_subscription(sub)
        if done:
            self.get_logger().info("✅ 第一段轨迹执行完成")
        else:
            self.get_logger().warn("⚠️ 第一段轨迹执行超时或未完成")
    
    def check_ik_reachable(self, target_pose):
        ik_request = PositionIKRequest()
        ik_request.group_name = self.group_name
        ik_request.ik_link_name = ""
        ik_request.pose_stamped = PoseStamped()
        ik_request.pose_stamped.header.frame_id = "base_link"
        ik_request.pose_stamped.pose = target_pose
        ik_request.timeout.sec = 1
        # ik_request.attempts = 5

        req = GetPositionIK.Request()
        req.ik_request = ik_request

        client = self.create_client(GetPositionIK, '/compute_ik')
        self.get_logger().info('等待 IK 服务...')
        client.wait_for_service()

        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result().error_code.val == future.result().error_code.SUCCESS:
            self.get_logger().info('✅ IK 可达，准备发送规划请求')
            return True
        else:
            self.get_logger().warn('❌ IK 不可达，取消规划')
            return False
    
    def add_box_obstacle(self, name="box1"):
        collision_object = CollisionObject()
        collision_object.header.frame_id = "base_link"
        collision_object.id = name

        # 定义障碍物形状（立方体）
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.05, 0.05, 0.05]  # 长宽高 20cm

        # 定义障碍物位置
        box_pose = PoseStamped()
        box_pose.header.frame_id = "base_link"
        box_pose.pose.position.x = 0.3
        box_pose.pose.position.y = 0.0
        box_pose.pose.position.z = 1.0
        box_pose.pose.orientation.w = 1.000000

        collision_object.primitives.append(box)
        collision_object.primitive_poses.append(box_pose.pose)
        collision_object.operation = CollisionObject.ADD

        # 发布到 planning scene
        self.scene_pub = self.create_publisher(PlanningScene, "/planning_scene", 10)
        planning_scene = PlanningScene()
        planning_scene.is_diff = True
        planning_scene.world.collision_objects.append(collision_object)
        self.scene_pub.publish(planning_scene)
        self.get_logger().info(f"✅ 添加障碍物 {name} 到场景")


    #方法名：通过ompl获取轨迹
    def get_trajectory_by_ompl(self, x,y,z,roll,pitch,yaw,primitive):
        r = R.from_euler('xyz', [np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw)])
        qx, qy, qz, qw = r.as_quat()
        print("度数",np.deg2rad(roll),",",np.deg2rad(pitch),",",np.deg2rad(yaw),"换算的弧度",qx,",",qy,",",qz,",",qw)
        # 设置目标位姿
        pose = self.set_goal_pose(x, y, z, qx, qy, qz, qw)
        # 设置约束条件
        position_constraint,orientation_constraint = self.set_constraints(primitive,pose)
    
        # 构造 MotionPlanRequest
        request = self.set_motion_plan_request(position_constraint,orientation_constraint)

        # 构造目标消息
        goal_msg = self.create_goal(request,'RRTConnect', 'ompl', 0.5, 0.5, True, True,True)

        future = self._action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal was rejected by the action server.")
            return None
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        print("Planner:", goal_msg.request.planner_id)
        print("Pipeline:", goal_msg.request.pipeline_id)

        if result.error_code.val != 1:  # 1 表示成功
            self.get_logger().error("OMPL 规划失败，错误码：" + str(result.error_code.val))
            return None, None,None
        trajectory = result.planned_trajectory
        joint_names = trajectory.joint_trajectory.joint_names
        print("joint_names in trajectory:", joint_names)
        return trajectory,pose,request
    
    #创建目标位置
    def set_goal_pose(self, x, y, z, qx, qy, qz, qw)->PoseStamped:
        # 设置球体的位置   
        pose = Pose()
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        return pose

    # 创建约束
    def set_constraints(self, primitive,pose)->Constraints:
        # 创建约束区域
        bounding_volume = BoundingVolume()
        bounding_volume.primitives.append(primitive)
        bounding_volume.primitive_poses.append(pose)

        # 设置到 PositionConstraint
        position_constraint = PositionConstraint()
        position_constraint.link_name = "camera_1_link"
        position_constraint.header.frame_id = "base_link"  # 或 base_link
        position_constraint.constraint_region = bounding_volume
        position_constraint.target_point_offset.x = 0.0
        position_constraint.target_point_offset.y = 0.0
        position_constraint.target_point_offset.z = 0.0
        position_constraint.weight = 1.0

         # 设置 姿态约束
        orientation_constraint = OrientationConstraint()
        orientation_constraint.orientation = pose.orientation  # 目标姿态
        orientation_constraint.link_name = "camera_1_link"
        orientation_constraint.header.frame_id = "base_link"
        orientation_constraint.absolute_x_axis_tolerance = 0.01
        orientation_constraint.absolute_y_axis_tolerance = 0.01
        orientation_constraint.absolute_z_axis_tolerance = 0.01
        return position_constraint,orientation_constraint
    
    # 构造MotionPlanRequest
    def set_motion_plan_request(self, position_constraint,orientation_constraint)->MotionPlanRequest:
        request = MotionPlanRequest()
        request.group_name = self.group_name  # 替换为你的规划组名
        request.goal_constraints.append(Constraints())
        request.goal_constraints[0].position_constraints.append(position_constraint)
        request.start_state.is_diff = True
        request.start_state = self.get_current_robot_state()
        # request.goal_constraints[0].orientation_constraints.append(orientation_constraint)
        return request
    def get_current_robot_state(self) -> RobotState:
        robot_state = RobotState()
        joint_state = JointState()
        joint_state.name = self.joint_state_names
        joint_state.position = self.current_joint_positions
        robot_state.joint_state = joint_state
        return robot_state

    # 构造目标消息
    def create_goal(
            self,
            request, 
            planner_id='PTP',
            pipeline_id='pilz_industrial_motion_planner',
            velocity_factor=0.5,
            acceleration_factor=0.5,
            plan_only=False,
            look_around=False,
            replan=False)-> MoveGroup.Goal:
        goal_msg = MoveGroup.Goal()
        goal_msg.request = request
        goal_msg.request.planner_id = planner_id
        goal_msg.request.pipeline_id = pipeline_id
        goal_msg.request.allowed_planning_time = 5.0
        
        goal_msg.request.max_velocity_scaling_factor = velocity_factor  # 合理值，表示 50% 最大速度
        goal_msg.request.max_acceleration_scaling_factor = acceleration_factor  # 同样设置加速度缩放

        goal_msg.planning_options.plan_only = plan_only # 是否只生成路径而不执行
        goal_msg.planning_options.look_around = look_around #是否在规划失败时尝试“环顾四周”寻找其他路径
        goal_msg.planning_options.replan = replan #如果执行失败，是否自动重新规划
        return goal_msg


def main(args=None):
    rclpy.init(args=args)
    node = PilzSequenceClient()
    node.send_sequence()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
