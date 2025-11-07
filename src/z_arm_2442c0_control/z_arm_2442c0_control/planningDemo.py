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
import queue
from .MotionTask import MotionTask 
from flask import Flask, request, jsonify, Response
import threading
from typing import Optional
import json

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

        # 添加任务队列和控制变量
        self.task_queue = queue.Queue()
        self.current_task: Optional[MotionTask] = None
        self.task_interrupted = False
        self.task_status_listeners = {}  # 任务状态监听器字典
    
    # region 单独线程启动HTTP服务
    def start_http_service(self, host='127.0.0.1', port=8083):
        """启动 HTTP 服务"""
        # 直接使用当前实例的方法
        server_thread = threading.Thread(
            target=self._run_http_server, 
            args=(host, port),
            daemon=True
        )
        server_thread.start()  # HTTP服务已经在独立线程中运行
        self.get_logger().info(f"HTTP service started on {host}:{port}")
    # endregion
    # region http请求处理方法
    def _run_http_server(self, host='127.0.0.1', port=8083):
        """运行 HTTP 服务器的内部方法"""
        
        app = Flask(__name__)
        app.config['JSON_AS_ASCII'] = False  
        app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
        @app.route('/plan', methods=['POST'])
        def handle_plan():
            try:
                json_data = request.get_json()
                self.get_logger().info(f"Received plan request: {json_data}")
            except Exception as e:
                return jsonify({"status": "error", "message": f"无效的 JSON 数据: {str(e)}"})

            def generate():
                try:
                    if not json_data:
                        yield f"data: {json.dumps({'status': 'error', 'message': '请求体为空'})}\n\n"
                        return
                    
                    group_input = str(json_data.get("group_code"))
                    
                    if group_input in ['1', '2', '3']:
                        if group_input == '1':
                            group_name = "camera_1_group"
                        elif group_input == '2':
                            group_name = "camera_2_group"
                        else:
                            group_name = "camera_3_group"
                    else:
                        yield f"data: {json.dumps({'status': 'error', 'message': '无效的组代号'})}\n\n"
                        return
                        
                    x = float(json_data.get("x"))
                    y = float(json_data.get("y"))
                    z = float(json_data.get("z"))
                    roll = float(json_data.get("roll"))
                    pitch = float(json_data.get("pitch"))
                    yaw = float(json_data.get("yaw"))
                    
                    # 中断当前任务
                    self.interrupt_current_task()
                    
                    # 创建任务
                    task = MotionTask(
                        task_id=str(time.time()),  # 简单的任务ID
                        x=x, y=y, z=z,
                        roll=roll, pitch=pitch, yaw=yaw,
                        group_name=group_name,
                        response_container=None
                    )

                    # 创建任务状态监听器
                    status_updates = queue.Queue()
                    self.task_status_listeners[task.task_id] = status_updates
                    
                    # 将任务加入队列
                    self.task_queue.put(task)

                    yield f"data: {json.dumps({'status': 'accepted', 'task_id': task.task_id, 'message': '任务已接收'})}\n\n"
                    

                    
                    try:
                        # 持续监听任务状态更新
                        while True:
                            try:
                                # 等待状态更新（设置超时以允许定期检查）
                                status_update = status_updates.get(timeout=1.0)
                                
                                # 发送状态更新
                                yield f"data: {json.dumps(status_update)}\n\n"
                                
                                # 如果任务完成，结束流
                                if status_update.get('status') in ['success', 'error']:
                                    print("任务完成")
                                    yield f"data: {json.dumps({'status': 'completed', 'message': '任务执行完成'})}\n\n"
                                    break
                                    
                            except queue.Empty:
                                # 检查客户端是否断开连接
                                # 这里可以通过其他方式检测客户端断开
                                continue
                                
                    finally:
                        # 清理监听器
                        if task.task_id in self.task_status_listeners:
                            del self.task_status_listeners[task.task_id]

                    
                except (ValueError, TypeError) as e:
                    yield f"data: {json.dumps({'status': 'error', 'message': f'输入错误: {str(e)}'})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'status': 'error', 'message': f'服务器错误: {str(e)}'})}\n\n"
            return Response(generate(), mimetype='text/event-stream', headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*'
            })
        @app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({"status": "healthy"})
        
        app.run(host=host, port=port, debug=False, use_reloader=False)
    # endregion
    def interrupt_current_task(self):
        """中断当前任务"""
        if self.current_task is not None:
            self.task_interrupted = True
            self.get_logger().info("当前任务已被标记为中断")
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
                json_data:MotionTask = self.task_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            # 执行运动序列
            success = self.execute_motion_sequence(json_data.x, 
                                                   json_data.y, 
                                                   json_data.z, 
                                                   json_data.roll, 
                                                   json_data.pitch, 
                                                   json_data.yaw,json_data.task_id)
            if not success:
                self.get_logger().error("运动序列执行失败")
                status_queue:queue.Queue = self.task_status_listeners[json_data.task_id]
                status_queue.put({"status": "error", "message": "运动序列执行失败"})
            else:
                self.get_logger().info("运动序列执行成功")
                status_queue:queue.Queue = self.task_status_listeners[json_data.task_id]
                status_queue.put({"status": "success", "message": "运动序列执行成功"})
    #endregion
    
    #region 三段式运动序列
    def execute_motion_sequence(self, x, y, z, roll, pitch, yaw,task_id):
        """执行完整的运动序列"""
        # 1. 创建目标可视化
        self.visualization.create_goal_marker(x, y, z, radius=0.001)
        
        # 2. 创建目标约束区域
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [0.001]  # 1mm半径
        
        # 3. 第一段：粗略运动
        self.task_status_listeners[task_id].put({"status": "info", "message": "OMPL开始第一段粗略运动..."})
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
            self.task_status_listeners[task_id].put({"status": "error", "message": "第一段轨迹执行未完成或超时"})
            return False
        self.get_logger().info("第一段轨迹执行完成")
        self.task_status_listeners[task_id].put({"status": "info", "message": "第一段轨迹执行完成"})
        
        # 5. 第二段：精确微调
        request.start_state = self.state_provider.get_current_robot_state()
        request.start_state.is_diff = False
        request.goal_constraints[0] = Constraints()  # 清空旧约束
        request = self.constraint_builder.add_joint_constraints_to_prevent_large_rotations(request, max_rotation=math.pi)
        self.get_logger().info("开始第二段精确微调...,PTP规划")
        self.task_status_listeners[task_id].put({"status": "info", "message": "开始第二段精确微调...,PTP规划"})
        success = self.execute_precise_adjustment(request, planner_id='PTP', velocity_factor=0.5, acceleration_factor=0.5)
        if not success:
            self.task_status_listeners[task_id].put({"status": "error", "message": "第二段精确微调失败"})
            return False
        self.task_status_listeners[task_id].put({"status": "info", "message": "第二段精确微调完成"})
        # 6. 第三段：超精细微调
        self.task_status_listeners[task_id].put({"status": "info", "message": "开始第三段精确微调"})
        self.get_logger().info("开始第三段精确微调...,LIN规划")
        success = self.execute_precise_adjustment(request, planner_id='LIN', velocity_factor=0.05, acceleration_factor=0.05)
        if not success:
            self.task_status_listeners[task_id].put({"status": "error", "message": "第三段精确微调失败"})
            return False
        self.task_status_listeners[task_id].put({"status": "info", "message": "第三段精确微调完成"})
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
            replan=True  # 关闭重规划避免不确定行为
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
    node.start_http_service()
    node.send_sequence()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
