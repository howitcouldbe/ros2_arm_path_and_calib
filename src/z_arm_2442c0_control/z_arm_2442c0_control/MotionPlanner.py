from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
import rclpy

class MotionPlanner:
    """轨迹规划"""
    
    def __init__(self, node):
        self.node = node
        self._action_client = ActionClient(node, MoveGroup, 'move_action')
        self.group_to_ee_map = {
            "camera_1_group": "camera_1_link",
            "camera_2_group": "camera_2_link", 
            "camera_3_group": "camera_3_link"
        }
    
    def wait_for_server(self):
        self._action_client.wait_for_server()
    
    def create_goal(self, request, planner_id='PTP', pipeline_id='pilz_industrial_motion_planner',
                   velocity_factor=0.5, acceleration_factor=0.5, plan_only=False, 
                   look_around=False, replan=False, current_state=None):
        goal_msg = MoveGroup.Goal()
        goal_msg.request = request
        goal_msg.request.planner_id = planner_id
        goal_msg.request.pipeline_id = pipeline_id
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = velocity_factor
        goal_msg.request.max_acceleration_scaling_factor = acceleration_factor

        goal_msg.planning_options.plan_only = plan_only
        goal_msg.planning_options.look_around = look_around
        goal_msg.planning_options.replan = replan

        if current_state:
            goal_msg.planning_options.planning_scene_diff.robot_state.is_diff = False
            goal_msg.planning_options.planning_scene_diff.robot_state = current_state
        
        return goal_msg
    
    def execute_plan(self, goal_msg):
        future = self._action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self.node, future)
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            return None
            
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future)
        return result_future.result().result
