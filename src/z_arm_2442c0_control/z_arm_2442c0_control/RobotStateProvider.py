from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState


class RobotStateProvider:
    """机器人状态信息"""
    
    def __init__(self, node):
        self.node = node
        self.current_joint_positions = None
        self.joint_state_names = None
        self.subscription = node.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
    
    def get_current_robot_state(self) -> RobotState:
        robot_state = RobotState()
        joint_state = JointState()
        joint_state.name = self.joint_state_names
        joint_state.position = self.current_joint_positions
        robot_state.joint_state = joint_state
        return robot_state
    
    def joint_state_callback(self, msg):
        self.joint_state_names = msg.name
        self.current_joint_positions = msg.position
    
    def is_valid(self):
        return (self.current_joint_positions is not None and 
                len(self.current_joint_positions) >= 4)
