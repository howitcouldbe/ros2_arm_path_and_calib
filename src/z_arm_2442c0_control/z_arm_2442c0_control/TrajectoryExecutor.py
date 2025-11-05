from trajectory_msgs.msg import JointTrajectory
from control_msgs.msg import JointTrajectoryControllerState
import time
import math
import rclpy

class TrajectoryExecutor:
    """轨迹执行,监控"""
    
    def __init__(self, node):
        self.node = node
        self.trajectory_pub = node.create_publisher(JointTrajectory, 
                                                    '/camera_controller/joint_trajectory', 
                                                    10)
    
    def publish_trajectory(self, trajectory):
        joint_trajectory = trajectory.joint_trajectory
        self.trajectory_pub.publish(joint_trajectory)
        return joint_trajectory.points[-1]
    
    def wait_until_done(self, joint_names, target_positions, tol=0.02, timeout=60.0):
        def normalize_angle(angle):
            while angle > math.pi:
                angle -= 2 * math.pi
            while angle < -math.pi:
                angle += 2 * math.pi
            return angle
        
        def angle_difference(angle1, angle2):
            diff = abs(normalize_angle(angle1 - angle2))
            return diff
        
        normalized_targets = [normalize_angle(pos) for pos in target_positions]
        start_time = time.time()
        done = False

        def callback(msg):
            nonlocal done
            actual = dict(zip(msg.joint_names, msg.actual.positions))
            errors = []
            for name, target in zip(joint_names, normalized_targets):
                actual_normalized = normalize_angle(actual[name])
                target_normalized = normalize_angle(target)
                error = angle_difference(actual_normalized, target_normalized)
                errors.append(error)
            
            done = all(error < tol for error in errors)

        sub = self.node.create_subscription(
            JointTrajectoryControllerState, '/camera_controller/state', callback, 10)

        while not done and time.time() - start_time < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        
        self.node.destroy_subscription(sub)
        return done
