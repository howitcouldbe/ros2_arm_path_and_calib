from moveit_msgs.msg import BoundingVolume
from moveit_msgs.msg import PositionConstraint,OrientationConstraint,JointConstraint,MotionPlanRequest
import math


class ConstraintBuilder:
    """约束构建"""
    
    def __init__(self, group_name, group_to_ee_map):
        self.group_name = group_name
        self.group_to_ee_map = group_to_ee_map
    
    def create_position_constraint(self, primitive, pose):
        bounding_volume = BoundingVolume()
        bounding_volume.primitives.append(primitive)
        bounding_volume.primitive_poses.append(pose)

        position_constraint = PositionConstraint()
        position_constraint.link_name = self.group_to_ee_map[self.group_name]
        position_constraint.header.frame_id = "base_link"
        position_constraint.constraint_region = bounding_volume
        position_constraint.target_point_offset.x = 0.0
        position_constraint.target_point_offset.y = 0.0
        position_constraint.target_point_offset.z = 0.0
        position_constraint.weight = 1.0
        
        return position_constraint
    
    def create_orientation_constraint(self, pose):
        orientation_constraint = OrientationConstraint()
        orientation_constraint.orientation = pose.orientation
        orientation_constraint.link_name = self.group_to_ee_map[self.group_name]
        orientation_constraint.header.frame_id = "base_link"
        orientation_constraint.absolute_x_axis_tolerance = 0.01
        orientation_constraint.absolute_y_axis_tolerance = 0.01
        orientation_constraint.absolute_z_axis_tolerance = 0.01
        return orientation_constraint
    # region 添加关节约束防止大范围旋转
    def add_joint_constraints_to_prevent_large_rotations(self, request:MotionPlanRequest,max_rotation=math.pi):
        """添加关节约束防止大范围旋转"""
        current_state = request.start_state
        
        if current_state.joint_state.name and current_state.joint_state.position:
            for i, joint_name in enumerate(current_state.joint_state.name):
                if joint_name == "door_joint":
                    continue  # 跳过门关节
                current_pos = current_state.joint_state.position[i]
                
                # 为每个关节添加约束
                constraint = JointConstraint()
                constraint.joint_name = joint_name
                constraint.position = current_pos  # 以当前位置为中心
                constraint.tolerance_above = max_rotation  # 限制最大旋转
                constraint.tolerance_below = max_rotation
                constraint.weight = 0.1  # 较小权重，只是引导而非强制
                
                request.goal_constraints[0].joint_constraints.append(constraint)
        return request
