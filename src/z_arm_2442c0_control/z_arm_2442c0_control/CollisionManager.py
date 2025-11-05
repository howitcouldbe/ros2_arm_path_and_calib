from moveit_msgs.msg import PlanningScene
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState

class CollisionManager:
    """碰撞模块"""
    
    def __init__(self, node):
        self.scene_pub = node.create_publisher(PlanningScene, "/planning_scene", 10)
    
    def add_box_obstacle(self, name="box1", x=0.3, y=0.0, z=1.0, size=[0.05, 0.05, 0.05]):
        collision_object = CollisionObject()
        collision_object.header.frame_id = "base_link"
        collision_object.id = name

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = size

        box_pose = Pose()
        box_pose.position.x = x
        box_pose.position.y = y
        box_pose.position.z = z
        box_pose.orientation.w = 1.0

        collision_object.primitives.append(box)
        collision_object.primitive_poses.append(box_pose)
        collision_object.operation = CollisionObject.ADD

        planning_scene = PlanningScene()
        planning_scene.is_diff = True
        planning_scene.world.collision_objects.append(collision_object)
        self.scene_pub.publish(planning_scene)
        return collision_object
    
    def add_point_cloud_obstacle(self, point_cloud_data, name="irregular_hollow_object"):
        """使用点云数据添加不规则障碍物"""
        collision_object = CollisionObject()
        collision_object.header.frame_id = "base_link"
        collision_object.id = name
        
        # 将点云转换为多个小球体或立方体
        for point in point_cloud_data:
            # 方法1：使用小球体近似
            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.SPHERE
            primitive.dimensions = [0.01]  # 小球半径1cm
            
            pose = Pose()
            pose.position.x = point[0]
            pose.position.y = point[1]
            pose.position.z = point[2]
            pose.orientation.w = 1.0
            
            collision_object.primitives.append(primitive)
            collision_object.primitive_poses.append(pose)
        
        collision_object.operation = CollisionObject.ADD
        
        # 发布到规划场景
        planning_scene = PlanningScene()
        planning_scene.is_diff = True
        planning_scene.world.collision_objects.append(collision_object)
        self.scene_pub.publish(planning_scene)

    def get_current_state(self):
        robot_state = RobotState()
        joint_state = JointState()
        joint_state.name = self.joint_state_names
        joint_state.position = self.current_joint_positions
        robot_state.joint_state = joint_state
        return robot_state
