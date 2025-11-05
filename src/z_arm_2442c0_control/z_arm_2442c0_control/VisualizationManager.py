from visualization_msgs.msg import Marker

class VisualizationManager:
    """可视化"""
    
    def __init__(self, node):
        self.marker_pub = node.create_publisher(Marker, "/visualization_marker", 10)
    
    def create_goal_marker(self, x, y, z, radius=0.002):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.type = Marker.SPHERE
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.scale.x = radius * 2
        marker.scale.y = radius * 2
        marker.scale.z = radius * 2
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.marker_pub.publish(marker)
