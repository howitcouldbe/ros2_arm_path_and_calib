from dataclasses import dataclass

@dataclass
class MotionTask:
    """运动任务数据结构"""
    task_id: str
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    group_name: str
    response_container: dict  # 用于存放响应结果

    @staticmethod
    def from_dict(data: dict) -> 'MotionTask':
        return MotionTask(
            task_id=data.get("task_id", ""),
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
            z=data.get("z", 0.0),
            roll=data.get("roll", 0.0),
            pitch=data.get("pitch", 0.0),
            yaw=data.get("yaw", 0.0),
            group_name=data.get("group_name", "manipulator"),
            response_container={}
        )