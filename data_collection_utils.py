from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
import numpy as np
import rospy

# 修改data_collection_utils.py
def motion2array(translation: np.ndarray, rotation_angles: np.ndarray) -> np.ndarray:
    """
    将运动数据转换为6维数组
    translation: [x, y, z] - 位移增量
    rotation_angles: [rx, ry, rz] - 欧拉角增量
    """
    return np.concatenate([translation, rotation_angles])

def array2motion(array: np.ndarray) -> tuple:
    """
    将6维数组转换回运动数据
    返回: (translation_delta, rotation_angles)
    """
    return array[:3], array[3:]

# 现有的pose函数保持不变，用于其他地方
def pose2array(pose: RigidTransform) -> np.ndarray:
    """转换累积位姿为7维数组（用于当前状态存储）"""
    return np.concatenate([pose.translation, pose.quaternion])

def array2pose(array: np.ndarray) -> RigidTransform:
    """从7维数组重建位姿"""
    return RigidTransform(
        translation=array[:3],
        rotation=RigidTransform.rotation_from_quaternion(array[3:])
    )

def publish_pose(pose:RigidTransform, id:int, timestamp:float, pub:rospy.Publisher, rate:rospy.Rate):
    """
    通过ros发布动作，实现连续操作，其中内涵了sleep
    """
    timestamp = timestamp
    traj_gen_proto_msg = PosePositionSensorMessage(
        id=id, timestamp=timestamp,
        position=pose.translation, quaternion=pose.quaternion
    )
    ros_msg = make_sensor_group_msg(
        trajectory_generator_sensor_msg=sensor_proto2ros_msg(
            traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
    )
    # rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
    pub.publish(ros_msg)
    rate.sleep()

