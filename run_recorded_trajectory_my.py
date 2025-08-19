import pickle as pkl
import numpy as np

from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from scipy.spatial.transform import Rotation as R

import rospy

if __name__ == "__main__":
    fa = FrankaArm()
    fa.reset_joints()

    rospy.loginfo('Generating Trajectory')

    traj_dict = pkl.load(open('franka_traj_my.pkl','rb')) # 一个字典，里面每个元素是一个包含了多个array的array
    pose_traj = []
    for i, pose_array in enumerate(traj_dict['current_pose_7d']):
        # 将旋转转换为四元数
        pose = RigidTransform(
            translation=pose_array[:3],
            rotation=RigidTransform.rotation_from_quaternion(pose_array[3:]),
            from_frame = 'franka_tool',
            to_frame= 'world'  # 假设目标坐标系是世界坐标系
        )
        pose_traj.append(pose)

    T = 30 # 总执行时间
    dt = 0.02 # 时间步长 —— 频率100 Hz
    ts = np.arange(0, T, dt) # 10s * 100HZ = 1000个时间点 —— 1000个时间点，每个时间点执行一个轨迹点

    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=10)
    rate = rospy.Rate(1 / dt)

    rospy.loginfo('Publishing pose trajectory...')
    # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
    # 为了确保轨迹执行完成，设置的buffer时间比实际需要的时间长
    fa.goto_pose(pose_traj[1], duration=T, dynamic=True, buffer_time=10,
    )
    init_time = rospy.Time.now().to_time()
    for i in range(2, len(ts)):
        timestamp = rospy.Time.now().to_time() - init_time
        traj_gen_proto_msg = PosePositionSensorMessage(
            id=i, timestamp=timestamp,
            position=pose_traj[i].translation, quaternion=pose_traj[i].quaternion
		)
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
            )

        rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
        pub.publish(ros_msg)
        rate.sleep()

    # Stop the skill
    # Alternatively can call fa.stop_skill()
    term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - init_time, should_terminate=True)
    ros_msg = make_sensor_group_msg(
        termination_handler_sensor_msg=sensor_proto2ros_msg(
            term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
        )
    pub.publish(ros_msg)

    rospy.loginfo('Done')
