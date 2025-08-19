import pickle as pkl
from data_collection_frankapy_continous import gripper_open_flag
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
    gripper_thres = 0.05

    rospy.loginfo('Generating Trajectory')

    traj_dict = pkl.load(open('franka_traj_my.pkl','rb')) # 一个字典，里面每个元素是一个包含了多个array的array
    pose_len = len(traj_dict['current_pose_7d']) # 用pose len动态控制执行时间
    pose_traj = []
    for i, pose_array in enumerate(traj_dict['current_pose_7d']):
        # 将旋转转换为四元数
        pose = RigidTransform(
            translation=pose_array[:3],
            rotation=RigidTransform.rotation_from_quaternion(pose_array[3:]),
            from_frame = 'franka_tool',
            to_frame= 'world'  # 假设目标坐标系是世界坐标系
        )
        # 获取对应时间点的夹爪状态
        gripper_state = traj_dict['keyboard_states'][i]
        pose_with_grip = {
            'pose': pose,
            'gripper_state': gripper_state
        }
        pose_traj.append(pose_with_grip)

    T = 30 # 总执行时间
    dt = 0.02 # 时间步长 —— 频率100 Hz
    #! ts = np.arange(0, T, dt) # 10s * 100HZ = 1000个时间点 —— 1000个时间点，每个时间点执行一个轨迹点
    #! 自适应执行时间
    ts = pose_len
    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=10)
    rate = rospy.Rate(1 / dt)

    rospy.loginfo('Publishing pose trajectory...')
    # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
    # 为了确保轨迹执行完成，设置的buffer时间比实际需要的时间长
    fa.goto_pose(pose_traj[1]['pose'], duration=T, dynamic=True, buffer_time=30)
    
    init_time = rospy.Time.now().to_time()
    previous_gripper_state = None  # 跟踪上一个夹爪状态，避免重复操作
    
    for i in range(2, len(pose_traj)):
        timestamp = rospy.Time.now().to_time() - init_time
        current_pose = pose_traj[i]['pose']
        current_gripper_state = pose_traj[i]['gripper_state']
        
        # 发布位置指令
        traj_gen_proto_msg = PosePositionSensorMessage(
            id=i, timestamp=timestamp,
            position=current_pose.translation, 
            quaternion=current_pose.quaternion
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
        )
        
        rospy.loginfo('Publishing: ID {} - Gripper: {}'.format(i, 'CLOSED' if current_gripper_state else 'OPEN'))
        pub.publish(ros_msg)
        
        # 检查是否需要改变夹爪状态
        if previous_gripper_state is None or current_gripper_state != previous_gripper_state:
            rospy.loginfo('Changing gripper state to: {}'.format('CLOSED' if current_gripper_state else 'OPEN'))
            
            if current_gripper_state:  # True = 夹爪关闭
                fa.close_gripper()
            else:  # False = 夹爪打开
                fa.open_gripper()
            
            # 等待夹爪动作完成（重要！）
            rospy.sleep(1.0)  # 给夹爪足够时间完成动作
            
        previous_gripper_state = current_gripper_state
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
