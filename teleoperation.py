import pygame
import pickle as pkl
import numpy as np
from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
import rospy

class GamepadTeleoperation:
    def __init__(self):
        # 初始化pygame和手柄
        pygame.init()
        pygame.joystick.init()
        
        if pygame.joystick.get_count() == 0:
            raise ValueError("未检测到游戏手柄！")
            
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
        # 机器人初始化
        self.fa = FrankaArm()
        self.fa.reset_joints()
        self.current_pose = self.fa.get_pose()
        self.target_pose = self.current_pose.copy()
        
        # 控制参数
        self.max_velocity = 0.1  # 最大速度 10cm/s
        self.max_angular_velocity = 0.5  # 最大角速度
        
        # 数据记录
        self.trajectory_data = []
        self.is_recording = False
        
        # ROS设置
        self.pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=10)
        self.rate = rospy.Rate(50)
        
        self.fa.goto_pose(self.target_pose, duration=1000, dynamic=True, buffer_time=1000)
        self.init_time = rospy.Time.now().to_time()
        
    def update_target_pose(self):
        """根据手柄输入更新目标位姿"""
        pygame.event.pump()
        
        # 获取摇杆输入 (-1到1)
        left_x = self.joystick.get_axis(0)   # 左摇杆X轴
        left_y = self.joystick.get_axis(1)   # 左摇杆Y轴
        right_x = self.joystick.get_axis(2)  # 右摇杆X轴
        right_y = self.joystick.get_axis(3)  # 右摇杆Y轴
        
        # 获取扳机输入
        lt = self.joystick.get_axis(4)       # 左扳机 (上升)
        rt = self.joystick.get_axis(5)       # 右扳机 (下降)
        
        # 死区处理
        deadzone = 0.1
        if abs(left_x) < deadzone: left_x = 0
        if abs(left_y) < deadzone: left_y = 0
        if abs(right_x) < deadzone: right_x = 0
        if abs(right_y) < deadzone: right_y = 0
        
        # 计算位置增量 (dt = 1/50 = 0.02s)
        dt = 0.02
        position_delta = np.array([
            -left_y * self.max_velocity * dt,    # 前后移动
            left_x * self.max_velocity * dt,     # 左右移动
            (rt - lt) * self.max_velocity * dt   # 上下移动
        ])
        
        # 更新目标位置
        self.target_pose.translation += position_delta
        
        # 旋转控制 (可选)
        if abs(right_x) > deadzone or abs(right_y) > deadzone:
            # 这里可以添加旋转控制逻辑
            pass
            
        # 按钮处理
        if self.joystick.get_button(0):  # A键 - 开始/停止录制
            self.toggle_recording()
        if self.joystick.get_button(1):  # B键 - 保存数据
            self.save_trajectory()
            
    def toggle_recording(self):
        """切换录制状态"""
        self.is_recording = not self.is_recording
        print(f"Recording: {'ON' if self.is_recording else 'OFF'}")
        
    def save_trajectory(self):
        """保存轨迹"""
        if self.trajectory_data:
            filename = f"gamepad_traj_{int(time.time())}.pkl"
            with open(filename, 'wb') as f:
                pkl.dump([data['pose'] for data in self.trajectory_data], f)
            print(f"轨迹已保存: {filename}")
            
    def run(self):
        """主控制循环"""
        print("游戏手柄遥操作已启动!")
        print("- 左摇杆: XY移动")
        print("- 扳机: 上下移动")
        print("- A键: 录制开关")
        print("- B键: 保存轨迹")
        
        msg_id = 0
        try:
            while True:
                self.update_target_pose()
                
                # 发送控制指令
                timestamp = rospy.Time.now().to_time() - self.init_time
                traj_msg = PosePositionSensorMessage(
                    id=msg_id,
                    timestamp=timestamp,
                    position=self.target_pose.translation,
                    quaternion=self.target_pose.quaternion
                )
                
                ros_msg = make_sensor_group_msg(
                    trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                        traj_msg, SensorDataMessageType.POSE_POSITION)
                )
                
                self.pub.publish(ros_msg)
                
                # 记录数据
                if self.is_recording:
                    self.trajectory_data.append({
                        'timestamp': timestamp,
                        'pose': self.fa.get_pose(),
                        'target_pose': self.target_pose.copy()
                    })
                
                msg_id += 1
                self.rate.sleep()
                
        except KeyboardInterrupt:
            print("遥操作已停止")
            self.stop_robot()

if __name__ == "__main__":
    controller = GamepadTeleoperation()
    controller.run()