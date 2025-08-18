from input_device.spacemouse import FrankaSpacemouse  # 统一使用绝对导入
from data_collection_utils import pose2array, array2pose, publish_pose, motion2array
from shared_memory.shared_memory_queue import SharedMemoryQueue
from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from franka_interface_msgs.msg import SensorDataGroup
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage
import numpy as np
import rospy
from multiprocessing.managers import SharedMemoryManager

class Sm_franka_teleop:
    def __init__(self, T: float, dt: float):
        """
        初始化SpaceMouse遥操作控制器
        
        Args:
            T: 总操作时间（秒）
            dt: 控制时间步长（秒）
        """
        # 机械臂初始化
        self.fa = FrankaArm()
        self.fa.reset_joints()
        self.current_pose = self.fa.get_pose()
        self.target_pose = self.current_pose.copy()
        
        # 时间和频率参数
        self.T = T
        self.dt = dt
        self.frequency = 1 / dt
        
        # ROS发布器和速率控制
        self.rate = rospy.Rate(self.frequency)
        self.publisher = rospy.Publisher(
            FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, 
            SensorDataGroup, 
            queue_size=1  # 低延迟设置
        )
        
        # 控制状态
        self.init_time = None
        self.index = 0
        self.trajectory_queue = None  # 在run_teleoperation中初始化
        
    def run_teleoperation(self):
        """执行SpaceMouse遥操作"""
        with SharedMemoryManager() as shm_manager:
            # 创建轨迹队列（需要SharedMemoryManager）
            self.trajectory_queue = SharedMemoryQueue.create_from_examples({
                'motion_6d': np.zeros(6),        # [dx,dy,dz,drx,dry,drz]
                'current_pose_7d': np.zeros(7),  # [x,y,z,qw,qx,qy,qz]
                'timestamp': 0.0,
                'sequence_id': 0
            }, buffer_size=10000)
            
            # 创建SpaceMouse控制器
            spacemouse = FrankaSpacemouse(
                shm_manager, 
                frequency=self.frequency,
                deadzone=0.02,               # 2%死区
                position_sensitivity=1,   # 降低平移灵敏度
                rotation_sensitivity=1,  # 降低旋转灵敏度
                debug=False  # 关闭调试输出，提升性能
            )
            
            rospy.loginfo(f"开始SpaceMouse遥操作 - 时长: {self.T}s, 频率: {self.frequency}Hz")
            rospy.loginfo("按Ctrl+C停止操作")
            
            with spacemouse:
                try:
                    total_iterations = int(self.T * self.frequency)
                    
                    for i in range(total_iterations):
                        self.index += 1
                        
                        # 读取SpaceMouse输入
                        motion = spacemouse.get_motion_state()
                        
                        # 计算控制增量
                        translation_delta = motion[:3] * self.dt
                        rotation_angles = motion[3:] * self.dt
                        
                        # 应用平移
                        self.target_pose.translation += translation_delta
                        
                        # 应用旋转（仅在有显著旋转时）
                        if np.linalg.norm(rotation_angles) > 1e-6:
                            rotation_matrix_delta = RigidTransform.rotation_from_euler(
                                rotation_angles, axes='xyz'
                            )
                            self.target_pose.rotation = self.target_pose.rotation @ rotation_matrix_delta
                        
                        # 记录轨迹数据
                        if self.init_time is not None:  # 确保时间已初始化
                            self.trajectory_queue.put({
                                'motion_6d': motion2array(translation_delta, rotation_angles),
                                'current_pose_7d': pose2array(self.target_pose),
                                'timestamp': rospy.Time.now().to_time() - self.init_time,
                                'sequence_id': self.index
                            })
                        
                        # 发送控制指令
                        if self.index == 1:
                            # 首次启动动态控制
                            rospy.loginfo("启动动态位姿控制...")
                            self.fa.goto_pose(
                                self.target_pose, 
                                duration=self.T, 
                                dynamic=True, 
                                buffer_time=10,
                                cartesian_impedances=[600.0, 600.0, 600.0, 50.0, 50.0, 50.0]
                            )
                            self.init_time = rospy.Time.now().to_time()
                        else:
                            # 发布连续控制指令
                            timestamp = rospy.Time.now().to_time() - self.init_time
                            publish_pose(
                                self.target_pose, 
                                self.index, 
                                timestamp, 
                                pub=self.publisher, 
                                rate=self.rate
                            )
                        
                        # 性能监控（可选）
                        if self.index % 100 == 0:  # 每秒输出一次状态
                            rospy.loginfo(f"已执行 {self.index} 步，剩余 {total_iterations - self.index} 步")
                        
                except KeyboardInterrupt:
                    rospy.loginfo("用户中断操作")
                except Exception as e:
                    rospy.logerr(f"遥操作错误: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # 优雅停止
                    rospy.loginfo("正在停止机械臂...")
                    try:
                        self.fa.stop_skill()
                    except:
                        pass
                
                rospy.loginfo(f"遥操作结束 - 共记录 {self.index} 个轨迹点")
                
                # 可选：保存轨迹到文件
                if self.trajectory_queue and self.trajectory_queue.qsize() > 0:
                    rospy.loginfo(f"队列中有 {self.trajectory_queue.qsize()} 个数据点可用于回放")

    def get_recorded_trajectory(self):
        """获取记录的轨迹数据"""
        if self.trajectory_queue and not self.trajectory_queue.empty():
            return self.trajectory_queue.get_all()
        return None

# 使用示例
if __name__ == "__main__":
    import rospy
    rospy.init_node('spacemouse_teleoperation', anonymous=True)
    
    # 创建遥操作控制器
    teleop = Sm_franka_teleop(T=30.0, dt=0.02)  # 30秒，50Hz
    
    try:
        teleop.run_teleoperation()
    except Exception as e:
        rospy.logerr(f"主程序错误: {e}")