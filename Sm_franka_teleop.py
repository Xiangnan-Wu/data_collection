import sys
import os
# 添加当前目录和子目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'input_device'))
sys.path.insert(0, os.path.join(current_dir, 'shared_memory'))

<<<<<<< HEAD

# 导入pynput库替代keyboard（不需要sudo权限）
try:
   from pynput import keyboard as pynput_keyboard
   pynput_available = True
except ImportError:
   print("Warning: pynput not installed. Please install: pip install pynput")
   pynput_keyboard = None
   pynput_available = False


=======
# 导入pynput库替代keyboard（不需要sudo权限）
try:
    from pynput import keyboard as pynput_keyboard
    pynput_available = True
except ImportError:
    print("Warning: pynput not installed. Please install: pip install pynput")
    pynput_keyboard = None
    pynput_available = False
>>>>>>> 6bb17a265de97c1b510c6b8789ff36a36ea93bd6


# 直接导入模块文件
from input_device.spacemouse import FrankaSpacemouse
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
from scipy.spatial.transform import Rotation as R
import pickle as pkl

<<<<<<< HEAD



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
      self.fa.open_gripper()
      self.current_pose = self.fa.get_pose()
      self.target_pose = self.current_pose.copy()
      print(f" target pose from frame = {self.target_pose.from_frame}, to frame = {self.target_pose.to_frame}")
    
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
     
      # 键盘状态
      self.gripper_state = False
      self.recording_state = False
     
      # pynput键盘状态跟踪
      self.keys_pressed = set()  # 跟踪当前按下的键
      self.last_g_state = False
      self.last_r_state = False
 
  def update_keyboard_state(self):
      """更新键盘状态 - 在主循环中调用"""
      if not pynput_available:
          return
         
      # 检测当前按键状态（基于按键集合）
      current_g = 'g' in self.keys_pressed
      current_r = 'r' in self.keys_pressed
     
      # 检测按键按下事件（边沿触发，避免重复切换）
      if current_g and not self.last_g_state:
          self.gripper_state = not self.gripper_state
          rospy.loginfo(f"🤖 夹爪状态切换: {'打开' if self.gripper_state else '关闭'}")
         
      if current_r and not self.last_r_state:
          self.recording_state = not self.recording_state
          if self.recording_state:
              rospy.loginfo(f"🔴 开始录制机械臂状态...")
          else:
              rospy.loginfo(f"⏹️  停止录制机械臂状态")
     
      # 保存当前状态用于下次比较
      self.last_g_state = current_g
      self.last_r_state = current_r
     
      return {
          'gripper_state': self.gripper_state,
          'recording_state': self.recording_state
      }
 
  def on_key_press(self, key):
      """pynput按键按下回调"""
      try:
          # 普通字符键
          self.keys_pressed.add(key.char.lower())
      except AttributeError:
          # 特殊键（如Ctrl, Alt等）
          pass
 
  def on_key_release(self, key):
      """pynput按键释放回调"""
      try:
          # 普通字符键
          self.keys_pressed.discard(key.char.lower())
      except AttributeError:
          # 特殊键
          pass
    
  def run_teleoperation(self, save_traj:bool = False):
      """执行SpaceMouse遥操作"""
      with SharedMemoryManager() as shm_manager:
          # 创建轨迹队列（需要SharedMemoryManager）
          self.trajectory_queue = SharedMemoryQueue.create_from_examples(
              shm_manager,{
              'motion_6d': np.zeros(6),        # [dx,dy,dz,drx,dry,drz]
              'current_pose_7d': np.zeros(7),  # [x,y,z,qw,qx,qy,qz]
              'keyboard_states': False,    # gripper状态
              'timestamp': 0.0,
              'sequence_id': 0,
          }, buffer_size=10000)
        
          # 创建SpaceMouse控制器
          spacemouse = FrankaSpacemouse(
              shm_manager,
              frequency=self.frequency,
              deadzone=0.05,               # 2%死区
              position_sensitivity=0.15,   # 降低平移灵敏度
              rotation_sensitivity=0.3,  # 降低旋转灵敏度
              debug=False  # 关闭调试输出，提升性能
          )
        
          rospy.loginfo(f"开始SpaceMouse遥操作 - 时长: {self.T}s, 频率: {self.frequency}Hz")
          rospy.loginfo("🎮 控制说明:")
          rospy.loginfo("  - SpaceMouse: 控制机械臂移动")
          rospy.loginfo("  - 键盘 'R' 键: 开始/停止录制轨迹")
          rospy.loginfo("  - 键盘 'G' 键: 切换夹爪开/关")
          rospy.loginfo("  - Ctrl+C: 停止操作")
          rospy.loginfo("--------------------")
         
          # 启动键盘监听器（pynput）
          keyboard_listener = None
          if pynput_available:
              keyboard_listener = pynput_keyboard.Listener(
                  on_press=self.on_key_press,
                  on_release=self.on_key_release
              )
              keyboard_listener.start()
              rospy.loginfo("✅ 键盘监听已启动 (pynput)")
          else:
              rospy.logwarn("⚠️  键盘功能不可用 - 请安装: pip install pynput")
        
          with spacemouse:
              try:
                  total_iterations = int(self.T * self.frequency)
                
                  for i in range(total_iterations):
                      self.index += 1
                    
                      # 读取SpaceMouse输入
                      motion = spacemouse.get_motion_state()
                      # 更新键盘状态
                      keyboard_states = self.update_keyboard_state()
                      #! 为了控制spacemouse按照理想方式移动，需要对motion进行一定的修改
                      motion[0] = -motion[0]  # 反转X轴方向
                      motion[4] = -motion[4]
                      motion[3], motion[4] = motion[4], motion[3]
                      # 计算控制增量
                      translation_delta = motion[:3] * self.dt
                      rotation_angles = motion[3:] * self.dt
                    
                      # 应用平移
                      self.target_pose.translation += translation_delta
                    
                      # 应用旋转（仅在有显著旋转时）
                      if np.linalg.norm(rotation_angles) > 1e-6:
                          rotation_scipy = R.from_euler('xyz', rotation_angles)
                          rotation_matrix_delta = rotation_scipy.as_matrix()
                          self.target_pose.rotation = self.target_pose.rotation @ rotation_matrix_delta
                    
                      # 记录轨迹数据
                      if self.init_time is not None and self.recording_state:  # 确保时间已初始化
                          self.trajectory_queue.put({
                              'motion_6d': motion2array(translation_delta, rotation_angles),
                              'current_pose_7d': pose2array(self.target_pose),
                              'keyboard_states': self.gripper_state,
                              'timestamp': rospy.Time.now().to_time() - self.init_time,
                              'sequence_id': self.index,
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
                 
                  # 停止键盘监听器
                  if keyboard_listener:
                      keyboard_listener.stop()
                      rospy.loginfo("✅ 键盘监听已停止")
            
              rospy.loginfo(f"遥操作结束 - 共执行 {self.index} 个控制步")
            
              # 保存录制的轨迹数据
              if self.trajectory_queue and self.trajectory_queue.qsize() > 0:
                  rospy.loginfo(f"📊 队列中有 {self.trajectory_queue.qsize()} 个录制数据点")




                  # 在SharedMemoryManager上下文内保存数据
                  try:
                      # 使用peek_all()进行非消费性读取，保留数据供后续使用
                      traj_data = self.trajectory_queue.peek_all()
                      filename = "franka_traj_my.pkl"
                      with open(filename, 'wb') as f:
                          pkl.dump(traj_data, f)




                      # 显示保存信息
                      num_points = len(next(iter(traj_data.values())))
                      rospy.loginfo(f"✅ 录制的轨迹数据已保存到: {filename}")
                      rospy.loginfo(f"   📝 记录了 {num_points} 个有效数据点")
                      rospy.loginfo(f"   🤖 包含夹爪状态和机械臂轨迹信息")




                      # 显示数据结构信息
                      for key, value in traj_data.items():
                          rospy.loginfo(f"   {key}: shape={value.shape}, dtype={value.dtype}")




                  except Exception as e:
                      rospy.logerr(f"保存轨迹数据失败: {e}")
                      import traceback
                      rospy.logerr(traceback.format_exc())
              else:
                  rospy.logwarn("⚠️  没有记录到轨迹数据 - 请按 'R' 键开始录制")




  def get_recorded_trajectory(self):
      """获取记录的轨迹数据"""
      if self.trajectory_queue and not self.trajectory_queue.empty():
          return self.trajectory_queue.peek_all()
      return None


=======

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
       print(f" target pose from frame = {self.target_pose.from_frame}, to frame = {self.target_pose.to_frame}")
      
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
       
       # 键盘状态
       self.gripper_state = False
       self.recording_state = False
       
       # pynput键盘状态跟踪
       self.keys_pressed = set()  # 跟踪当前按下的键
       self.last_g_state = False
       self.last_r_state = False
   
   def update_keyboard_state(self):
       """更新键盘状态 - 在主循环中调用"""
       if not pynput_available:
           return
           
       # 检测当前按键状态（基于按键集合）
       current_g = 'g' in self.keys_pressed
       current_r = 'r' in self.keys_pressed
       
       # 模式选择：True=Toggle模式(按下切换), False=Hold模式(按住激活)
       USE_TOGGLE_MODE = True
       
       if USE_TOGGLE_MODE:
           # Toggle模式：按下瞬间切换状态
           if current_g and not self.last_g_state:
               self.gripper_state = not self.gripper_state
               rospy.loginfo(f"🤖 夹爪状态切换: {'打开' if self.gripper_state else '关闭'}")
               
           if current_r and not self.last_r_state:
               self.recording_state = not self.recording_state
               if self.recording_state:
                   rospy.loginfo(f"🔴 开始录制机械臂状态...")
               else:
                   rospy.loginfo(f"⏹️  停止录制机械臂状态")
       else:
           # Hold模式：按住时激活，松开时停止
           # 夹爪状态跟随按键
           if current_g != self.gripper_state:
               self.gripper_state = current_g
               rospy.loginfo(f"🤖 夹爪: {'按住-打开' if current_g else '松开-关闭'}")
           
           # 录制状态跟随按键  
           if current_r != self.recording_state:
               self.recording_state = current_r
               if current_r:
                   rospy.loginfo(f"🔴 按住R键-开始录制...")
               else:
                   rospy.loginfo(f"⏹️  松开R键-停止录制")
       
       # 保存当前状态用于下次比较
       self.last_g_state = current_g
       self.last_r_state = current_r
       
       return {
           'gripper_state': self.gripper_state,
           'recording_state': self.recording_state
       }
   
   def on_key_press(self, key):
       """pynput按键按下回调"""
       try:
           # 普通字符键
           self.keys_pressed.add(key.char.lower())
       except AttributeError:
           # 特殊键（如Ctrl, Alt等）
           pass
   
   def on_key_release(self, key):
       """pynput按键释放回调"""
       try:
           # 普通字符键
           self.keys_pressed.discard(key.char.lower())
       except AttributeError:
           # 特殊键
           pass
      
   def run_teleoperation(self, save_traj:bool = False):
       """执行SpaceMouse遥操作"""
       with SharedMemoryManager() as shm_manager:
           # 创建轨迹队列（需要SharedMemoryManager）
           self.trajectory_queue = SharedMemoryQueue.create_from_examples(
               shm_manager,{
               'motion_6d': np.zeros(6),        # [dx,dy,dz,drx,dry,drz]
               'current_pose_7d': np.zeros(7),  # [x,y,z,qw,qx,qy,qz]
               'keyboard_states': False,    # gripper状态
               'timestamp': 0.0,
               'sequence_id': 0,
           }, buffer_size=10000)
          
           # 创建SpaceMouse控制器
           spacemouse = FrankaSpacemouse(
               shm_manager,
               frequency=self.frequency,
               deadzone=0.05,               # 2%死区
               position_sensitivity=0.15,   # 降低平移灵敏度
               rotation_sensitivity=0.15,  # 降低旋转灵敏度
               debug=False  # 关闭调试输出，提升性能
           )
          
           rospy.loginfo(f"开始SpaceMouse遥操作 - 时长: {self.T}s, 频率: {self.frequency}Hz")
           rospy.loginfo("🎮 控制说明:")
           rospy.loginfo("  - SpaceMouse: 控制机械臂移动")
           rospy.loginfo("  - 键盘 'R' 键: 开始/停止录制轨迹")
           rospy.loginfo("  - 键盘 'G' 键: 切换夹爪开/关")
           rospy.loginfo("  - Ctrl+C: 停止操作")
           rospy.loginfo("--------------------")
           
           # 启动键盘监听器（pynput）
           keyboard_listener = None
           if pynput_available:
               keyboard_listener = pynput_keyboard.Listener(
                   on_press=self.on_key_press,
                   on_release=self.on_key_release
               )
               keyboard_listener.start()
               rospy.loginfo("✅ 键盘监听已启动 (pynput)")
           else:
               rospy.logwarn("⚠️  键盘功能不可用 - 请安装: pip install pynput")
          
           with spacemouse:
               try:
                   total_iterations = int(self.T * self.frequency)
                  
                   for i in range(total_iterations):
                       self.index += 1
                      
                       # 读取SpaceMouse输入
                       motion = spacemouse.get_motion_state()
                       # 更新键盘状态
                       keyboard_states = self.update_keyboard_state()
                       #! 为了控制spacemouse按照理想方式移动，需要对motion进行一定的修改
                       motion[0] = -motion[0]  # 反转X轴方向
                       motion[4] = -motion[4]
                       motion[3], motion[4] = motion[4], motion[3]
                       # 计算控制增量
                       translation_delta = motion[:3] * self.dt
                       rotation_angles = motion[3:] * self.dt
                      
                       # 应用平移
                       self.target_pose.translation += translation_delta
                      
                       # 应用旋转（仅在有显著旋转时）
                       if np.linalg.norm(rotation_angles) > 1e-6:
                           rotation_scipy = R.from_euler('xyz', rotation_angles)
                           rotation_matrix_delta = rotation_scipy.as_matrix()
                           self.target_pose.rotation = self.target_pose.rotation @ rotation_matrix_delta
                      
                       # 记录轨迹数据
                       if self.init_time is not None and self.recording_state:  # 确保时间已初始化
                           self.trajectory_queue.put({
                               'motion_6d': motion2array(translation_delta, rotation_angles),
                               'current_pose_7d': pose2array(self.target_pose),
                               'keyboard_states': self.gripper_state,
                               'timestamp': rospy.Time.now().to_time() - self.init_time,
                               'sequence_id': self.index,
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
                   
                   # 停止键盘监听器
                   if keyboard_listener:
                       keyboard_listener.stop()
                       rospy.loginfo("✅ 键盘监听已停止")
              
               rospy.loginfo(f"遥操作结束 - 共执行 {self.index} 个控制步")
              
               # 保存录制的轨迹数据
               if self.trajectory_queue and self.trajectory_queue.qsize() > 0:
                   rospy.loginfo(f"📊 队列中有 {self.trajectory_queue.qsize()} 个录制数据点")


                   # 在SharedMemoryManager上下文内保存数据
                   try:
                       # 使用peek_all()进行非消费性读取，保留数据供后续使用
                       traj_data = self.trajectory_queue.peek_all()
                       filename = "franka_traj_my.pkl"
                       with open(filename, 'wb') as f:
                           pkl.dump(traj_data, f)


                       # 显示保存信息
                       num_points = len(next(iter(traj_data.values())))
                       rospy.loginfo(f"✅ 录制的轨迹数据已保存到: {filename}")
                       rospy.loginfo(f"   📝 记录了 {num_points} 个有效数据点")
                       rospy.loginfo(f"   🤖 包含夹爪状态和机械臂轨迹信息")


                       # 显示数据结构信息
                       for key, value in traj_data.items():
                           rospy.loginfo(f"   {key}: shape={value.shape}, dtype={value.dtype}")


                   except Exception as e:
                       rospy.logerr(f"保存轨迹数据失败: {e}")
                       import traceback
                       rospy.logerr(traceback.format_exc())
               else:
                   rospy.logwarn("⚠️  没有记录到轨迹数据 - 请按 'R' 键开始录制")


   def get_recorded_trajectory(self):
       """获取记录的轨迹数据"""
       if self.trajectory_queue and not self.trajectory_queue.empty():
           return self.trajectory_queue.peek_all()
       return None
>>>>>>> 6bb17a265de97c1b510c6b8789ff36a36ea93bd6


# 使用示例
if __name__ == "__main__":
<<<<<<< HEAD
   # 创建遥操作控制器
  teleop = Sm_franka_teleop(T=90.0, dt=0.01)  # 30秒，50Hz
  try:
      teleop.run_teleoperation(save_traj=True)
  except Exception as e:
      rospy.logerr(f"主程序错误: {e}")








=======
  
   # 创建遥操作控制器
   teleop = Sm_franka_teleop(T=30.0, dt=0.02)  # 30秒，50Hz
  
   try:
       teleop.run_teleoperation(save_traj=True)
   except Exception as e:
       rospy.logerr(f"主程序错误: {e}")
>>>>>>> 6bb17a265de97c1b510c6b8789ff36a36ea93bd6



