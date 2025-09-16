"""
用于对机械臂遥操作并保存数据的代码进行Debug
"""
import sys
import os
import time
import pickle as pkl
import numpy as np
from typing import Dict, List, Optional, Any


# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'input_device'))
sys.path.insert(0, os.path.join(current_dir, 'shared_memory'))


# 导入依赖
import rospy
from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from franka_interface_msgs.msg import SensorDataGroup
from frankapy.proto import PosePositionSensorMessage
from scipy.spatial.transform import Rotation as R
from multiprocessing.managers import SharedMemoryManager
import cv2


# 导入自定义模块
try:
   from pynput import keyboard as pynput_keyboard
   pynput_available = True
except ImportError:
   print("Warning: pynput not installed. Please install: pip install pynput")
   pynput_keyboard = None
   pynput_available = False


from input_device.spacemouse import FrankaSpacemouse
from shared_memory.shared_memory_queue import SharedMemoryQueue
from shared_memory.shared_ndarray import SharedNDArray
from real_camera_utils_new import Camera
from data_collection_utils import pose2array, motion2array, publish_pose


class CollectDataWithTeleop2:
   """遥操作控制器 + img_rgb + depth + 机械臂目标状态采集"""
  
   def __init__(self, frequency: float = 30.0, duration:float = 60.0, task_name:str = 'debug1', trail:int = 0, gripper_thres:float = 0.05, instruction:str = "place the block on the plate", save_interval: int = 1):
       """
           Args:
               frequency: 目标的采集频率
               duration: 采集动作时长
               task_name: 任务名称
               gripper_thres: 夹爪阈值
               instruction: 任务描述
               save_interval: 保存间隔，每N步保存一次数据（默认每步都保存）
       """
       self.task_name = task_name
       self.trail = trail
       self.frequency = frequency
       self.dt = 1.0 / frequency
       self.duration = duration
       self.total_steps = int(duration * frequency)
       self.gripper_thres = gripper_thres
       self.instruction = instruction
       self.save_interval = save_interval  # 保存间隔
      
       # 初始化机械臂
       rospy.loginfo("🤖 初始化机械臂...")
       self.fa = FrankaArm()
       self.fa.reset_joints()
       self.current_pose = self.fa.get_pose()
       # 暂存的目标动作，用于后续通过ros向机械臂传输动作
       self.target_pose = self.current_pose.copy()
      
       # 初始化相机 采用默认配置 VGA: 672 * 384 100HZ
       rospy.loginfo("📷 初始化相机 (VGA)...")
       self.camera = Camera(camera_type = "3rd")
      
       # ROS相关内容
       self.rate = rospy.Rate(frequency)
       self.publisher = rospy.Publisher(
           FC.DEFAULT_SENSOR_PUBLISHER_TOPIC,
           SensorDataGroup,
           queue_size = 20
       )
      
       #* 记录状态变量
       self.init_time = None
       self.step_counter = 0
       self.control_step_counter = 0  # 控制步数计数器（用于计算保存间隔）
       self.recording = False
       self.gripper_control_in_progress = False  # 夹爪控制进行中标志
       self.should_exit = False  # 退出标志
      
       #* 键盘状态
       self.keys_pressed = set()
       self.gripper_state = False # False夹爪打开，True夹爪闭合
       self.last_g_state = False
       self.last_r_state = False
      
       # 初始化夹爪状态
       try:
           rospy.loginfo("🔧 初始化夹爪状态...")
           self.fa.open_gripper()  # 确保夹爪处于打开状态
           rospy.sleep(1.0)  # 等待夹爪动作完成
           rospy.loginfo("✅ 夹爪初始化完成（打开状态）")
       except Exception as e:
           rospy.logerr(f"夹爪初始化失败: {e}")
      
       #* 数据存储功能
       self.data_arrays: Dict[str, SharedNDArray] = {} # 存储机械臂状态相关内容
      
   def setup_shared_arrays(self, shm_manager: SharedMemoryManager):
       """设置共享内存数组"""
       rospy.loginfo(f"{'=' * 20} 正在设置共享内存数组 {'=' * 20}")
      
       # 图像大小
       img_shape = (self.total_steps,  376, 672, 3)
      
       # 保存图像BGR
       self.data_arrays['bgr_images'] = SharedNDArray.create_from_shape(
           shm_manager, img_shape, np.uint8
       ) # 0-255
      
       # 记录机械臂位姿 xyz + quat
       self.data_arrays['poses'] = SharedNDArray.create_from_shape(
           shm_manager, (self.total_steps, 7), np.float32
       ) # 机械臂位姿 [x,y,z,qw,qx,qy,qz] 0-1
      
       self.data_arrays['gripper_states'] = SharedNDArray.create_from_shape(
           shm_manager, (self.total_steps,), np.bool_
       ) # 夹爪状态
      
       self.data_arrays['depth'] = SharedNDArray.create_from_shape(
           shm_manager, (self.total_steps, 376, 672), np.float32
       ) #深度图
      
       self.data_arrays['pcd'] = SharedNDArray.create_from_shape(
           shm_manager, (self.total_steps, 376, 672, 3), np.float32
       ) # 点云数据
       self.data_arrays['joints'] = SharedNDArray.create_from_shape(
              shm_manager, (self.total_steps, 7), np.float32
         )
      
       rospy.loginfo(f"✅ 共享数组创建完成，预分配 {self.total_steps} 个数据点")
      
  
   def update_keyboard_state(self) -> Dict[str, Any]:
       """更新键盘状态"""
       if not pynput_available:
           return {'gripper_state': self.gripper_state, 'recording': self.recording}
          
       current_g = 'g' in self.keys_pressed
       current_r = 'r' in self.keys_pressed
      
       # 添加调试信息
       if len(self.keys_pressed) > 0 and (current_g or current_r):
           rospy.loginfo(f"当前按下的键: {self.keys_pressed}")
      
       # Toggle模式
       if current_g and not self.last_g_state and not self.gripper_control_in_progress:
           self.gripper_control_in_progress = True  # 设置夹爪控制进行中标志
           self.gripper_state = not self.gripper_state
           rospy.loginfo(f"🤖 夹爪: {'闭合' if self.gripper_state else '打开'}")
          
           # 执行夹爪动作（阻塞控制）
           try:
               if self.gripper_state:  # True = 夹爪关闭
                   rospy.loginfo("🔒 执行夹爪关闭动作...")
                   self.fa.close_gripper()
               else:  # False = 夹爪打开
                   rospy.loginfo("🔓 执行夹爪打开动作...")
                   self.fa.open_gripper()
              
               # 等待夹爪动作完成（阻塞控制）
               rospy.sleep(1.0)  # 给夹爪足够时间完成动作
               rospy.loginfo("✅ 夹爪动作完成")
              
           except Exception as e:
               rospy.logerr(f"夹爪控制出错: {e}")
               # 如果夹爪控制失败，恢复之前的状态
               self.gripper_state = not self.gripper_state
               rospy.logwarn(f"夹爪状态已恢复为: {'闭合' if self.gripper_state else '打开'}")
           finally:
               self.gripper_control_in_progress = False  # 清除夹爪控制进行中标志
          
       if current_r and not self.last_r_state:
           if self.recording:
               # 停止录制并设置退出标志
               self.recording = False
               self.should_exit = True
               rospy.loginfo("⏹️ 停止录制，准备退出程序")
           else:
               # 开始录制
               self.recording = True
               rospy.loginfo("🔴 开始录制")
      
       self.last_g_state = current_g
       self.last_r_state = current_r
      
       return {'gripper_state': self.gripper_state, 'recording': self.recording}
  
   def on_key_press(self, key):
       """按键按下回调"""
       try:
           if hasattr(key, 'char') and key.char is not None:
               self.keys_pressed.add(key.char.lower())
           elif hasattr(key, 'name'):
               self.keys_pressed.add(key.name.lower())
       except Exception as e:
           rospy.logwarn(f"键盘按键处理错误: {e}")
      
   def on_key_release(self, key):
       """按键释放回调"""
       try:
           if hasattr(key, 'char') and key.char is not None:
               self.keys_pressed.discard(key.char.lower())
           elif hasattr(key, 'name'):
               self.keys_pressed.discard(key.name.lower())
       except Exception as e:
           rospy.logwarn(f"键盘按键处理错误: {e}")
      
   def control_step(self) -> bool:
       """控制步骤，首先拍照记录当前Observation，然后记录机械臂状态，最后通过ros发布机械臂动作"""
      
       # 如果未开启录制，则直接返回gotoe() - step_start
       # 更新键盘状态
       keyboard_states = self.update_keyboard_state()
      
       # 如果未开启录制，则直接返回
       if not self.recording:
           return True
      
       #* 首先拍照获取当前状态
       step_start = time.time()
       # -- result_dict['3rd'] 包含了bgr图像, depth, point cloud
       result_dict = self.camera.capture()
       capture_time = time.time() - step_start
      
       #* 记录相关数据到self.data_arrays
       if self.init_time is None:
           self.init_time = time.time()
          
       #* 在保存范围内，根据保存间隔决定是否保存数据
       time_before_save = time.time()
       if self.step_counter < self.total_steps and self.control_step_counter % self.save_interval == 0:
           self.data_arrays['bgr_images'].get()[self.step_counter] = result_dict['3rd']['rgb']
           self.data_arrays['depth'].get()[self.step_counter] = result_dict['3rd']['depth']
           self.data_arrays['pcd'].get()[self.step_counter] = result_dict['3rd']['pcd']
           actual_pose = self.fa.get_pose()
           actual_joints = self.fa.get_joints()
        #    print(actual_joints) #TODO
           self.data_arrays['poses'].get()[self.step_counter] = pose2array(actual_pose)
           self.data_arrays['joints'].get()[self.step_counter] = np.array(actual_joints)
           self.data_arrays['gripper_states'].get()[self.step_counter] = self.gripper_state
           time_after_save = time.time()
          
           #* 性能监控
           if self.step_counter % 30 == 0:
               rospy.loginfo(f"Step {self.step_counter}, Capture: {capture_time*1000:.1f}ms, Save: {(time_after_save - time_before_save)*1000:.1f}ms")
       else:
           time_after_save = time.time()
          
                       #* 性能监控（不保存时）
           if self.control_step_counter % 30 == 0:
               rospy.loginfo(f"Control Step {self.control_step_counter}, Capture: {capture_time*1000:.1f}ms, Skip Save")
      
       return True
  
   def run_data_collection(self, save_dir: str = "./teleop_data", trail:int = 0):
       """运行数据采集"""
       rospy.loginfo(f"🚀 开始高频数据采集 - {self.frequency}Hz, {self.duration}s")


       with SharedMemoryManager() as shm_manager:
           # 设置共享内存
           self.setup_shared_arrays(shm_manager)
          
           # 创建Spacemouse控制器
           spacemouse = FrankaSpacemouse(
               shm_manager,
               frequency=self.frequency,
               deadzone=0.05,
               position_sensitivity=0.15, #TODO 在这里改变灵敏度
               rotation_sensitivity=0.3,
               debug=False
           )
          
           # 启动键盘监听
           keyboard_listener = None
           if pynput_available:
               keyboard_listener = pynput_keyboard.Listener(
                   on_press=self.on_key_press,
                   on_release=self.on_key_release
               )
               keyboard_listener.start()
               rospy.loginfo("✅ 键盘监听已启动")
          
           rospy.loginfo("🎮 控制说明:")
           rospy.loginfo("  - SpaceMouse: 控制机械臂移动")
           rospy.loginfo("  - 'R' 键: 开始/停止录制")
           rospy.loginfo("  - 'G' 键: 切换夹爪")
           rospy.loginfo("  - Ctrl+C: 停止采集")
           rospy.loginfo("="*50)
          
           with spacemouse:
               try:
                   # 启动机械臂动态控制
                   self.fa.goto_pose(
                       self.target_pose,
                       duration=self.duration,
                       dynamic=True,
                       buffer_time=10,
                       cartesian_impedances=[600.0, 600.0, 600.0, 50.0, 50.0, 50.0] # TODO 可以在这里降低阻抗以提高响应速度
                   )
                  
                   start_time = time.time()
                   rospy.loginfo("🔄 开始控制循环，正确观察-动作同步")
                   for i in range(self.total_steps):
                       loop_start = time.time()
                      
                       #* === 步骤1: 读取输入 (~1ms) ===
                       # 更新键盘状态（无论是否录制都要检查）
                       self.update_keyboard_state()
                      
                       motion = spacemouse.get_motion_state()
                      
                       # 调整运动方向
                       motion[0] = -motion[0]
                       motion[4] = -motion[4]
                       motion[3], motion[4] = motion[4], motion[3]
                      
                       #* === 步骤2: 计算机械臂增量 ===
                       translation_delta = motion[:3] * self.dt
                       rotation_angles = motion[3:] * self.dt
                      
                       #* === 步骤3: 将位姿增量添加到目标位姿上，这时候机械臂还没有进行移动 ===
                       self.target_pose.translation += translation_delta
                      
                       if np.linalg.norm(rotation_angles) > 1e-6:
                           rotation_scipy = R.from_euler('xyz', rotation_angles)
                           rotation_matrix_delta = rotation_scipy.as_matrix()
                           self.target_pose.rotation = self.target_pose.rotation @ rotation_matrix_delta
                      
                       #* === 步骤4: 记录相机照片以及基于目标位姿的机械臂状态 ===
                       # 只有在录制状态下且夹爪控制不在进行中时才进行数据采集
                       if self.recording and not self.gripper_control_in_progress:
                           self.control_step()
                      
                       #* === 步骤5: 发布控制指令 ===
                       # 只有在录制状态下且夹爪控制不在进行中时才发布控制指令
                       if self.recording and not self.gripper_control_in_progress and i > 0:
                           timestamp = time.time() - start_time
                           publish_pose(
                               self.target_pose,
                               i,
                               timestamp,
                               pub=self.publisher,
                               rate=self.rate
                           )
                      
                       # 增加控制步数计数器（无论是否保存数据）
                       if self.recording and not self.gripper_control_in_progress:
                           self.control_step_counter += 1
                          
                           # 只有在保存间隔时才增加数据步数计数器
                           if self.control_step_counter % self.save_interval == 0:
                               self.step_counter += 1
                      
                       #* === 步骤6: 检查退出条件 ===
                       if self.should_exit:
                           rospy.loginfo(f"⏹️ 录制已停止，当前步数: {self.step_counter}")
                           rospy.loginfo("🛑 发送终止信号，停止机械臂控制...")
                           # 停止机械臂控制
                           try:
                               self.fa.stop_skill()
                               rospy.loginfo("✅ 机械臂控制已停止")
                           except Exception as e:
                               rospy.logerr(f"停止机械臂控制时出错: {e}")
                           break
                      
                       #* === 步骤7: 频率控制 ===
                       elapsed = time.time() - loop_start
                       sleep_time = max(0, self.dt - elapsed)
                      
                       #* 性能监控
                       if i % 60 == 0:
                           if self.should_exit:
                               status = "🛑 准备退出"
                           elif self.gripper_control_in_progress:
                               status = "🔧 夹爪控制中"
                           elif self.recording:
                               status = "🔴 录制中"
                           else:
                               status = "⏸️ 暂停中"
                           rospy.loginfo(f"{status} - 第 {i} 步: {elapsed*1000:.1f}ms (target: {self.dt*1000:.1f}ms), 控制步: {self.control_step_counter}, 已记录: {self.step_counter} 步")
                      
                       if sleep_time > 0:
                           time.sleep(sleep_time)
                       elif elapsed > self.dt * 1.2:
                           rospy.logwarn(f"拍照 + 控制循环 超时: 第 {i} 步: {elapsed*1000:.1f}ms")
                      
                       if time.time() - start_time > self.duration:
                           rospy.loginfo("🏁 采集完成，保存数据...")
                           break
                      
               except KeyboardInterrupt:
                   rospy.loginfo("🚨 用户中断，停止采集")
               except Exception as e:
                   rospy.logerr(f"采集错误: {e}")
                   import traceback
                   traceback.print_exc()
               finally:
                   # 停止机械臂
                   try:
                       self.fa.stop_skill()
                   except:
                       pass
                  
                   # 停止键盘监听
                   if keyboard_listener:
                       keyboard_listener.stop()
                  
                   rospy.loginfo("Data collection ended")
                  
                   # 保存数据
                   if self.step_counter > 0:
                       rospy.loginfo(f"💾 保存 {self.step_counter} 步数据...")
                       self.save_collected_data(save_dir)
                   else:
                       rospy.logwarn("没有数据需要保存")
                  
   def save_collected_data(self, save_dir: str):
       """保存采集的数据"""
       rospy.loginfo("💾 保存数据中...")
      
       # 创建保存目录
       os.makedirs(os.path.join(save_dir, self.task_name,f"trail_{self.trail}"), exist_ok=False)
      
       # 获取实际录制的数据长度
       actual_length = self.step_counter
      
       if actual_length == 0:
           rospy.logwarn("未采集到数据")
           return
      
       # 从共享内存提取数据
       data_dict = {
           'bgr_images': self.data_arrays['bgr_images'].get()[:actual_length].copy(),
           'poses': self.data_arrays['poses'].get()[:actual_length].copy(),
           'gripper_states': self.data_arrays['gripper_states'].get()[:actual_length].copy(),
           'depth': self.data_arrays['depth'].get()[:actual_length].copy(),
           'pcd': self.data_arrays['pcd'].get()[:actual_length].copy(),
           'joints': self.data_arrays['joints'].get()[:actual_length].copy()
       }
      
       # 保存为pkl文件
       dir_names = ['bgr_images', '3rd_bgr','depth', 'pcd', 'poses', 'gripper_states', 'joints']
       # 保存指令
       with open(os.path.join(save_dir, self.task_name,f"trail_{self.trail}", "instruction.txt"), 'w') as f:
           f.write(self.instruction)
      
       for dir_name in dir_names:
           dir_path = os.path.join(save_dir, self.task_name,f"trail_{self.trail}", dir_name)
           os.makedirs(dir_path, exist_ok=True)
           if dir_name == 'bgr_images':
               print("正在保存图像文件")
               for i in range(actual_length):
                   img = data_dict[dir_name][i]
                   img_path = os.path.join(dir_path, f"{i:06d}.png")
                   cv2.imwrite(img_path, img)
           elif dir_name == '3rd_bgr':
               print("正在保存bgr数组")
               for i in range(actual_length):
                   img = data_dict['bgr_images'][i]
                   with open(os.path.join(dir_path, f"{i:06d}.pkl"), 'wb') as f:
                       pkl.dump(img, f, protocol=pkl.HIGHEST_PROTOCOL)
           else:
               print(f"正在保存{dir_name}数组")
               for i in range(actual_length):
                   data = data_dict[dir_name][i]
                   with open(os.path.join(dir_path, f"{i:06d}.pkl"), 'wb') as f:
                       pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)
              
       rospy.loginfo(f"数据已保存到: {save_dir}")
      
def main():
   """主函数"""
   frequency = 75.0  # 控制频率：80Hz
   duration = 160.0
   task_name = 'debug'
   gripper_thres = 0.05
   instruction = "close the upper drawer"
   task_idx = 1
   data_result_dir = "/media/casia/data4/wxn/data/cyx"
   save_interval = 4  # 每4步保存一次数据（即20Hz保存频率）


   rospy.loginfo("Starting high-frequency teleoperation data collection system")
   rospy.loginfo(f"Configuration: {frequency}Hz control, {frequency/save_interval}Hz save, {duration}s")
  
   try:
       collector = CollectDataWithTeleop2(
           task_name=task_name,
           gripper_thres=gripper_thres,
           instruction=instruction,
           trail=task_idx,
           frequency=frequency,
           duration=duration,
           save_interval=save_interval
       )
       collector.run_data_collection(save_dir=data_result_dir)
   except KeyboardInterrupt:
       rospy.loginfo("🚨 用户中断，停止采集")
   except Exception as e:
       rospy.logerr(f"采集错误: {e}")
       import traceback
       traceback.print_exc()
      
if __name__ == "__main__":
   main()



