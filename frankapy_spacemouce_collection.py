#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Franka机械臂遥操作控制与数据采集
使用SpaceMouse控制Franka机械臂，并同时采集图像和末端位姿数据
"""

import os
import time
import numpy as np
import cv2
import threading
import queue
from datetime import datetime
from pathlib import Path

# FrankaArm相关导入
try:
    from frankapy import FrankaArm
    import rospy
    FRANKAPY_AVAILABLE = True
except ImportError:
    print("警告: 未找到frankapy，请确保已正确安装")
    FRANKAPY_AVAILABLE = False

# SpaceMouse相关导入
try:
    import pyspacemouse
    PYSPACEMOUSE_AVAILABLE = True
except ImportError:
    print("警告: 未找到pyspacemouse，请安装: pip install pyspacemouse")
    PYSPACEMOUSE_AVAILABLE = False

# 变换相关
try:
    from scipy.spatial.transform import Rotation as R
    SCIPY_AVAILABLE = True
except ImportError:
    print("警告: 未找到scipy，请安装: pip install scipy")
    SCIPY_AVAILABLE = False


class SpaceMouseController:
    """SpaceMouse控制器"""
    
    def __init__(self, sensitivity=0.1, deadzone=0.05):
        """
        初始化SpaceMouse控制器
        
        参数:
            sensitivity (float): 控制灵敏度
            deadzone (float): 死区大小
        """
        self.sensitivity = sensitivity
        self.deadzone = deadzone
        self.running = False
        self.latest_state = {
            'tx': 0.0, 'ty': 0.0, 'tz': 0.0,
            'rx': 0.0, 'ry': 0.0, 'rz': 0.0,
            'buttons': []
        }
        
        if not PYSPACEMOUSE_AVAILABLE:
            raise ImportError("pyspacemouse未安装")
        
        # 初始化SpaceMouse
        success = pyspacemouse.open()
        if not success:
            raise RuntimeError("无法连接SpaceMouse设备")
        
        print("SpaceMouse连接成功")
    
    def start(self):
        """启动SpaceMouse读取线程"""
        self.running = True
        self.thread = threading.Thread(target=self._read_loop)
        self.thread.daemon = True
        self.thread.start()
        print("SpaceMouse控制器已启动")
    
    def stop(self):
        """停止SpaceMouse"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        pyspacemouse.close()
        print("SpaceMouse控制器已停止")
    
    def _read_loop(self):
        """SpaceMouse读取循环，100HZ读取频率，将xyzrpybottons返回在self.latest_state"""
        while self.running:
            try:
                state = pyspacemouse.read()
                if state:
                    # 应用死区和灵敏度
                    tx = self._apply_deadzone(state.x * self.sensitivity)
                    ty = self._apply_deadzone(state.y * self.sensitivity)
                    tz = self._apply_deadzone(state.z * self.sensitivity)
                    rx = self._apply_deadzone(state.roll * self.sensitivity)
                    ry = self._apply_deadzone(state.pitch * self.sensitivity)
                    rz = self._apply_deadzone(state.yaw * self.sensitivity)
                    
                    # 获取按钮状态
                    buttons = []
                    if hasattr(state, 'buttons'):
                        for i, pressed in enumerate(state.buttons):
                            if pressed:
                                buttons.append(i)
                    
                    self.latest_state = {
                        'tx': tx, 'ty': ty, 'tz': tz,
                        'rx': rx, 'ry': ry, 'rz': rz,
                        'buttons': buttons
                    }
                
                time.sleep(0.01)  # 100Hz读取频率
            except Exception as e:
                print(f"SpaceMouse读取错误: {e}")
                time.sleep(0.1)
    
    def _apply_deadzone(self, value):
        """应用死区"""
        return 0.0 if abs(value) < self.deadzone else value
    
    def get_control_command(self):
        """获取控制命令"""
        return self.latest_state.copy()


class CameraCapture:
    """相机捕获类"""
    
    def __init__(self, camera_id=0, resolution=(640, 480)):
        """
        初始化相机
        
        参数:
            camera_id (int): 相机ID
            resolution (tuple): 分辨率
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.cap = None
        self.running = False
        
    def start(self):
        """启动相机"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开相机 {self.camera_id}")
        
        # 设置分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        self.running = True
        print(f"相机 {self.camera_id} 已启动，分辨率: {self.resolution}")
    
    def stop(self):
        """停止相机"""
        self.running = False
        if self.cap:
            self.cap.release()
        print("相机已停止")
    
    def capture_frame(self):
        """捕获一帧图像"""
        if not self.running or not self.cap:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None


class FrankaController:
    """Franka机械臂控制器"""
    
    def __init__(self, max_translation_speed=0.1, max_rotation_speed=0.3):
        """
        初始化Franka控制器
        
        参数:
            max_translation_speed (float): 最大平移速度 (m/s)
            max_rotation_speed (float): 最大旋转速度 (rad/s)
        """
        if not FRANKAPY_AVAILABLE:
            raise ImportError("frankapy未安装")
        
        # 初始化ROS节点
        if not rospy.get_node_uri():
            rospy.init_node('franka_spacemouse_control', anonymous=True)
        
        # 初始化Franka机械臂
        self.fa = FrankaArm()
        self.max_translation_speed = max_translation_speed
        self.max_rotation_speed = max_rotation_speed
        
        # 获取并保存初始位姿作为reset位置
        self.reset_pose = self.fa.get_pose()
        self.current_pose = self.reset_pose
        self.gripper_state = 0  # 0: 关闭, 1: 打开
        
        print("Franka机械臂已连接")
        print(f"初始位姿已保存为reset位置: {self.reset_pose}")
    
    def reset_to_initial_pose(self, duration=3.0):
        """
        将机械臂reset到初始位置
        
        参数:
            duration (float): 移动到初始位置的时间
        """
        try:
            print("正在将机械臂reset到初始位置...")
            
            # 移动到初始位置
            self.fa.goto_pose(self.reset_pose, duration=duration)
            
            # 确保夹爪处于开启状态
            if self.gripper_state == 0:
                print("开启夹爪...")
                self.fa.open_gripper()
                self.gripper_state = 1
            
            self.current_pose = self.reset_pose
            print("机械臂已reset到初始位置")
            
        except Exception as e:
            print(f"Reset机械臂时出错: {e}")
    
    def update_pose(self, spacemouse_command, dt=0.02):
        """
        根据SpaceMouse命令更新机械臂位姿
        
        参数:
            spacemouse_command (dict): SpaceMouse命令
            dt (float): 时间步长
        """
        try:
            # 获取当前位姿
            current_pose = self.fa.get_pose()
            
            # 计算位置增量
            dx = spacemouse_command['tx'] * self.max_translation_speed * dt
            dy = spacemouse_command['ty'] * self.max_translation_speed * dt
            dz = spacemouse_command['tz'] * self.max_translation_speed * dt
            
            # 计算姿态增量
            droll = spacemouse_command['rx'] * self.max_rotation_speed * dt
            dpitch = spacemouse_command['ry'] * self.max_rotation_speed * dt
            dyaw = spacemouse_command['rz'] * self.max_rotation_speed * dt
            
            # 更新位置
            current_pose.translation[0] += dx
            current_pose.translation[1] += dy
            current_pose.translation[2] += dz
            
            # 更新姿态（欧拉角增量）
            if SCIPY_AVAILABLE and (abs(droll) > 1e-6 or abs(dpitch) > 1e-6 or abs(dyaw) > 1e-6):
                # 获取当前旋转矩阵
                current_rotation = R.from_matrix(current_pose.rotation)
                
                # 创建增量旋转
                delta_rotation = R.from_euler('xyz', [droll, dpitch, dyaw])
                
                # 应用增量旋转
                new_rotation = current_rotation * delta_rotation
                current_pose.rotation = new_rotation.as_matrix()
            
            # 发送新位姿到机械臂
            self.fa.goto_pose(current_pose, duration=dt*2)
            
            # 处理夹爪控制
            if 0 in spacemouse_command['buttons']:  # 按钮1控制夹爪
                if self.gripper_state == 0:
                    self.fa.open_gripper()
                    self.gripper_state = 1
                    print("夹爪打开")
                else:
                    self.fa.close_gripper()
                    self.gripper_state = 0
                    print("夹爪关闭")
                
                # 等待一段时间避免重复触发
                time.sleep(0.2)
            
            self.current_pose = current_pose
            
        except Exception as e:
            print(f"更新机械臂位姿时出错: {e}")
    
    def get_end_effector_pose(self):
        """
        获取末端执行器位姿
        
        返回:
            pose_array (np.ndarray): [x, y, z, roll, pitch, yaw, gripper_state]
        """
        try:
            pose = self.fa.get_pose()
            
            # 提取位置
            x, y, z = pose.translation
            
            # 提取欧拉角
            if SCIPY_AVAILABLE:
                rotation = R.from_matrix(pose.rotation)
                roll, pitch, yaw = rotation.as_euler('xyz')
            else:
                # 如果没有scipy，返回零角度
                roll, pitch, yaw = 0.0, 0.0, 0.0
            
            return np.array([x, y, z, roll, pitch, yaw, self.gripper_state])
            
        except Exception as e:
            print(f"获取末端位姿时出错: {e}")
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])



class DataCollector:
    """数据采集器 - 支持多次采集和自动reset"""
    
    def __init__(self, base_output_dir="./franka_data", frequency=50):
        """
        初始化数据采集器
        
        参数:
            base_output_dir (str): 基础数据保存目录
            frequency (int): 采集频率 (Hz)
        """
        self.base_output_dir = Path(base_output_dir)
        self.frequency = frequency
        self.dt = 1.0 / frequency
        
        self.running = False
        self.collection_count = 0  # 采集次数计数
        
        # 当前采集的数据存储
        self.current_images = []
        self.current_poses = []
        self.current_frame_count = 0
        
        # 创建基础输出目录
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"数据将保存到: {self.base_output_dir}")
    
    def start_collection(self, camera, franka_controller):
        """开始新的数据采集"""
        if self.running:
            print("当前正在采集中，请先停止当前采集")
            return
        
        self.running = True
        self.collection_count += 1
        
        # 重置当前采集的数据
        self.current_images = []
        self.current_poses = []
        self.current_frame_count = 0
        
        # 启动采集线程
        self.thread = threading.Thread(
            target=self._collection_loop, 
            args=(camera, franka_controller)
        )
        self.thread.daemon = True
        self.thread.start()
        
        print(f"开始第 {self.collection_count} 次数据采集，频率: {self.frequency}Hz")
    
    def stop_collection_and_reset(self, franka_controller):
        """停止数据采集、保存数据并reset机械臂"""
        if not self.running:
            print("当前没有进行数据采集")
            return
        
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2.0)
        
        print(f"第 {self.collection_count} 次数据采集已停止，共采集 {self.current_frame_count} 帧")
        
        # 保存数据
        if self.current_frame_count > 0:
            print("正在保存数据...")
            self.save_current_data()
            
            # 数据保存完成后，reset机械臂
            print("数据保存完成，正在reset机械臂...")
            franka_controller.reset_to_initial_pose(duration=3.0)
            
            print(">>> 准备就绪，可以开始下一次采集 <<<")
        else:
            print("没有采集到数据，跳过保存")
    
    def _collection_loop(self, camera, franka_controller):
        """数据采集循环"""
        last_time = time.time()
        
        while self.running:
            current_time = time.time()
            
            # 控制采集频率
            if current_time - last_time >= self.dt:
                try:
                    # 采集图像
                    frame = camera.capture_frame()
                    if frame is not None:
                        self.current_images.append(frame.copy())
                    
                    # 采集末端位姿
                    pose = franka_controller.get_end_effector_pose()
                    self.current_poses.append(pose)
                    
                    self.current_frame_count += 1
                    
                    if self.current_frame_count % 50 == 0:  # 每秒打印一次
                        print(f"第 {self.collection_count} 次采集: 已采集 {self.current_frame_count} 帧")
                    
                    last_time = current_time
                    
                except Exception as e:
                    print(f"数据采集错误: {e}")
            
            time.sleep(0.001)  # 短暂休眠
    
    def save_current_data(self):
        """保存当前采集的数据"""
        if not self.current_images or not self.current_poses:
            print("没有数据可保存")
            return
        
        # 创建当前采集的输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_output_dir = self.base_output_dir / f"collection_{self.collection_count:03d}_{timestamp}"
        
        rgb_dir = current_output_dir / "RGB_imgs"
        pose_dir = current_output_dir / "EET_pose"
        
        rgb_dir.mkdir(parents=True, exist_ok=True)
        pose_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"正在保存第 {self.collection_count} 次采集的数据...")
        
        # 保存图像
        for i, image in enumerate(self.current_images):
            image_path = rgb_dir / f"frame_{i:06d}.png"
            cv2.imwrite(str(image_path), image)
        
        # 保存位姿数据
        poses_array = np.array(self.current_poses)
        pose_path = pose_dir / "end_effector_poses.npy"
        np.save(str(pose_path), poses_array)
        
        # 保存元数据
        metadata = {
            "collection_number": self.collection_count,
            "timestamp": timestamp,
            "total_frames": len(self.current_images),
            "frequency": self.frequency,
            "duration": len(self.current_images) / self.frequency,
            "pose_format": "[x, y, z, roll, pitch, yaw, gripper_state]",
            "reset_after_collection": True
        }
        
        import json
        metadata_path = current_output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"第 {self.collection_count} 次采集数据保存完成:")
        print(f"  - 保存目录: {current_output_dir}")
        print(f"  - 图像: {len(self.current_images)} 张")
        print(f"  - 位姿: {len(self.current_poses)} 个")
        print(f"  - 持续时间: {len(self.current_images) / self.frequency:.2f} 秒")
    
    def get_collection_status(self):
        """获取采集状态"""
        return {
            "is_collecting": self.running,
            "collection_count": self.collection_count,
            "current_frames": self.current_frame_count if self.running else 0
        }


def main():
    """主函数 - 支持多次采集和自动reset"""
    print("=== Franka机械臂SpaceMouse遥操作控制 (多次采集+自动Reset模式) ===")
    
    try:
        # 初始化各个组件
        print("初始化SpaceMouse...")
        spacemouse = SpaceMouseController(sensitivity=0.1, deadzone=0.05)
        
        print("初始化相机...")
        camera = CameraCapture(camera_id=0, resolution=(640, 480))
        
        print("初始化Franka机械臂...")
        franka = FrankaController(
            max_translation_speed=0.1, 
            max_rotation_speed=0.3
        )
        
        print("初始化数据采集器...")
        base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        collector = DataCollector(
            base_output_dir=f"./franka_data_{base_timestamp}", 
            frequency=50
        )
        
        # 启动各个组件
        spacemouse.start()
        camera.start()
        
        print("\n=== 控制说明 ===")
        print("- 移动SpaceMouse控制机械臂位置和姿态")
        print("- 按下SpaceMouse按钮1切换夹爪开闭")
        print("- 按下SpaceMouse按钮2开始/停止数据采集")
        print("- 每次采集结束后，机械臂会自动reset到初始位置")
        print("- 程序支持多次采集，每次采集会保存到独立的文件夹")
        print("- 按Ctrl+C退出程序")
        print("\n程序已启动，等待操作...")
        
        last_button_state = []
        button2_press_time = 0
        
        # 主控制循环
        while True:
            try:
                # 获取SpaceMouse命令
                command = spacemouse.get_control_command()
                
                # 更新机械臂位姿
                franka.update_pose(command, dt=0.02)
                
                # 检查按钮2是否被按下（开始/停止采集）
                current_time = time.time()
                if 1 in command['buttons'] and 1 not in last_button_state:
                    # 防抖动：确保按钮按下间隔至少0.5秒
                    if current_time - button2_press_time > 0.5:
                        status = collector.get_collection_status()
                        
                        if not status["is_collecting"]:
                            print(f"\n>>> 开始第 {status['collection_count'] + 1} 次数据采集...")
                            collector.start_collection(camera, franka)
                        else:
                            print(f"\n>>> 停止第 {status['collection_count']} 次数据采集...")
                            # 修改：使用新的方法，包含reset功能
                            collector.stop_collection_and_reset(franka)
                        
                        button2_press_time = current_time
                
                # 显示当前状态（每2秒显示一次）
                if int(current_time) % 2 == 0 and int(current_time * 10) % 20 == 0:
                    status = collector.get_collection_status()
                    if status["is_collecting"]:
                        print(f"正在进行第 {status['collection_count']} 次采集: {status['current_frames']} 帧")
                
                last_button_state = command['buttons'].copy()
                
                time.sleep(0.02)  # 50Hz控制频率
                
            except KeyboardInterrupt:
                print("\n程序被用户中断")
                break
            except Exception as e:
                print(f"主循环错误: {e}")
                time.sleep(0.1)
    
    except Exception as e:
        print(f"初始化错误: {e}")
        return 1
    
    finally:
        # 清理资源
        print("正在清理资源...")
        
        # 如果正在采集，先停止并保存，然后reset
        if 'collector' in locals() and 'franka' in locals():
            status = collector.get_collection_status()
            if status["is_collecting"]:
                print("保存最后一次采集的数据并reset机械臂...")
                collector.stop_collection_and_reset(franka)
        
        if 'spacemouse' in locals():
            spacemouse.stop()
        
        if 'camera' in locals():
            camera.stop()
        
        print("程序已退出")
    
    return 0


if __name__ == "__main__":
    exit(main())