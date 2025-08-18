import argparse
import time
from frankapy import FrankaArm
import pickle as pkl
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', '-t', type=float, default=10) # 一次数据采集的动作时间
    parser.add_argument('--open_gripper', '-o', action='store_true') # 采集过程中，夹爪的开闭
    parser.add_argument('--file', '-f', default='franka_traj.pkl') # 数据保存路径
    args = parser.parse_args()
    
    T = args.time
    dt = 0.01 # 100 HZ
    ts = np.arange(0, T, dt) # 步骤数量
    
    print("开始进行机械臂控制")
    fa = FrankaArm()
    if args.open_gripper:
        fa.open_gripper()
    print("进行零重力控制 {} s".format(args.time))
    
    # 末端位姿列表
    end_effector_position = []
    fa.run_guide_mode(args.time, block=False) # 进行为期T秒的自由控制
    
    for i in range(len(ts)):
        end_effector_position.append(fa.get_pose())
        time.sleep(dt) # ts * dt = T 这个循环进行了T 秒
    
    pkl.dump(end_effector_position, open(args.file, 'wb'))
    print("数据保存完成")
    