from real_camera_utils_new import Camera
import time
import numpy as np
import open3d as o3d
import scipy.spatial.transform.rotation as R
import os
import cv2

def get_cam_extrinsic(type):
    if type == "3rd":
        # trans = np.array([1.1340949379013272, 0.561350863040624, 0.5357989602947655])
        # quat = np.array([-0.3851963087555203, -0.7686884118133567, 0.4146037462420932, 0.2980698959422155])
        trans=np.array([1.028818510131928, -0.04212360892289513,  0.6338377191806316])
        quat=np.array([ -0.6333204911007358, -0.6400364927579377, 0.3240327100190967,0.29027787777872616]) # x y z w
    elif type == "wrist":
        trans = np.array([0.6871684912377796 , -0.7279882263970943,  0.8123566411202088])
        quat = np.array([-0.869967706085017,  -0.2561670369853595,  0.13940123346877276,  0.39762034107764127])
    else:
        raise ValueError("Invalid type")
    
    transform = np.eye(4)
    rot = R.from_quat(quat)
    transform[:3, :3] = rot.as_matrix()
    transform[:3, 3] = trans.T
    
    return transform

def convert_pcd_to_base(
            type="3rd",
            pcd=[]
        ):
        transform = get_cam_extrinsic(type)
        
        h, w = pcd.shape[:2]
        pcd = pcd.reshape(-1, 3)
        
        pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
        # pcd = (np.linalg.inv(transform) @ pcd.T).T[:, :3]
        pcd = (transform @ pcd.T).T[:, :3]
        
        pcd = pcd.reshape(h, w, 3)
        return pcd 

def save_rgb_image(rgb_array, save_path):
    """
    保存 observation["3rd"]["rgb"] 到指定路径
    :param rgb_array: numpy array, HxWx3, RGB格式，值范围[0,255]或[0,1]
    :param save_path: str, 保存路径
    """
    # 如果是float类型且范围在[0,1]，先转为[0,255] uint8
    if rgb_array.dtype != np.uint8:
        rgb_array = (rgb_array * 255).clip(0, 255).astype(np.uint8)
    # OpenCV保存为BGR格式
    bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, bgr)  

if __name__ == "__main__":
    l = []
    camera = Camera(camera_type='3rd')
    time1 = time.time()
    for i in range(100):
        result_dict = camera.capture()
        result_dict['3rd']['rgb'] = result_dict['3rd']['rgb'][:,:,::-1].copy()
        if i % 25 == 0:
            l.append(result_dict)
    time2 = time.time()
    print(f"一张照片耗时:{(time2-time1)/100}, 以共存了 {len(l)} 张照片")
    for i in range(len(l)):
        pcd_flat = l[i]['3rd']['pcd'].reshape(-1, 3)
        rgb_flat = l[i]['3rd']['rgb'].reshape(-1, 3) / 255.0

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_flat)
        pcd_o3d.colors = o3d.utility.Vector3dVector(rgb_flat)

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1,  # 坐标系的大小，可以根据你的点云尺度调整
            origin=[0, 0, 0]  # 坐标系的原点位置
        )  # 用来判断标定是否正确
        o3d.visualization.draw_geometries([pcd_o3d, coordinate_frame])
        save_rgb_image(l[i]['3rd']['rgb'], f"/media/casia/data4/wxn/debug_{i}.png")
