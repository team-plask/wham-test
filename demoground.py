import os
import argparse
from glob import glob
from collections import defaultdict
import cv2
import torch
import joblib
import numpy as np
from loguru import logger
from progress.bar import Bar
from configs.config import get_cfg_defaults, get_cfg
import os
import cv2
import joblib
import numpy as np
from lib.models import build_body_model
from lib.utils import transforms
from lib.data.datasets.dataset_custom import convert_dpvo_to_cam_angvel
from configs import constants as _C
import smplx
from smplx.lbs import vertices2joints
from lib.data.utils.normalizer import Normalizer

import os.path as osp

#intermediate parameters blank declaration
labels = None
tracking_results = None
slam_results = None
fps = 30
scaleFactor = 1.1
J_regressor_eval = None#torch.tensor(joblib.load('J_regressor_extra.npy')).to('cuda')
J_regressor_wham = None#torch.tensor(joblib.load('J_regressor_wham.npy')).to('cuda')
smpl_batch_size = None#cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
smpl = None#build_body_model(cfg.DEVICE, smpl_batch_size)
global_path = None
folder_path = None
n_frames = None
J_regressor_feet= None#torch.tensor(joblib.load('J_regressor_feet.npy')).to('cuda')

smpl = smplx.create(model_path="dataset/body_models/", model_type='smpl')
keypoints_normalizer = None



def run(cfg, _global_path="dummy/test/", _image_subdir="images/"):
    #initialize
    global labels, tracking_results, slam_results, fps, scaleFactor, J_regressor_eval, J_regressor_wham,J_regressor_feet, smpl_batch_size, smpl, global_path, image_subdir, folder_path, n_frames, keypoints_normalizer
    #smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    #smpl = build_body_model(cfg.DEVICE, smpl_batch_size)

    global_path = _global_path
    J_regressor_eval = torch.from_numpy(
            np.load(_C.BMODEL.JOINTS_REGRESSOR_H36M)
        )[_C.KEYPOINTS.H36M_TO_J14, :].unsqueeze(0).float().to('cuda')
    J_regressor_wham = torch.from_numpy(np.load(_C.BMODEL.JOINTS_REGRESSOR_WHAM)).float()
    J_regressor_feet = torch.from_numpy(np.load('dataset/body_models/J_regressor_feet.npy')).float()
    keypoints_normalizer = Normalizer(cfg)
    bar = Bar('Processing', max=len(os.listdir(global_path)))
    for vid in os.listdir(global_path):
        #initialize about vid
        labels = joblib.load(f'{global_path}{vid}/{vid}_data.pkl')
        folder_path = f'{global_path}{vid}'
        tracking_results = joblib.load(f'{global_path}{vid}/tracking_results.pth')
        slam_results = joblib.load(f'{global_path}{vid}/slam_results.pth')
        n_frames = labels['n_frames']

        get_single_sequence()
        bar.next()
    bar.finish()

    pass

def compute_contact_label(feet, thr=1e-2, alpha=5):
    vel = torch.zeros_like(feet[..., 0]).clone()
    label = torch.zeros_like(feet[..., 0]).clone()
        
    vel[1:-1] = (feet[2:] - feet[:-2]).norm(dim=-1) / 2.0
    vel[0] = vel[1].clone()
    vel[-1] = vel[-2].clone()
        
    label = 1 / (1 + torch.exp(alpha * (thr ** -1) * (vel - thr)))
    return label.clone()

def perspective_projection(kp3d, K, R, T):
    """
    3D 점들을 2D 이미지 평면으로 투영합니다.

    :param kp3d: 3D 점들의 좌표, 크기는 [n_joints, 3]입니다.
    :param K: 카메라의 내부 매개변수 행렬, 크기는 [3, 3]입니다.
    :param R: 회전 행렬, 크기는 [3, 3]입니다.
    :param T: 이동 벡터, 크기는 [3]입니다.
    :return: 2D 투영된 점들의 좌표, 크기는 [n_joints, 2]입니다.
    """
    # T를 [3, 1]로 만듭니다.
    T = T.view(3, 1).float()
    R = R.float()
    kp3d = kp3d.float()
    K = K.float()
    
    
    # 3D 점을 균질 좌표로 확장합니다.
    ones = torch.ones(kp3d.shape[0], 1)
    kp3d_homogeneous = torch.cat((kp3d, ones), dim=1)  # [n_joints, 4]
    
    # 카메라 외부 매개변수 (R, T)를 사용하여 3D 공간에서의 위치 변환을 수행합니다.
    RT = torch.cat((R, T), dim=1)  # [3, 4]
    kp3d_transformed = torch.matmul(kp3d_homogeneous, RT.T)  # [n_joints, 4]
    print("transformed", kp3d_transformed.shape)
    
    # 내부 매개변수 K를 사용하여 2D 이미지 평면으로 투영합니다.
    kp2d_homogeneous = torch.matmul(kp3d_transformed, K.T)  # [n_joints, 3]
    print("kp2d_homogeneous", kp2d_homogeneous.shape)
    
    # 균질 좌표에서 일반 좌표로 변환합니다.
    kp2d = kp2d_homogeneous[:, :2] / kp2d_homogeneous[:, 2].unsqueeze(1)  # [n_joints, 2]
    print("kp2d", kp2d.shape)
    
    return kp2d.squeeze()


def get_kp2d():
    #print(self.tracking_results[0]['keypoints'])
    three =tracking_results[0]['keypoints']
    three = torch.tensor(three).clone()
    #print(three.shape)
    kp2d = keypoints_normalizer(three[:,:,:2], torch.tensor(get_res()), torch.tensor(labels['camera']["intrinsics"]).clone(), 224, 224, torch.tensor(tracking_results[0]['bbox']))
    print(kp2d[0].shape)
    return torch.tensor(kp2d[0])

def get_features():
    return tracking_results[0]['features']

def get_gt_kp2d():
    kp3ds = get_gt_kp3d()[..., :3] #except confidence
    cam_intrinsic = torch.tensor(labels['camera']["intrinsics"]).clone()
    R = get_R()
    T = torch.tensor(labels['camera']["extrinsics"][:, :3, 3]).clone()
    gt_kp2d = []
    for frame_index in range(n_frames):
        x2d = perspective_projection(kp3ds[frame_index].squeeze(), cam_intrinsic, R[frame_index], T[frame_index])
        gt_kp2d.append(x2d)

    return torch.stack(gt_kp2d)

def get_weak_kp2d():
    return torch.zeros_like(get_gt_kp2d()).clone()

def get_full_kp2d():
    print("full_kp2d",get_gt_kp2d().shape)
    return get_gt_kp2d().squeeze().clone()

def get_init_kp2d():
    return torch.tensor(labels["kp2d"][0]).clone()

def get_R():
    return torch.tensor(labels['camera']["extrinsics"][:, :3, :3]).clone()

def get_cam_angvel():
    cam_angvel = convert_dpvo_to_cam_angvel(slam_results, fps=fps)
    return cam_angvel.clone()

def get_gt_cam_angvel():
    R = get_R()
    cam_angvel = transforms.matrix_to_rotation_6d(R[:-1] @ R[1:].transpose(-1, -2))
    cam_angvel = (cam_angvel - torch.tensor([[1, 0, 0, 0, 1, 0]])) * fps
    return torch.tensor(cam_angvel).clone()

def get_cam_poses():
    return torch.tensor(labels['camera']["extrinsics"]).clone()

def get_res():
    return (labels['camera']['width'], labels['camera']['height'])

def get_bbox():
    bboxes = []
    bboxes_np = labels['bboxes']['bboxes']  # 이 부분이 NumPy 배열을 반환한다고 가정합니다.
    bboxes_tensor = torch.tensor(bboxes_np, dtype=torch.float)  # NumPy 배열을 PyTorch 텐서로 변환
    center_x = (bboxes_tensor[:, 0] + bboxes_tensor[:, 2]) / 2
    center_y = (bboxes_tensor[:, 1] + bboxes_tensor[:, 3]) / 2
    bbox_w = bboxes_tensor[:, 2] - bboxes_tensor[:, 0]
    bbox_h = bboxes_tensor[:, 3] - bboxes_tensor[:, 1]
    scale = torch.stack((bbox_w, bbox_h)).max(0)[0] / scaleFactor  # scaleFactor는 적절한 스케일 팩터로 정의되어 있어야 함
    
    return torch.stack((center_x, center_y, scale), dim=1).clone()

def get_gt_kp3d():
    gt_output = get_gt_smpl()
    vertices  = []
    for frame_index in range(n_frames):
        vertices.append(gt_output[frame_index].vertices.squeeze(0))
    vertices = torch.stack(vertices, dim=0)
    kp3ds = vertices2joints(J_regressor_wham, vertices)
    kp3ds_with_confidence = torch.cat((kp3ds, torch.ones_like(kp3ds[..., 0]).unsqueeze(-1)), dim=-1)
    print("kp3d",kp3ds_with_confidence.shape)
    return kp3ds_with_confidence.clone()

    return kp3ds.clone()

def get_init_kp3d():
    return get_gt_kp3d()[0].clone()


def get_pose():
    poses_root = torch.tensor(labels['smpl']['poses_root'][:])
    poses_body = torch.tensor(labels['smpl']['poses_body'][:])
    poses = torch.cat((poses_root, poses_body), dim=1)
    matrix_poses = []
    for pose in poses:
        pose = pose.view(24,3)
        matrix_one_pose = []
        for one_joint in pose:
            # 축-각도에서 회전 행렬로 변환
            rot_matrix = transforms.axis_angle_to_matrix(one_joint)
            # 회전 행렬에서 6D 회전으로 변환
            rot_6d = transforms.matrix_to_rotation_6d(rot_matrix)
            matrix_one_pose.append(rot_6d)
        matrix_one_pose_tensor = torch.stack(matrix_one_pose, dim=0)
        matrix_poses.append(matrix_one_pose_tensor)
    matrix_poses_tensor = torch.stack(matrix_poses, dim=0)
    return matrix_poses_tensor.clone()

def get_init_pose():
    return get_pose()[0].clone()

def get_pose_root():
    pose_roots = []
    for root in labels['smpl']['poses_root']:
        root = torch.tensor(root, dtype=torch.float)
        pose_roots.append(transforms.matrix_to_rotation_6d(transforms.axis_angle_to_matrix(root)))
    return torch.stack(pose_roots, dim=0).clone()

def get_vel_root():
    pose_roots_temp = []
    for root in labels['smpl']['poses_root']:
        root = torch.tensor(root, dtype=torch.float)
        pose_roots_temp.append(transforms.axis_angle_to_matrix(root))
    pose_roots = torch.stack(pose_roots_temp, dim=0)
    vel_world = labels['smpl']['trans'][1:] - labels['smpl']['trans'][:-1]
    vel_world = torch.tensor(vel_world)
    vel_root = (pose_roots[:-1].transpose(-1, -2) @ vel_world.unsqueeze(-1)).squeeze(-1)
    return vel_root.clone()

def get_contact():
    gt_output = get_gt_smpl()
    contacts = []
    vertices  = []
    for frame_index in range(n_frames):
        vertices.append(gt_output[frame_index].vertices.squeeze(0))
    vertices = torch.stack(vertices, dim=0)
    feet = vertices2joints(J_regressor_feet, vertices)
    
    return compute_contact_label(feet).clone()

def get_betas():
    return torch.tensor(labels['smpl']['betas']).clone()

def get_gt_smpl():
    global_orient = torch.zeros(1, 3)
    body_pose = torch.zeros(1, 69)
    betas = torch.zeros(1, 10)  
    output = smpl(betas=betas, body_pose=body_pose, global_orient=global_orient)
    #print("smpl availble")
    smpl_outputs = []
    for frame_index in range(n_frames):
        #print(frame_index   )
        global_orient = torch.tensor(labels['smpl']['poses_root'][frame_index]).unsqueeze(0).float()
        body_pose = torch.tensor(labels['smpl']['poses_body'][frame_index]).unsqueeze(0).float()
        betas = torch.tensor(labels['smpl']['betas']).unsqueeze(0).float()

        gt_output = smpl(
            body_pose=body_pose,
            global_orient=global_orient,
            betas=betas)
        
        smpl_outputs.append(gt_output)
    return smpl_outputs
    

    #print("done")

    return smpl_outputs

def get_single_sequence():
    target = {
        'kp2d': get_kp2d(),
        'features': get_features(),
        'cam_angvel': get_cam_angvel(),
        'res': get_res(),
        'init_pose': get_init_pose(),
        'init_kp3d': get_init_kp3d(),
        'init_kp2d': get_init_kp2d(),
        'pose': get_pose(), #pose is 3x3 -> 3x2 matrix format. use transforms.axis_angle_to_matrix() for axis angles, then use 
        'betas': get_betas(),
        'vel_root': get_vel_root(),
        'pose_root': get_pose_root(),
        'gt_kp3d': get_gt_kp3d(),
        'cam_poses': get_cam_poses(),
        'R': get_R(),
        'gt_cam_angvel': get_gt_cam_angvel(),
        'bbox': get_bbox(),
        'gt_kp2d': get_gt_kp2d(),
        'weak_kp2d': get_weak_kp2d(),
        'contact': get_contact(),
        'full_kp2d': get_full_kp2d()
    }

    joblib.dump(target, f'{folder_path}/target.pkl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--global_path', type=str, default='dummy/test/', help='output folder to write results')
    parser.add_argument('--image_subdir', type=str, default='images/', help='output folder to write results')
    parser.add_argument('--calib', type=str, default=None)
    parser.add_argument('-c', '--cfg', type=str, default='./configs/debug.yaml', help='cfg file path')
    parser.add_argument(
        "opts", default=None, nargs=argparse.REMAINDER,
        help="Modify config options using the command-line")
    args = parser.parse_args()
    cfg = get_cfg(args, test=False)

    run(cfg, args.global_path, args.image_subdir)



