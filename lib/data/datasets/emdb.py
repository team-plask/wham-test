import torch
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor
from configs.config import get_cfg_defaults
import os
import cv2
import joblib
import numpy as np
from lib.models import build_body_model
from .._dataset import BaseDataset
from ...utils import transforms
from .dataset_custom import convert_dpvo_to_cam_angvel
import multiprocessing

#cfg = get_cfg_defaults()
def compute_contact_label(feet, thr=1e-2, alpha=5):
    vel = torch.zeros_like(feet[..., 0]).clone()
    label = torch.zeros_like(feet[..., 0]).clone()
    
    vel[1:-1] = (feet[2:] - feet[:-2]).norm(dim=-1) / 2.0
    vel[0] = vel[1].clone()
    vel[-1] = vel[-2].clone()
    
    label = 1 / (1 + torch.exp(alpha * (thr ** -1) * (vel - thr)))
    return label

class EMDBDataset(BaseDataset):
    def __init__(self, cfg):
        # Initialize the dataset here
        #torch.backends.cudnn.benchmark = True
        multiprocessing.set_start_method('spawn')
        super(EMDBDataset, self).__init__(cfg, training=True)
        self.labels = []
        # load_an array of images from 'folder_path'
        self.folder_path = 'folder/to/global_path/vid/'
        self.images = [] #array of images
        self.fps = 30
        self.scaleFactor = 1.1
        smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
        self.smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
        self.global_path = 'dummy/test/' #folder/to/emdb
        self.vid = os.listdir(self.global_path)
        self.video_indices = os.listdir(self.global_path)
        self.cfg = cfg
        self.tracking_results = []
        self.slam_results = []

        #for vid in self.vid:
        #    self.do_kp2d_features_slam(vid)
        pass

    # def do_kp2d_features_slam(self, vid: str):
    #     # Run detection model to get kp2d
    #     # Run detection model to get kp2d

    #     output_pth = f'{self.global_path}{vid}'

    #     if not (os.path.exists(os.path.join(output_pth, 'tracking_results.pth')) and 
    #             os.path.exists(os.path.join(output_pth, 'slam_results.pth'))):
    #         detector = DetectionModel(self.cfg.DEVICE.lower())  # Replace with your actual detection model
    #         extractor = FeatureExtractor(self.cfg.DEVICE.lower(), self.cfg.FLIP_EVAL)

    #         image_folder = f'{self.global_path}{vid}/images/'
    #         images = []
    #         for filename in sorted(os.listdir(image_folder)):
    #             if filename.endswith(".jpg") or filename.endswith(".png"):
    #                 image_path = os.path.join(image_folder, filename)
    #                 image = cv2.imread(image_path)
    #                 images.append(image)

    #         width, height = images[0].shape[:2]
    #         slam = SLAMModel(video=image_folder, is_images=True, output_pth=output_pth, width=width, height=height)
    #         # Run detection model on each image
    #         # EMBD dataset's cropped images' fps is 30
    #         for img in images:
    #             # 2D detection and tracking
    #             detector.track(img, self.fps, length=len(images))
                
    #             # SLAM
    #             if slam is not None: 
    #                 slam.track()

    #         tracking_results = detector.process(fps=self.fps)
    #         tracking_results = extractor.run(video=images, tracking_results=tracking_results)

    #         joblib.dump(tracking_results, os.path.join(output_pth, 'tracking_results.pth'))
        

    #         # Run SLAM on the images
    #         if slam is not None: 
    #             slam.track()
    #             slam_results = slam.process()
    #         else:
    #             slam_results = np.zeros((len(images), 7))
    #             slam_results[:, 3] = 1.0    # Unit quaternion
    #         joblib.dump(slam_results, os.path.join(output_pth, 'slam_results.pth'))
        
    #     print("do kp2d features slam done")
        
    #     return

    @property
    def __name__(self, ):
        return 'EMDB'

    def get_kp2d(self):
        return self.tracking_results['kp2d']
    
    def get_features(self):
        return self.tracking_results['features']
        pass
       
    def get_gt_kp2d(self):
        # Retrieve kp2d data here
        return self.labels["kp2d"]
        pass

    def get_weak_kp2d(self):
        # Retrieve weak_kp2d data here
        return torch.zeros_like(self.get_gt_kp2())
        pass

    def get_full_kp2d(self):
        # Retrieve full_kp2d data here
        return self.get_gt_kp2d()
        pass

    def get_init_kp2d(self):
        # Retrieve init_kp2d data here
        return self.labels["kp2d"][0]
        pass

    def get_R(self):
        # Retrieve R data here
        R = self.labels['camera']["extrinsics"][:, :3, :3].clone()
        return R
        pass

    def get_cam_angvel(self):
        # # Retrieve cam_angvel data here
        # slam = SLAMModel(video=self.images, is_images=True, output_pth='path/to/output', width=self.labels["camera"]['width'], height=self.labels["camera"]['height'])
        # if slam is not None: 
        #     slam.track()
        #     slam_results = slam.process()
        # else:
        #     slam_results = np.zeros((self.labels['n_frames'], 7))
        #     slam_results[:, 3] = 1.0    # Unit quaternion
        # Process SLAM results
        cam_angvel = convert_dpvo_to_cam_angvel(self.slam_results, fps=self.fps)
        return cam_angvel
    
    def get_gt_cam_angvel(self):
        # Retrieve gt_cam_angvel data here
        R = self.get_R()
        cam_angvel = transforms.matrix_to_rotation_6d(R[:-1] @ R[1:].transpose(-1, -2))
        cam_angvel = (cam_angvel - torch.tensor([[1, 0, 0, 0, 1, 0]]).to(cam_angvel)) * self.fps
        return cam_angvel
        pass

    def get_cam_poses(self):
        # Retrieve cam_poses data here
        return self.labels['camera']["extrinsics"]
        pass

    def get_res(self):
        # Retrieve res data here
        return (self.labels['camera']['width'], self.labels['camera']['height'])
        pass

    def get_bbox(self):
        # Retrieve bbox data here
        # bboxes format of embd((x_min, y_min, x_max, y_max)) should be changed to (center_x, center_y, scale/200) with square box
        # scaleFactor = 1.1, scale/200 = height/scaleFactor. Don't ask why!TODO
        
        center_x = (self.labels['bboxes']['bboxes'][:, 0] + self.labels['bboxes']['bboxes'][:, 2]) / 2
        center_y = (self.labels['bboxes']['bboxes'][:, 1] + self.labels['bboxes']['bboxes'][:, 3]) / 2
        bbox_w = self.labels['bboxes']['bboxes'][:, 2] - self.labels['bboxes']['bboxes'][:, 0]
        bbox_h = self.labels['bboxes']['bboxes'][:, 3] - self.labels['bboxes']['bboxes'][:, 1]
        scale = torch.stack((bbox_w, bbox_h)).max(0)[0] / self.scaleFactor
        bbox = torch.stack((center_x, center_y, scale)).T

        return bbox

    def get_gt_kp3d(self):
        # Retrieve kp3d data here
        gt_output = self.get_gt_smpl()
        return torch.matmul(self.J_regressor_eval, gt_output.vertices)
        pass

    def get_init_kp3d(self):
        # Retrieve init_kp3d data here
        return self.get_gt_kp3d()[0]
        pass

    def get_pose(self):
        # Retrieve pose data here
        poses_root = torch.tensor(self.labels['smpl']['poses_root'][:])  # Convert numpy array to torch tensor
        poses_body = torch.tensor(self.labels['smpl']['poses_body'][:])  # Convert numpy array to torch tensor
        poses = torch.cat((poses_root, poses_body), dim=1)  # Concatenate tensors along the second dimension
        return poses

    def get_init_pose(self):
        # Retrieve init_pose data here
        return self.get_pose()[0]
        pass
    def get_pose_root(self):
        # Retrieve pose_root data here
        return self.labels['smpl']['poses_root'][:]
        pass

    def get_vel_root(self):
        # Retrieve vel_root data here
        pose_root = self.get_pose_root()
        vel_world = self.labels['smpl']['trans'][1:] - self.labels['smpl']['trans'][:-1]
        pose_root = self.labels['smpl']['poses_root'].clone()
        vel_root = (pose_root[:-1].transpose(-1, -2) @ vel_world.unsqueeze(-1)).squeeze(-1)
        
        return vel_root

    def get_contact(self):
        # Retrieve contact data here
        gt_output = self.get_gt_smpl()

        contact = compute_contact_label(gt_output.feet)
        # if 'tread' in target['vid']:
        #     contact = torch.ones_like(contact) * (-1)
        return contact
        pass

    def get_betas(self):
        # Retrieve beta data here
        return torch.tensor(self.labels['smpl']['betas']).to('cuda') #.to('cuda')
        pass

    def get_gt_smpl(self):
        #smpl_batch_size = self.cfg.TRAIN.BATCH_SIZE * self.cfg.DATASET.SEQLEN
        #self.smpl = build_body_model(self.cfg.DEVICE, smpl_batch_size)
        gt_output = self.smpl.get_output(
            body_pose=torch.tensor(self.labels['smpl']['poses_body'][:]),
            global_orient=torch.tensor(self.labels['smpl']['poses_root'][:]),
            betas=self.get_betas(),
            pose2rot=False
        )

        return gt_output
        

    def get_single_sequence(self, index):
        # Retrieve single sequence data here
        # prepare for path
        self.vid = self.vid[index]
        self.labels = joblib.load(f'{self.global_path}{self.vid}/{self.vid}_data.pkl')
        self.folder_path = f'{self.global_path}{self.vid}'
        self.tracking_results = joblib.load(f'{self.global_path}{self.vid}/tracking_results.pth')
        self.slam_results = joblib.load(f'{self.global_path}{self.vid}/slam_results.pth')
        # read images and store them in self.images
        image_folder = f'{self.global_path}{self.vid}/images/'
        images = []
        for filename in sorted(os.listdir(image_folder)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(image_folder, filename)
                image = cv2.imread(image_path)
                images.append(image)
        self.images = images

        # pre-calculation for some input data-kp2d, features

        # target['bbox'] = self.get_bbox()
        # Register components for training WHAM and each stage
        # input_data = {
        #     'kp2d': kp2d,
        #     'features': features,
        #     'cam_angvel': self.get_cam_angvel(), # this can be derived from both DPVO(slam)
        # }

        # aux_data = {
        #     # 'contact': self.get_contact(),
        #     'res': self.get_res(),
        #     'init_pose': self.get_init_pose(),
        #     'init_kp3d': self.get_init_kp3d(),
        #     'init_kp2d': self.get_init_kp2d(),
            
        # }

        # gt_data = {
        #     'pose': self.get_pose(),
        #     'betas': self.get_betas(),
        #     'vel_root': self.get_vel_root(),
        #     'pose_root': self.get_pose_root(),
        #     'gt_kp3d': self.get_kp3d(), # used for stage 1
        #     'cam_poses': self.get_cam_poses(), # =cam_extrinsics
        #     'R': self.get_R(), # <cam_extrinsics
        #     'gt_cam_angvel': self.get_gt_cam_angvel(), # this is derived from R.
        #     'bbox': self.get_bbox(),
        #     'pose_root': self.get_pose_root(),
        #     'gt_kp2d': self.get_gt_kp2d(),
        #     'weak_kp2d': self.get_weak_kp2d(),
        #     'full_kp2d': self.get_full_kp2d(),
        #     'contact': self.get_contact()
        # }
        target = {
            'kp2d': self.get_kp2d(),
            'features': self.get_features(),
            'cam_angvel': self.get_cam_angvel(),
            'res': self.get_res(),
            'init_pose': self.get_init_pose(),
            'init_kp3d': self.get_init_kp3d(),
            'init_kp2d': self.get_init_kp2d(),
            'pose': self.get_pose(),
            'betas': self.get_betas(),
            'vel_root': self.get_vel_root(),
            'pose_root': self.get_pose_root(),
            'gt_kp3d': self.get_gt_kp3d(),
            'cam_poses': self.get_cam_poses(),
            'R': self.get_R(),
            'gt_cam_angvel': self.get_gt_cam_angvel(),
            'bbox': self.get_bbox(),
            'gt_kp2d': self.get_gt_kp2d(),
            'weak_kp2d': self.get_weak_kp2d(),
            'contact': self.get_contact(),
            'full_kp2d': self.get_full_kp2d()
            
        }

        return target
        pass
