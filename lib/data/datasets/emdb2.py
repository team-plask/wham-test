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
from .dataset3d import Dataset3D
from ...utils import transforms
from .dataset_custom import convert_dpvo_to_cam_angvel
import multiprocessing
from configs import constants as _C
from ..utils.normalizer import Normalizer
from skimage.util.shape import view_as_windows

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
        #multiprocessing.set_start_method('spawn')
        super(EMDBDataset, self).__init__(cfg, training=True)
        self.labels = []
        # load_an array of images from 'folder_path'
        self.folder_path = 'folder/to/global_path/vid/'
        self.images = [] #array of images
        self.fps = 30
        self.scaleFactor = 1.1
        #smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
        #self.smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
        self.global_path = 'dummy/test/' #folder/to/emdb
        self.vidlist = []
        self.video_indices = []
        self.vid = []
        items = os.listdir(self.global_path)
        index = 0
        count = 0
        # subdirs = [item for item in items if os.path.isdir(f'{self.global_path}/{item}')]
        # for item in subdirs:
        #     # 폴더 이름을 self.vidlist에 추가
        #     self.vidlist.append(item)
        #     #vid 추가
        #     start = count
        #     for os.path in os.listdir(f'{self.global_path}/{item}' + '/images'):
        #         self.vid.append(index)
        #         count += 1
        #     # 여기서는 예시를 위해 self.video_indices에 (0, 0)을 추가
        #     self.video_indices.append((start, count-1))
        # print(self.vidlist)
        # self.vid = torch.tensor(self.vid)
        # self.video_indices = torch.tensor(self.video_indices)
        self.epoch = 0
        #self.labels = { 'vid': self.vidlist.copy() }
        self.labels = { 'vid': "P0_04_mvs_e" }
        self.n_frames = 667
        self.cfg = cfg
        self.tracking_results = None
        self.slam_results = None
        self.target=None

        label_pth = _C.PATHS.AMASS_LABEL

        self.supervise_pose = cfg.TRAIN.STAGE == 'stage1'
        self.labels = joblib.load(label_pth)

        self.keypoints_normalizer = Normalizer(cfg)
        # Load augmentators
        
        self.n_frames = 81
        self.smpl = build_body_model('cpu', self.n_frames)
        self.samples = []
        self.prepare_video_batch()

        
        # Naive assumption of image intrinsics
        self.img_w, self.img_h = 1000, 1000
        self.get_naive_intrinsics((self.img_w, self.img_h))
        

        #for vid in self.vid:
        #    self.do_kp2d_features_slam(vid)

        self.vid = None
        self.start = None
        self.end = None

        pass
    @property
    def __name__(self, ):
        return 'EMDB'
    
    def prepare_video_batch(self):
        #samples is list of (vid, start_index, end_index)
        self.samples = []
        vidList = os.listdir(self.global_path)
        special_n = 4
        r = self.epoch % special_n
        for vid in vidList:
            # get target data
            target_path = os.path.join(self.global_path, vid)
            image_path = os.path.join(target_path, 'images')
            image_files = os.listdir(image_path)
            num_images = len(image_files)
            if vid == "P1_14_outdoor_climb":
                num_images = 1000
            # if the whole sequence length of one video is smaller than n_frames, skip
            if num_images < self.n_frames: continue

            #make indexes (0, 1, 2, ..., num_images-1)
            indexes = np.arange(num_images)
            #make chunks of indexes with widnow_length: n_frames, step : n_frames//4
            chunks = view_as_windows(
                indexes, (self.n_frames), step=(self.n_frames // special_n)
            )
            start_finish = chunks[r::4, (0, -1)].tolist()
            for sf in start_finish:
                self.samples.append((vid, sf))
                #print(f"vid: {vid}, start: {sf[0]}, end: {sf[1]}")
    
    def __len__(self):
        # 데이터셋의 총 샘플 수
        return len(self.samples)

    def get_kp2d(self):
        try:
            return self.target['kp2d'][self.start:self.end].detach()
        except KeyError:
            print("KeyError: 'kp2d' key not found in target dictionary.")
            return None
    
    def get_features(self):
        return self.tracking_results[0]['features'][self.start:self.end].clone()
        pass
       
    def get_gt_kp2d(self):
        # Retrieve kp2d data here
        return self.target["gt_kp2d"][self.start:self.end].detach()
        pass

    def get_weak_kp2d(self):
        # Retrieve weak_kp2d data here
        return self.target['weak_kp2d'][self.start:self.end].detach().squeeze()
        pass

    def get_full_kp2d(self):
        # Retrieve full_kp2d data here
        
        kp2d = self.target['full_kp2d'][self.start:self.end].detach().squeeze()
        ones_to_add = torch.ones(kp2d.size(0), kp2d.size(1), 1)
        return torch.cat((kp2d, ones_to_add), dim=2)
        return self.target['full_kp2d'].detach().squeeze()
        pass

    def get_init_kp2d(self):
        # Retrieve init_kp2d data here
        # (1, 17*2 + 3)
        temp = self.get_kp2d()
        if temp is None or temp.shape[0]<self.n_frames-5:
            print("kp2d is None", self.vid, self.start, self.end)
        return self.get_kp2d()[0].unsqueeze(0)

        pass

    def get_R(self):
        # Retrieve R data here
        if "extrinsics" not in self.ori['camera']:
            print(self.vid)
            raise KeyError("Extrinsics data not found.")
        return torch.tensor(self.ori['camera']["extrinsics"][self.start:self.end, :3, :3]).detach()
        pass

    def get_cam_angvel(self):
        # # Retrieve cam_angvel data here
        # slam = SLAMModel(video=self.images, is_images=True, output_pth='path/to/output', width=self.target["camera"]['width'], height=self.target["camera"]['height'])
        # if slam is not None: 
        #     slam.track()
        #     slam_results = slam.process()
        # else:
        #     slam_results = np.zeros((self.target['n_frames'], 7))
        #     slam_results[:, 3] = 1.0    # Unit quaternion
        # Process SLAM results
        cam_angvel = self.target['cam_angvel'][self.start:self.end].detach()
        return cam_angvel
    
    def get_gt_cam_angvel(self):
        # Retrieve gt_cam_angvel data here
        cam_angvel = self.target['gt_cam_angvel'][self.start:self.end].detach()
        return cam_angvel
        pass

    def get_cam_poses(self):
        # Retrieve cam_poses data here
        cam_pose = self.target['cam_poses'][self.start:self.end].detach()
        return cam_pose
        pass

    def get_res(self):
        # Retrieve res data here
        return torch.tensor(self.target['res']).clone()
        pass

    def get_bbox(self):
        # Retrieve bbox data here
        # bboxes format of embd((x_min, y_min, x_max, y_max)) should be changed to (center_x, center_y, scale/200) with square box
        # scaleFactor = 1.1, scale/200 = height/scaleFactor. Don't ask why!TODO
        #bbox =self.target['bbox'][self.start:self.end].detach()
        bbox = self.tracking_results[0]['bbox'][self.start:self.end]

        return torch.tensor(bbox)

    def get_gt_kp3d(self):
        # Retrieve kp3d data here
        kp3d = self.target['gt_kp3d'][self.start:self.end].detach()
        return kp3d
        # ones_to_add = torch.ones(kp3d.size(0), kp3d.size(1), 1)

        # return torch.cat((kp3d, ones_to_add), dim=2)
        pass

    def get_init_kp3d(self):
        # Retrieve init_kp3d data here
        init = self.get_gt_kp3d()[:1, :self.n_joints, :-1] #get_gt_kp3d is already cropped 
        init = init.reshape(-1, 17*3)
        return init
        pass

    def get_pose(self):
        # Retrieve pose data here
        poses = self.target['pose'][self.start:self.end].detach()
        return poses

    def get_init_pose(self):
        # Retrieve init_pose data here
        return self.target['pose'][self.start].detach().unsqueeze(0)
        return self.get_pose()[0]
        pass
    def get_pose_root(self):
        # Retrieve pose_root data here
        # 마지막 값 복제하여 추가
        gt_pose_root = self.target['pose_root'][self.start:self.end].detach()
        gt_pose_root_extended = torch.cat([gt_pose_root, gt_pose_root[-1:, :]], dim=0)

        return gt_pose_root_extended
        return self.labels['smpl']['poses_root'][:]
        pass
    def get_init_root(self):
        # Retrieve init_root data here
        return self.get_pose_root()[0].unsqueeze(0)
        pass

    def get_vel_root(self):
        # Retrieve vel_root data here
        return self.target['vel_root'][self.start:self.end].detach()
        pose_root = self.get_pose_root()
        vel_world = self.labels['smpl']['trans'][1:] - self.labels['smpl']['trans'][:-1]
        pose_root = self.labels['smpl']['poses_root'].clone()
        vel_root = (pose_root[:-1].transpose(-1, -2) @ vel_world.unsqueeze(-1)).squeeze(-1)
        
        return vel_root

    def get_contact(self):
        # Retrieve contact data here
        return self.target['contact'][self.start:self.end].detach()
        gt_output = self.get_gt_smpl()

        contact = compute_contact_label(gt_output.feet)
        # if 'tread' in target['vid']:
        #     contact = torch.ones_like(contact) * (-1)
        return contact
        pass

    def get_betas(self):
        betas = self.target['betas'].detach()
        betas = betas.repeat(self.n_frames-1, 1)
        return betas
        return self.target['betas'].detach().unsqueeze(0)
        # Retrieve beta data here
        return torch.tensor(self.labels['smpl']['betas']).to('cuda') #.to('cuda')
        pass

    def get_gt_smpl(self):
        #not used
        return self.target['gt_smpl'][self.start:self.end]
        #smpl_batch_size = self.cfg.TRAIN.BATCH_SIZE * self.cfg.DATASET.SEQLEN
        #self.smpl = build_body_model(self.cfg.DEVICE, smpl_batch_size)
        gt_output = self.smpl.get_output(
            body_pose=torch.tensor(self.labels['smpl']['poses_body'][:]),
            global_orient=torch.tensor(self.labels['smpl']['poses_root'][:]),
            betas=self.get_betas(),
            pose2rot=False
        )

        return gt_output
    
    def get_cam_intrinsics(self):
        # Retrieve cam_intrinsics data here
        #[1,3,3]
        return torch.tensor(self.ori['camera']['intrinsics']).detach().unsqueeze(0)
        pass

    def get_mask(self, vis_thr=0.6):
        # Retrieve mask data here

        temp= self.tracking_results[0]["keypoints"][self.start:self.end][..., -1] < vis_thr
        return torch.tensor(temp)
        return torch.ones((self.n_frames, 17), dtype=torch.bool)
        pass
    

    def get_single_sequence(self, index):
        # Retrieve single sequence data here
        # prepare for path
        #print(index)
        vid = self.samples[index][0]
        self.vid = vid
        self.range = self.samples[index][1]
        self.start = self.samples[index][1][0]
        self.end = self.samples[index][1][1]

        #self.labels = joblib.load(f'{self.global_path}{self.vid}/{self.vid}_data.pkl')
        self.folder_path = f'{self.global_path}{vid}'
        self.tracking_results = joblib.load(f'{self.global_path}{vid}/tracking_results.pth')
        self.slam_results = joblib.load(f'{self.global_path}{vid}/slam_results.pth')
        self.target = joblib.load(f'{self.global_path}{vid}/target.pkl')
        self.ori = joblib.load(f'{self.global_path}{vid}/{vid}_data.pkl')
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
            'features': self.get_features().clone(),
            'cam_angvel': self.get_gt_cam_angvel().clone(), # not dpvo yet
            'res': self.get_res(),
            'init_pose': self.get_init_pose().clone(),
            'init_kp3d': self.get_init_kp3d().clone(),
            'init_kp2d': self.get_init_kp2d().clone(),
            'pose': self.get_pose().clone(),
            'betas': self.get_betas().clone(),
            'vel_root': self.get_vel_root().clone(),
            'pose_root': self.get_pose_root().clone(),
            'kp3d': self.get_gt_kp3d().clone(),
            'cam': self.get_cam_poses().clone(),
            'R': self.get_R().clone(),
            'gt_cam_angvel': self.get_gt_cam_angvel().clone(),
            'bbox': self.get_bbox().clone(),
            'gt_kp2d': self.get_gt_kp2d().clone(),
            'weak_kp2d': self.get_weak_kp2d().clone(),
            'contact': self.get_contact().clone(),
            'full_kp2d': self.get_full_kp2d().clone(),
            'init_root': self.get_init_root().clone(),
            'cam_intrinsics': self.get_cam_intrinsics().clone(),
            'has_smpl': torch.tensor(True),
            'has_full_screen': torch.tensor(True),
            'has_verts': torch.tensor(False),
            'mask' : self.get_mask().clone()
            #'vid' : torch.tensor(vid)
        }
        # # 2D keypoints detection
        # kp2d = target['kp2d']
        # bbox = target['bbox'].clone()
        # bbox[:, 2] = bbox[:, 2] / 200
        #print("kp3d shape", target['kp3d'].shape)
        # print("bbox shape", bbox.shape)
        #target['kp2d'], target['bbox'] = self.keypoints_normalizer(torch.tensor(target['kp2d']).clone(), target['res'], self.cam_intrinsics, 224, 224, torch.tensor(self['bbox']).clone()) 

        # # 2d gt
        # kp2d = target['gt_kp2d']
        # bbox = target['bbox'].clone()
        # target['gt_kp2d'], target['bbox'] = self.keypoints_normalizer(kp2d, target['res'], self.cam_intrinsics, 224, 224, bbox)

        # path = "dummy/keypoints3d/static/animations"
        #     # write a file
        # folder = f"epoch_index{index}_{self.vid}_gt"
        # os.makedirs(f"{path}/{folder}", exist_ok=True)
        # for frame in range(len(target['kp3d'])):
        #     with open(f"{path}/{folder}/{frame:03}.obj", "w") as file:
        #         for joint in range(len(target['kp3d'][frame])):
        #             # CUDA 텐서를 CPU로 옮기고, .item()으로 실제 값을 가져온 후, 문자열로 변환
        #             a = target['kp3d'][frame][joint][0]
        #             b = target['kp3d'][frame][joint][1]
        #             c = target['kp3d'][frame][joint][2] 
        #             file.write(f"v {a} {b} {c}\n")

        # for key, value in target.items():
        #     print(f"{key}: {value.shape}")
        #print(target)
        #print("No R is nonsensical", target['R'])
        return target
        pass
