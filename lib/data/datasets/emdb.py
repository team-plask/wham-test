import torch
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor
from configs.config import get_cfg_defaults
import os
import cv2
import joblib

cfg = get_cfg_defaults()

class EMDBDataset:
    def __init__(self):
        # Initialize the dataset here
        label_pth = 'path/to/label.pth'
        self.labels = joblib.load(label_pth)
        # load_an array of images from 'folder_path'
        folder_path = 'path/to/folder'
        images = []
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)
                images.append(image)
        self.images = images
        pass

    def get_kp2d_feature(self):
        # Run detection model to get kp2d
        detector = DetectionModel(cfg.DEVICE.lower())  # Replace with your actual detection model
        extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)

        images = self.images
        
        # Run detection model on each image
        # EMBD dataset's cropped images' fps is 30
        detector.track(images, fps=30, length=len(images))
        tracking_results = detector.process(fps=30)
        tracking_results = extractor.run(video=images, tracking_results=tracking_results)

        return tracking_results['kp2d'] ,tracking_results['features']

            
    def get_gt_kp2d(self):
        # Retrieve kp2d data here
        pass

    def get_R(self):
        # Retrieve R data here
        pass

    def get_cam_angvel(self):
        # Retrieve cam_angvel data here
        if run_global: slam = SLAMModel(video=self.images, output_pth, width, height, calib)
        if slam is not None: 
            slam.track()
            slam_results = slam.process()
        else:
            slam_results = np.zeros((length, 7))
            slam_results[:, 3] = 1.0    # Unit quaternion
        # Process SLAM results
        cam_angvel = convert_dpvo_to_cam_angvel(self.slam_results, self.fps)
        return cam_angvel

    def get_cam_poses(self):
        # Retrieve cam_poses data here
        pass

    def get_res(self):
        # Retrieve res data here
        pass

    def get_bbox(self):
        # Retrieve bbox data here
        pass

    def get_init_pose(self):
        # Retrieve init_pose data here
        pass

    def get_init_kp3d(self):
        # Retrieve init_kp3d data here
        pass

    def get_init_kp2d(self):
        # Retrieve init_kp2d data here
        pass

    def get_features(self):
        # Retrieve features data here
        pass

    def get_betas(self):
        # Retrieve betas data here
        pass

    def get_pose(self):
        # Retrieve pose data here
        pass

    def get_vel_root(self):
        # Retrieve vel_root data here
        pass

    def get_pose_root(self):
        # Retrieve pose_root data here
        pass

    def get_contact(self):
        # Retrieve contact data here
        pass

    def get_single_sequence(self):
        # Retrieve single sequence data here
        
        # pre-calculation for some input data-kp2d, features
        kp2d, features = self.get_kp2d_feature()

        # target['bbox'] = self.get_bbox()
        # Register components for training WHAM and each stage
        input_data = {
            'kp2d': kp2d,
            'features': features,
            'cam_angvel': self.get_cam_angvel(), # this can be derived from both DPVO(slam)
        }

        aux_data = {
            # 'contact': self.get_contact(),
            'res': self.get_res(),
            'init_pose': self.get_init_pose(),
            'init_kp3d': self.get_init_kp3d(),
            'init_kp2d': self.get_init_kp2d(),
            
        }

        gt_data = {
            'pose': self.get_pose(),
            'betas': self.get_betas(),
            'vel_root': self.get_vel_root(),
            'pose_root': self.get_pose_root(),
            'gt_kp3d': self.get_kp3d(), # used for stage 1
            'cam_poses': self.get_cam_poses(), # =cam_extrinsics
            'R': self.get_R(), # <cam_extrinsics
            'gt_cam_angvel': self.get_gt_cam_angvel(), # this is derived from R.
            'bbox': self.get_bbox(),
            'vel_root': self.get_vel_root(),
            'pose_root': self.get_pose_root(),
            'gt_kp2d': self.get_gt_kp2d(),
            'weak_kp2d': self.get_weak_kp2d(),
            'full_kp2d': self.get_full_kp2d(),
            'contact': self.get_contact()
        }

        # Additional components for each stage


        # Register components for each stage

        # Additional components for each stage
        stage1_data = {
            'bbox': self.get_bbox()
        }

        #data for groundtruth
        target['pose'] = self.get_pose()
        target['betas'] = self.get_betas()
        target['vel_root'] = self.get_vel_root() # transl is required
        target['pose_root'] = self.get_pose_root() #
        target['']
        pass
