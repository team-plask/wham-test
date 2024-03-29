import torch
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor
from configs.config import get_cfg_defaults
import os
import cv2
import joblib
import numpy as np
from lib.models import build_body_model
from lib.data._dataset import BaseDataset
try: 
    from lib.models.preproc.slam import SLAMModel
    run_global = True
except: 
    # logger.info('DPVO is not properly installed. Only estimate in local coordinates !')
    #print("No Slam available?????????????")
    run_global = False

cfg = get_cfg_defaults()

class EMDBDataset(BaseDataset):
    def __init__(self):
        # Initialize the dataset here
        # load_an array of images from 'folder_path'
        self.images = []
        self.fps = 30
        self.global_path = 'dummy/test/'
        self.vid = os.listdir(self.global_path)

        for vid in self.vid:
            self.do_kp2d_features_slam(vid)

        pass        

        pass
    def do_kp2d_features_slam(self, vid: str):
        # Run detection model to get kp2d
        # Run detection model to get kp2d

        output_pth = f'{self.global_path}{vid}'

        if not (os.path.exists(os.path.join(output_pth, 'tracking_results.pth')) and 
                os.path.exists(os.path.join(output_pth, 'slam_results.pth'))):
            detector = DetectionModel(cfg.DEVICE.lower())  # Replace with your actual detection model
            extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)

            image_folder = f'{self.global_path}{vid}/images/'
            images = []
            for filename in sorted(os.listdir(image_folder)):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(image_folder, filename)
                    image = cv2.imread(image_path)
                    images.append(image)

            width, height = images[0].shape[:2]
            slam = SLAMModel(video=image_folder, is_images=True, output_pth=output_pth, width=width, height=height)
            # Run detection model on each image
            # EMBD dataset's cropped images' fps is 30
            for img in images:
                # 2D detection and tracking
                detector.track(img, self.fps, length=len(images))
                
                # SLAM
                if slam is not None: 
                    slam.track()

            tracking_results = detector.process(fps=self.fps)
            tracking_results = extractor.run(video=images, tracking_results=tracking_results)

            joblib.dump(tracking_results, os.path.join(output_pth, 'tracking_results.pth'))
        

            # Run SLAM on the images
            if slam is not None: 
                slam.track()
                slam_results = slam.process()
            else:
                slam_results = np.zeros((len(images), 7))
                slam_results[:, 3] = 1.0    # Unit quaternion
            joblib.dump(slam_results, os.path.join(output_pth, 'slam_results.pth'))
        
        #print("do kp2d features slam done")
        
        return



a = EMDBDataset()



