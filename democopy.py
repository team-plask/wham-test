import os
import argparse
import os.path as osp
from glob import glob
from collections import defaultdict

import cv2
import torch
import joblib
import numpy as np
from loguru import logger
from progress.bar import Bar

from configs.config import get_cfg_defaults
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor

try: 
    from lib.models.preproc.slam import SLAMModel
    _run_global = True
    print("DPVO module loaded")
except: 
    logger.info('DPVO is not properly installed. Only estimate in local coordinates !')
    _run_global = False

def run(cfg,
        global_path = "dummy/test/",
        image_subdir = "images/", #image directory under vid/ directory
        redo_tracking = False,
        redo_slam = True,
        calib=None,
        run_global=True,
        save_pkl=False,
        visualize=False):
    
    
    # Whether or not estimating motion in global coordinates
    run_global = run_global and _run_global
    
    # Preprocess
    bar2 = Bar(f'Preprocess: vid in {global_path}', fill='#', max=len(os.listdir(global_path)))
    folders = [f for f in sorted(os.listdir(global_path)) if os.path.isdir(os.path.join(global_path, f))]
    for vid in sorted(folders):
        output_pth = f'{global_path}{vid}'
        print(vid)
        with torch.no_grad():
            do_2d_detection = (not osp.exists(osp.join(output_pth, 'tracking_results.pth'))) or redo_tracking
            if do_2d_detection: print("do_2d_detection is True")
            do_slam = (not osp.exists(osp.join(output_pth, 'slam_results.pth'))) or redo_slam
            if do_slam: print("do_slam is True")
            if do_2d_detection or do_slam:

                detector = DetectionModel(cfg.DEVICE.lower())
                extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)

                image_folder = f'{output_pth}/{image_subdir}'
                images = []
                image_list=[]
                for filename in sorted(os.listdir(image_folder)):
                    if filename.endswith(".jpg") or filename.endswith(".png"):
                        image_path = os.path.join(image_folder, filename)
                        image = cv2.imread(image_path)
                        images.append(image)
                        image_list.append(image_path)
                length = len(images)
                fps = 30
                width, height = images[0].shape[:2]
                if run_global: slam = SLAMModel(image_folder, output_pth, width, height, calib, is_images=True)
                else: slam = None

                if slam is None: print("slam is None")

                bar = Bar('Preprocess: 2D detection and SLAM', fill='#', max=length)
                #while (cap.isOpened()):

                
                for img in images:
                    #flag, img = cap.read()
                    #if not flag: break

                    # 2D detection and tracking
                    if do_2d_detection:
                        detector.track(img, fps, length)

                    # SLAM
                    if do_slam and slam is not None:
                        slam.track()

                    bar.next()

                print("doing detector.process")
                if do_2d_detection:
                    tracking_results = detector.process(fps)

                print("doing slam.process")
                if slam is not None and do_slam: 
                    slam_results = slam.process()
                elif do_slam:
                    slam_results = np.zeros((length, 7))
                    slam_results[:, 3] = 1.0    # Unit quaternion

                print("doing extractor.run")
                # Extract image features
                # TODO: Merge this into the previous while loop with an online bbox smoothing.
                if do_2d_detection:
                    tracking_results = extractor.run(image_list, tracking_results, is_video=False)
                logger.info('Complete Data preprocessing!')

                # Save the processed data
                if do_2d_detection:
                    joblib.dump(tracking_results, osp.join(output_pth, 'tracking_results.pth'))
                if do_slam:
                    joblib.dump(slam_results, osp.join(output_pth, 'slam_results.pth'))
                logger.info(f'Save processed data at {output_pth}')
        bar2.next()
    bar2.finish()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--global_path', type=str, default='dummy/test/', 
                        help='output folder to write results')
    
    parser.add_argument('--image_subdir', type=str, default='images/', 
                        help='output folder to write results')

    parser.add_argument('--calib', type=str, default=None, 
                        help='Camera calibration file path')

    parser.add_argument('--estimate_local_only', action='store_true',
                        help='Only estimate motion in camera coordinate if True')
    
    parser.add_argument('--run_smplify', action='store_true',
                        help='Run Temporal SMPLify for post processing')

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')
    
    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')    
    
    
    run(cfg=cfg,
        global_path=args.global_path,
        image_subdir=args.image_subdir,
        calib=args.calib,
        run_global=not args.estimate_local_only)

        
    print()
    logger.info('Done !')