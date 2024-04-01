import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import joblib
from skimage.util.shape import view_as_windows

class VideoDataset(Dataset):
    def __init__(self, video_data, video_info):
        """
        video_data: 비디오 데이터를 포함하는 리스트. 각 요소는 하나의 비디오에 해당하며, 
                    프레임들을 포함하는 numpy 배열이거나 텐서로 가정합니다.
        video_info: 각 비디오에 대한 추가 정보를 포함하는 리스트.
        """
        self.video_data = video_data
        self.video_info = video_info
        self.n_frames = 81  # 샘플당 프레임 수

        # 모든 비디오를 81 길이의 프레임으로 자르고, 샘플 목록을 생성합니다.
        self.samples = self._create_samples()
    
    def _create_samples(self):
        #samples is list of (vid, start_index, end_index)
        samples = []

        vidList = os.listdir(self.global_path)
        # for one video
        r = self.epoch % 4
        for vid in vidList:
            # get target data
            target_path = os.path.join(self.global_path, vid)
            image_path = os.path.join(target_path, 'images')
            image_files = os.listdir(image_path)
            num_images = len(image_files)
            # if the whole sequence length of one video is smaller than n_frames, skip
            if num_images < self.n_frames: continue

            #make indexes (0, 1, 2, ..., num_images-1)
            indexes = np.arange(num_images)
            #make chunks of indexes with widnow_length: n_frames, step : n_frames//4
            chunks = view_as_windows(
                indexes, (self.n_frames), step=self.n_frames // 4
            )
            start_finish = chunks[r::4, (0, -1)].tolist()
            samples.append((vid, start_finish))

        return samples

    def __len__(self):
        # 데이터셋의 총 샘플 수
        return len(self.samples)

    def __getitem__(self, idx):
        # idx 인덱스에 해당하는 샘플을 반환합니다.
        frames, info = self.samples[idx]
        return frames, info
    
    def get_single_item(self, video, info):
        """
        info = sliced list of images for one video, in length of self.n_frames
        """
        # 비디오 데이터와 추가 정보를 입력으로 받아 처리하는 함수
        pass

# 예제 데이터셋 생성
video_data = [torch.randn(100, 10, 10) for _ in range(5)]  # 예: 5개의 비디오, 각 비디오는 100개의 10x10 프레임
video_info = [{'id': i} for i in range(5)]  # 각 비디오에 대한 정보

dataset = VideoDataset(video_data, video_info)

# DataLoader 사용
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# 배치 처리 예제
for frames, info in data_loader:
    print(frames.shape, info)  # frames: [배치 크기, 81, 프레임 높이, 프레임 너비]
