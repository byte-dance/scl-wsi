import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageFile
import random
from torchvision.transforms import ToTensor
from torchvision import transforms
import cv2
import pandas as pd

import torch.nn.functional as F

ImageFile.LOAD_TRUNCATED_IMAGES = True


def collate_features(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]


def eval_transforms(pretrained=False):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    trnsfrms_val = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )

    return trnsfrms_val


class CellDataset(data.Dataset):
    def __init__(self, ids):
        super(CellDataset, self).__init__()
        self.ids = pd.read_csv(ids)

    def __getitem__(self, index):
        sample = {}
        info = self.ids.iloc[index]
        file_name, label, feature_path = info.iloc[0], info.iloc[2], info.iloc[3]

        sample['label'] = label
        sample['id'] = file_name

        features = torch.tensor(pd.read_csv(feature_path).values)

        sample['image'] = features
        return sample

    def __len__(self):
        return len(self.ids)


class CellDatasetV2(data.Dataset):
    def __init__(self, ids):
        super(CellDatasetV2, self).__init__()
        self.ids = pd.read_csv(ids)

    def __getitem__(self, index):
        info = self.ids.iloc[index]
        label, feature_path = info.iloc[2], info.iloc[3]
        return feature_path, label

    def __len__(self):
        return len(self.ids)

