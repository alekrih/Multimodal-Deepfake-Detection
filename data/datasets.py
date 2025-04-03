import os

import cv2
import numpy as np
import torch
import torchaudio
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from random import random, choice
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter
from torchvision.transforms import InterpolationMode
from .videofolder import VideoFolder

ImageFile.LOAD_TRUNCATED_IMAGES = True


class IdentityTransform:
    def __call__(self, x):
        return x


class LambdaToTensor:
    def __call__(self, x):
        return x


class DeepfakeDataset:
    def __init__(self, opt):
        self.opt = opt
        # self.class_weights = self._calculate_class_weights()
        self.dataset = self._create_dataset('train' if opt.isTrain else 'val')
        self._calculate_dynamic_weights()  # Calculate initial weights

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if item is None:
            return None
        # Add dynamic weight to each sample
        item['weight'] = self.class_weights[item['label']]
        return item

    def _calculate_dynamic_weights(self):
        # Count actual samples per class in this dataset
        class_counts = torch.bincount(self.dataset.labels)
        self.class_weights = 1. / (class_counts.float() + 1e-6)  # Avoid division by zero
        self.class_weights = self.class_weights / self.class_weights.sum()  # Normalize

    def create_loaders(self):
        train_set = self._create_dataset('train')
        val_set = self._create_dataset('val')

        # Weighted sampler for training
        if self.opt.isTrain:
            weights = [self.class_weights[label] for label in train_set.labels]
            sampler = torch.utils.data.WeightedRandomSampler(
                weights, len(weights), replacement=True
            )
        else:
            sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.opt.batch_size,
            sampler=sampler,
            num_workers=self.opt.num_threads,
            pin_memory=True,
            drop_last=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=self.opt.num_threads,
            pin_memory=True
        )

        return train_loader, val_loader

    def _calculate_class_weights(self):
        class_counts = torch.tensor([435, 435, 8539, 9529])  # RR, RF, FR, FF
        return 1. / class_counts

    def _create_dataset(self, phase):
        # root = os.path.join(self.opt.dataroot, phase)
        root = self.opt.dataroot
        # print(root)
        transform = transforms.Compose([
            self._get_resize_transform(),
            self._get_crop_transform(),
            self._get_flip_transform(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        audio_transform = torchaudio.transforms.Resample(
            orig_freq=44100, new_freq=16000
        )

        return VideoFolder(
            root=root,
            transform=transform,
            audio_transform=audio_transform,
            frame_limit=getattr(self.opt, 'frame_limit', 6),  # Safe access with default
            phase=phase
        )

    def _get_resize_transform(self):
        no_resize = getattr(self.opt, 'no_resize', False)
        if no_resize and not self.opt.isTrain:
            return IdentityTransform()
        load_size = getattr(self.opt, 'loadSize', 256)
        return transforms.Resize((load_size, load_size))

    def _get_crop_transform(self):
        no_crop = getattr(self.opt, 'no_crop', False)
        if no_crop:
            return IdentityTransform()
        crop_size = getattr(self.opt, 'cropSize', 224)
        if self.opt.isTrain:
            return transforms.RandomCrop(crop_size)
        return transforms.CenterCrop(crop_size)

    def _get_flip_transform(self):
        no_flip = getattr(self.opt, 'no_flip', False)
        if not self.opt.isTrain or no_flip:
            return IdentityTransform()
        return transforms.RandomHorizontalFlip()
