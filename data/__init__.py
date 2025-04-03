import os
import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from .datasets import DeepfakeDataset


def get_weighted_sampler(dataset):
    """Create weighted sampler based on class distribution"""
    class_counts = torch.bincount(dataset.labels)
    class_weights = 1. / class_counts.float()
    sample_weights = class_weights[dataset.labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def custom_collate_fn(batch):
    """Handle mixed tensor/int types and None samples"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    # Convert all elements to tensors
    videos = torch.stack([item['video'] for item in batch])
    audios = pad_sequence(
        [item['audio'].squeeze(0) for item in batch],
        batch_first=True
    ).unsqueeze(1)

    # Explicitly convert labels to tensors
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    video_labels = torch.tensor([item['video_label'] for item in batch], dtype=torch.float32)
    audio_labels = torch.tensor([item['audio_label'] for item in batch], dtype=torch.float32)

    return {
        'video': videos,
        'audio': audios,
        'label': labels,
        'video_label': video_labels,
        'audio_label': audio_labels
    }


def create_dataloader(opt, phase='train'):
    """Create dataloader for specified phase (train/val)"""
    # Set phase-specific options
    phase_opt = type(opt)()  # Create a copy of the options
    for k, v in vars(opt).items():
        setattr(phase_opt, k, v)

    phase_opt.isTrain = (phase == 'train')
    if phase_opt.isTrain:
        phase_opt.dataroot = os.path.join(opt.dataroot, opt.train_split)
    else:
        phase_opt.dataroot = os.path.join(opt.dataroot, opt.val_split)
    print(opt.dataroot)
    dataset = DeepfakeDataset(phase_opt)

    # Only use sampler for training
    sampler = None
    if phase == 'train' and opt.class_bal:
        sampler = get_weighted_sampler(dataset)

    shuffle = (phase == 'train') and not opt.serial_batches and not opt.class_bal

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(opt.num_threads),
        pin_memory=True,
        collate_fn=custom_collate_fn,
        drop_last=(phase == 'train')
    )


def get_train_val_loaders(opt):
    """Get both train and validation loaders"""
    train_loader = create_dataloader(opt, phase='train')
    val_loader = create_dataloader(opt, phase='val')
    return train_loader, val_loader
