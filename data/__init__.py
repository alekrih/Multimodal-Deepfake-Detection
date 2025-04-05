import os
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from .datasets import DeepfakeDataset


def get_weighted_sampler(dataset, opt):
    labels = dataset.dataset.labels
    class_counts = torch.bincount(labels)
    class_weights = 1.0 / (class_counts.float() + 1e-6)
    sample_weights = class_weights[labels]
    minority_classes = [0, 1]  # RR=0, RF=1
    oversample_factor = opt.oversample_weight
    for cls in minority_classes:
        cls_mask = (labels == cls)
        sample_weights[cls_mask] *= oversample_factor
    sample_weights = sample_weights / sample_weights.mean()
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

    videos = torch.stack([item['video'] for item in batch])
    audios = pad_sequence(
        [item['audio'].squeeze(0) for item in batch],
        batch_first=True
    ).unsqueeze(1)

    # Explicitly convert labels to tensors
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    video_labels = torch.tensor([item['video_label'] for item in batch], dtype=torch.float32)
    audio_labels = torch.tensor([item['audio_label'] for item in batch], dtype=torch.float32)
    weights = torch.tensor([item['weight'] for item in batch], dtype=torch.float32)

    return {
        'video': videos,
        'audio': audios,
        'label': labels,
        'video_label': video_labels,
        'audio_label': audio_labels,
        'weight': weights
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
    dataset = DeepfakeDataset(phase_opt)

    # Only use sampler for training
    sampler = None
    if phase == 'train' and (opt.class_bal or hasattr(opt, 'oversample_weight')):
        sampler = get_weighted_sampler(dataset, opt)

    shuffle = (phase == 'train') and not opt.serial_batches and (sampler is None)

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
