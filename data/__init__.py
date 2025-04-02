import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from .datasets import dataset_folder

'''
def get_dataset(opt):
    dset_lst = []
    for cls in opt.classes:
        root = opt.dataroot + '/' + cls
        dset = dataset_folder(opt, root)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)
'''

import os


def get_dataset(opt):
    # opt.dataroot = os.path.join(opt.dataroot, "videos")
    classes = os.listdir(opt.dataroot) if len(opt.classes) == 0 else opt.classes
    if '0_real' not in classes or '1_fake' not in classes:
        dset_lst = []
        for cls in classes:
            root = opt.dataroot + '/' + cls
            dset = dataset_folder(opt, root)
            dset_lst.append(dset)
        return torch.utils.data.ConcatDataset(dset_lst)
    return dataset_folder(opt, opt.dataroot)


def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    videos = [item['video'] for item in batch]
    audios = [item['audio'] for item in batch]
    labels = [item['label'] for item in batch]
    videos = torch.stack(videos, dim=0)
    labels = torch.tensor(labels)
    audios_padded = pad_sequence(audios, batch_first=True)
    return {'video': videos, 'audio': audios_padded, 'label': labels}


def create_dataloader(opt):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset(opt)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None
    # print(len(dataset))
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=int(opt.num_threads),
                                              collate_fn=custom_collate_fn,)
    return data_loader
