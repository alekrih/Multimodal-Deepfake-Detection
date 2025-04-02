import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger
from data import create_dataloader
from validate import validate
from networks.unified_model import UnifiedModel

# Test config
vals = ['']
multiclass = [0, 1]


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    return val_opt


if __name__ == '__main__':
    opt = TrainOptions().parse()
    Testdataroot = os.path.join(opt.dataroot, 'test')
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))
    print('  '.join(list(sys.argv)))
    val_opt = get_val_opt()
    Testopt = TestOptions().parse(print_options=False)
    data_loader = create_dataloader(opt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UnifiedModel(opt, device).to(device)
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=opt.lr, betas=(opt.beta1, 0.999))

    def testmodel():
        print(Testdataroot)
        print('*' * 25)
        accs = []
        aps = []
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        for v_id, val in enumerate(vals):
            Testopt.dataroot = '{}/{}'.format(Testdataroot, val)
            Testopt.classes = ''
            Testopt.no_resize = False
            Testopt.no_crop = True
            acc, ap, _, _, _, _ = validate(model, Testopt)
            accs.append(acc)
            aps.append(ap)
            print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc * 100, ap * 100))
        print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id + 1, 'Mean', np.array(accs).mean() * 100,
                                                          np.array(aps).mean() * 100))
        print('*' * 25)
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

    model.train()
    for epoch in range(opt.niter):
        print("Epoch: {0}".format(epoch))
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(data_loader):
            if data is None:
                continue
            audio_input = data['audio'].to(device)
            video_input = data['video'].to(device)
            labels = data['label'].to(device)
            video_labels = (labels == 2) | (labels == 3)
            audio_labels = (labels == 1) | (labels == 3)
            video_labels = video_labels.float()
            audio_labels = audio_labels.float()
            outputs = model(audio_input, video_input)
            video_logits = outputs[:, 0]
            audio_logits = outputs[:, 1]

            video_loss = criterion(video_logits, video_labels)
            audio_loss = criterion(audio_logits, audio_labels)
            total_loss = video_loss + audio_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if i % opt.loss_freq == 0:
                print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                      "Train loss: {} at step: {}".format(total_loss.item(), i))
                train_writer.add_scalar('loss', total_loss.item(), epoch * len(data_loader) + i)

            model.train()
        model.eval()
        acc, ap = validate(model, val_opt)[:2]
        val_writer.add_scalar('accuracy', acc, epoch)
        val_writer.add_scalar('ap', ap, epoch)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))
        if epoch % opt.save_epoch_freq == 0:
            torch.save(model.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, f'epoch_{epoch}.pth'))
    model.eval()
    testmodel()
    torch.save(model.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, 'final_model.pth'))