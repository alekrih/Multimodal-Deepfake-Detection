import os
import sys
import time
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger
from data import create_dataloader
from validate import validate
from networks.unified_model import UnifiedModel


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    return val_opt


class UnifiedTrainer:
    def __init__(self, opt):
        self.opt = opt
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize model and optimizer
        self.model = UnifiedModel(self.device).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)

        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Monitor validation AP
            factor=0.5,  # Reduce LR by half
            patience=3,  # Number of epochs with no improvement
            verbose=True  # Print messages
        )

        # Loss functions with class weighting
        self.video_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([(435 + 435) / (8539 + 9529)]).to(self.device)
        )
        self.audio_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([(435 + 8539) / (435 + 9529)]).to(self.device)
        )

        # Create data loaders
        self.train_loader = create_dataloader(opt, phase='train')
        self.val_loader = create_dataloader(opt, phase='val')

        # Initialize logging
        self.writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name))
        self.best_val_ap = 0
        self._init_dynamic_loss()

    def _init_dynamic_loss(self):
        # Initialize base loss functions
        self.video_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.audio_criterion = nn.BCEWithLogitsLoss(reduction='none')

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        self._update_weights()
        for batch_idx, data in enumerate(self.train_loader):
            # Move data to device
            audio = data['audio'].to(self.device)
            video = data['video'].to(self.device)
            video_labels = data['video_label'].to(self.device)
            audio_labels = data['audio_label'].to(self.device)

            # Forward pass
            outputs = self.model(audio, video)
            video_logits = outputs[:, 0]
            audio_logits = outputs[:, 1]
            print(f"Video Pred: {video_logits}")
            print(f"Audio Pred: {audio_logits}")
            print(f"Video Actual: {video_labels}")
            print(f"Audio Actual: {audio_labels}")

            # Loss calculation
            video_loss = self.video_criterion(video_logits, data['video_label']) * data['weight']
            audio_loss = self.audio_criterion(audio_logits, data['audio_label']) * data['weight']

            # Combined loss
            loss = 0.3 * video_loss.mean() + 0.7 * audio_loss.mean()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Logging
            epoch_loss += loss.item()
            if batch_idx % 100 == 0:
                self._log_batch(epoch, batch_idx, video_logits, audio_logits,
                                video_labels, audio_labels)

            if opt.oversample_weight > 1:
                print(f"\nOversampling enabled with weight {opt.oversample_weight} for classes 0 and 1")
                self._log_class_distribution()

        return epoch_loss / len(self.train_loader)

    def _update_weights(self):
        """Update weights based on current validation performance"""
        with torch.no_grad():
            val_results = self.validate(-1)  # Run validation without logging
            class_acc = self._calculate_class_accuracy(val_results)

            # Update weights inversely proportional to accuracy
            new_weights = 1.0 / (class_acc + 0.1)  # Smoothing factor
            self.class_weights = new_weights / new_weights.sum()

    def _calculate_class_accuracy(self, val_results):
        """Calculate per-class accuracy from validation results"""
        preds, labels = val_results
        class_acc = torch.zeros(self.opt.num_classes)
        for c in range(self.opt.num_classes):
            mask = labels == c
            if mask.any():
                class_acc[c] = (preds[mask].argmax(1) == labels[mask]).float().mean()
        return class_acc

    def _log_batch(self, epoch, batch_idx, video_logits, audio_logits,
                   video_labels, audio_labels):
        with torch.no_grad():
            video_probs = torch.sigmoid(video_logits)
            audio_probs = torch.sigmoid(audio_logits)

            # Class-specific metrics
            real_vid_acc = (video_probs[video_labels == 1] > 0.5).float().mean()
            fake_vid_acc = (video_probs[video_labels == 0] < 0.5).float().mean()

            real_aud_acc = (audio_probs[audio_labels == 1] > 0.5).float().mean()
            fake_aud_acc = (audio_probs[audio_labels == 0] < 0.5).float().mean()

            # Logging
            self.writer.add_scalars('train/accuracy', {
                'video_real': real_vid_acc,
                'video_fake': fake_vid_acc,
                'audio_real': real_aud_acc,
                'audio_fake': fake_aud_acc
            }, epoch * len(self.train_loader) + batch_idx)

    def _log_class_distribution(self):
        """Log the class distribution and sampling weights"""
        class_counts = torch.bincount(self.train_loader.dataset.labels)
        weights = self.train_loader.sampler.weights

        print("\nClass Distribution:")
        for i, count in enumerate(class_counts):
            class_name = self.train_loader.dataset.classes[i]
            weight = weights[self.train_loader.dataset.labels == i].mean()
            print(f"{class_name} (class {i}): {count} samples, sampling weight: {weight:.2f}")

    def validate(self, epoch):
        val_acc, val_ap, _, _, _, _ = validate(self.model, self.val_loader)
        self.scheduler.step(val_ap)
        self.writer.add_scalar('val/accuracy', val_acc, epoch)
        self.writer.add_scalar('val/AP', val_ap, epoch)

        if val_ap > self.best_val_ap:
            self.best_val_ap = val_ap
            torch.save(
                self.model.state_dict(),
                os.path.join(self.opt.checkpoints_dir, self.opt.name, 'best_model.pth')
            )

        return val_acc, val_ap

    def train(self):
        for epoch in range(self.opt.niter):
            start_time = time.time()

            # Train phase
            train_loss = self.train_epoch(epoch)

            # Validation phase
            val_acc, val_ap = self.validate(epoch)

            # Epoch logging
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch} completed in {epoch_time:.1f}s - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Val Acc: {val_acc:.4f} - "
                  f"Val AP: {val_ap:.4f}")

            # Early stopping check
            if epoch > 10 and val_ap < 0.6:
                print("Early stopping due to poor validation performance")
                break


if __name__ == '__main__':
    opt = TrainOptions().parse()
    Testdataroot = os.path.join(opt.dataroot, 'test')
    # opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))
    print('  '.join(list(sys.argv)))
    val_opt = get_val_opt()
    Testopt = TestOptions().parse(print_options=False)
    data_loader = create_dataloader(opt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UnifiedModel(device).to(device)
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=opt.lr, betas=(opt.beta1, 0.999))
    trainer = UnifiedTrainer(opt)
    trainer.train()
