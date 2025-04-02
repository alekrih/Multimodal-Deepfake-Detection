import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from networks.freqnet import freqnet
from networks.sslmodel import SSLModel, getAttenF


class UnifiedModel(nn.Module):
    def __init__(self, args, device):
        super(UnifiedModel, self).__init__()
        self.device = device
        self.video_model = freqnet().to(device)
        self.audio_model = SSLModel(device).to(device)
        # self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fusion_fc = nn.Linear(1536, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.audio_fc = nn.Linear(1024, 1024)

    # def load_networks(self, epoch):
    #     load_filename = 'model_epoch_%s.pth' % epoch
    #     load_path = os.path.join(self.save_dir, load_filename)
    #
    #     print('loading the model from %s' % load_path)
    #     # if you are using PyTorch newer than 0.4 (e.g., built from
    #     # GitHub source), you can remove str() on self.device
    #     state_dict = torch.load(load_path, map_location=self.device)
    #     # if hasattr(state_dict, '_metadata'):
    #     #     del state_dict._metadata
    #
    #     self.model.load_state_dict(state_dict['model'])
    #     #
    #     # if self.isTrain and not self.opt.new_optim:
    #     #     self.optimizer.load_state_dict(state_dict['optimizer'])
    #     #     # move optimizer state to GPU
    #     #     for state in self.optimizer.state.values():
    #     #         for k, v in state.items():
    #     #             if torch.is_tensor(v):
    #     #                 state[k] = v.to(self.device)
    #     #
    #     #     for g in self.optimizer.param_groups:
    #     #         g['lr'] = self.opt.lr

    def forward(self, audio_input, video_input):
        """
        Forward pass for unified model.

        Args:
            audio_input (torch.Tensor): Audio input tensor of shape (B, 1, T).
            video_input (torch.Tensor): Video input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output log probabilities of shape (B, 2).
                          The first column is the video logit (0 for real, 1 for fake).
                          The second column is the audio logit (0 for real, 1 for fake).
        """
        # print("Audio input shape:", audio_input.shape)
        audio_feat, _ = self.audio_model.extract_feat(audio_input.squeeze(-1))
        # print("Audio feature shape after extraction:", audio_feat.shape)
        audio_feat = torch.mean(audio_feat, dim=1)
        # print("Audio feature shape after pooling:", audio_feat.shape)
        audio_feat = self.audio_fc(audio_feat)
        # print("Audio feature shape after fc:", audio_feat.shape)
        audio_feat = F.relu(audio_feat)
        video_feat = self.video_model(video_input)
        # print("Video feature shape after video_model:", video_feat.shape)
        if video_feat.shape[1] != 512:
            raise ValueError(f"Expected video_feat shape [B, 512], but got {video_feat.shape}")
        combined_feat = torch.cat((audio_feat, video_feat), dim=1)
        # print("Combined feature shape:", combined_feat.shape)
        fused_feat = self.fusion_fc(combined_feat)
        fused_feat = F.relu(fused_feat)
        x = self.fc1(fused_feat)
        x = F.relu(x)
        x = self.fc2(x)
        output = self.logsoftmax(x)
        return output
