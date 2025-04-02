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

        self.audio_fc = nn.Linear(1024, 1024)
        self.fusion_fc = nn.Linear(1536, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 2)

        self.dropout = nn.Dropout(0.5)
        self.fusion_bn = nn.BatchNorm1d(1024)
        self.fc1_bn = nn.BatchNorm1d(512)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def forward(self, audio_input, video_input):
        audio_feat, _ = self.audio_model.extract_feat(audio_input.squeeze(-1))
        audio_feat = torch.mean(audio_feat, dim=1)
        audio_feat = F.relu(self.audio_fc(audio_feat))

        # Video pathway
        video_feat = self.video_model(video_input)
        if video_feat.shape[1] != 512:
            raise ValueError(f"Expected video_feat shape [B, 512], but got {video_feat.shape}")

        combined_feat = torch.cat((audio_feat, video_feat), dim=1)
        fused_feat = self.dropout(F.relu(self.fusion_bn(self.fusion_fc(combined_feat))))
        x = self.dropout(F.relu(self.fc1_bn(self.fc1(fused_feat))))
        return self.fc2(x)  # Raw logits for BCEWithLogitsLoss
