import torch
import torch.nn as nn
import numpy as np
from networks.freqnet import freqnet
from networks.sslmodel import SSLModel


class UnifiedModel(nn.Module):
    def __init__(self, device, dataset_stats=None):
        super(UnifiedModel, self).__init__()
        self.device = device
        self.dataset_stats = dataset_stats
        self.video_model = freqnet().to(device)
        self.audio_model = SSLModel(device).to(device)

        self.audio_fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.fc_video = nn.Linear(512, 1)
        self.fc_audio = nn.Linear(512, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Adjusted output biases based on class priors
        if hasattr(self, 'dataset_stats'):
            ff_count = self.dataset_stats['FF']
            other_video_count = sum(self.dataset_stats[c] for c in ['RR', 'RF', 'FR'])
            video_bias = -np.log(ff_count / other_video_count)

            ff_rf_count = self.dataset_stats['FF'] + self.dataset_stats['RF']
            rr_fr_count = self.dataset_stats['RR'] + self.dataset_stats['FR']
            audio_bias = -np.log(ff_rf_count / rr_fr_count)

            nn.init.constant_(self.fc_video.bias, video_bias)
            nn.init.constant_(self.fc_audio.bias, audio_bias)
        else:
            # Fallback to default values if no dataset stats
            nn.init.constant_(self.fc_video.bias, -np.log(9539 / (435 + 435 + 8539)))
            nn.init.constant_(self.fc_audio.bias, -np.log(9529 / (435 + 8539)))
        # nn.init.constant_(self.fc_video.bias, -np.log(9539 / 435))  # FF vs RR+RF+FR
        # nn.init.constant_(self.fc_audio.bias, -np.log(9529 / 8539))  # FF+RF vs RR+FR

    def forward(self, audio_input, video_input):
        # Audio pathway
        audio_feat, _ = self.audio_model.extract_feat(audio_input.squeeze(-1))
        audio_feat = torch.mean(audio_feat, dim=1)
        audio_feat = self.audio_fc(audio_feat)

        # Video pathway
        video_feat = self.video_model(video_input)

        combined_feat = torch.cat((audio_feat, video_feat), dim=1)
        fused_feat = self.fusion(combined_feat)

        video_out = self.fc_video(fused_feat)
        audio_out = self.fc_audio(fused_feat)
        return torch.cat((video_out, audio_out), dim=1)
