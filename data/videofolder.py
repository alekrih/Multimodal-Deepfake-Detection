import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import subprocess
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from io import BytesIO


class VideoFolder(Dataset):
    def __init__(self, root, transform=None, frame_limit=6,
                 audio_transform=None, phase='train'):
        """
        Args:
            root (str): Path to the directory containing video files.
            transform (callable, optional): Transform to be applied to each video frame.
            frame_limit (int, optional): Maximum number of frames to extract from each video.
            audio_transform (callable, optional): Transform to be applied to the audio waveform.
        """
        self.root = root
        # print(root)
        self.transform = transform
        self.audio_transform = audio_transform
        self.frame_limit = frame_limit
        self.phase = phase

        self.video_files = self._find_video_files()
        self.classes, self.class_to_idx = self._find_classes()
        self.labels = self._get_labels()
        print(f"\n{phase.upper()} Dataset:")
        print(f"Total samples: {len(self.video_files)}")
        for cls in self.classes:
            count = sum(1 for f in self.video_files if cls in f)
            print(f"{cls}: {count} samples")

    def _find_video_files(self):
        """Recursively find all video files in the root directory and its subdirectories."""
        extensions = ('.mp4', '.avi', '.mov', '.mkv')
        videos = []
        for root_dir, _, files in os.walk(self.root):
            for file in files:
                if file.lower().endswith(extensions):
                    videos.append(os.path.join(root_dir, file))
        return videos

    def _find_classes(self):
        """Find the class folders with enforced consistent numbering."""
        class_order = [
            'RealVideo-RealAudio',
            'RealVideo-FakeAudio',
            'FakeVideo-RealAudio',
            'FakeVideo-FakeAudio'
        ]

        existing_classes = set()
        for video_path in self.video_files:
            class_name = os.path.basename(os.path.dirname(video_path))
            if class_name in class_order:
                existing_classes.add(class_name)

        class_to_idx = {cls: i for i, cls in enumerate(class_order)
                        if cls in existing_classes}
        classes = list(class_to_idx.keys())

        return classes, class_to_idx

    def _get_labels(self):
        """Get numeric labels for all videos."""
        labels = []
        for video_path in self.video_files:
            class_name = os.path.basename(os.path.dirname(video_path))
            labels.append(self.class_to_idx[class_name])
        return torch.tensor(labels)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        try:
            video_path = self.video_files[idx]
            frames = self._load_frames(video_path)
            if frames is None:
                return None

            waveform, _ = self._load_audio(video_path)
            class_name = os.path.basename(os.path.dirname(video_path))

            # Ensure all labels are tensors
            return {
                'video': frames,
                'audio': waveform,
                'label': torch.tensor(self.class_to_idx[class_name], dtype=torch.long),
                'video_label': torch.tensor(1 if 'FakeVideo' in class_name else 0, dtype=torch.float32),
                'audio_label': torch.tensor(1 if 'FakeAudio' in class_name else 0, dtype=torch.float32),
                'path': video_path
            }
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return None

    def _load_frames(self, video_path):
        """Load and transform video frames."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        while len(frames) < self.frame_limit:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)

            if self.transform:
                frame = self.transform(frame)

            frames.append(frame)
            frame_count += 1

        cap.release()

        # Pad if needed
        if len(frames) < self.frame_limit:
            padding = [torch.zeros_like(frames[0]) for _ in range(self.frame_limit - len(frames))]
            frames.extend(padding)

        return torch.stack(frames)

    def _load_audio(self, video_path, target_length=80000):
        try:
            command = [
                'ffmpeg',
                '-i', video_path,
                '-f', 'wav',
                '-ar', '16000',
                '-ac', '1',
                '-'
            ]
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10 ** 8
            )
            raw_audio = process.communicate()[0]
            audio_array = np.frombuffer(raw_audio, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0  # Normalize

            waveform = torch.from_numpy(audio_array).float().unsqueeze(0)

            # Handle padding/truncation
            if waveform.size(1) < target_length:
                pad = torch.zeros(1, target_length - waveform.size(1))
                waveform = torch.cat([waveform, pad], dim=1)
            else:
                waveform = waveform[:, :target_length]

            if self.audio_transform:
                waveform = self.audio_transform(waveform)

            return waveform, 16000

        except Exception as e:
            print(f"Audio extraction failed for {video_path}: {e}")
            return torch.zeros(1, target_length), 16000

    def collate_fn(self, batch):
        """Custom collate to handle None samples."""
        batch = [b for b in batch if b is not None]
        return torch.utils.data.dataloader.default_collate(batch)