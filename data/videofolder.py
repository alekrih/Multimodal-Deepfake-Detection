import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from io import BytesIO


class VideoFolder(Dataset):
    def __init__(self, root, transform=None, frame_limit=None, audio_transform=None):
        """
        Args:
            root (str): Path to the directory containing video files.
            transform (callable, optional): Transform to be applied to each video frame.
            frame_limit (int, optional): Maximum number of frames to extract from each video.
            audio_transform (callable, optional): Transform to be applied to the audio waveform.
        """
        self.root = root
        self.transform = transform
        self.audio_transform = audio_transform
        self.frame_limit = frame_limit
        self.video_files = self._find_video_files(root)
        self.classes, self.class_to_idx = self._find_classes(root)

    def _find_video_files(self, root):
        """Recursively find all video files in the root directory and its subdirectories."""
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        video_files = []
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.lower().endswith(video_extensions):
                    video_files.append(os.path.join(dirpath, filename))
        return video_files

    def _find_classes(self, dir):
        """Find the class folders with enforced consistent numbering."""
        enforced_order = [
            'RealVideo-RealAudio',
            'RealVideo-FakeAudio',
            'FakeVideo-RealAudio',
            'FakeVideo-FakeAudio'
        ]
        existing_classes = set()
        for root, dirs, files in os.walk(dir):
            for dir_name in dirs:
                if dir_name in enforced_order:
                    existing_classes.add(dir_name)
        class_to_idx = {}
        for idx, class_name in enumerate(enforced_order):
            if class_name in existing_classes:
                class_to_idx[class_name] = idx
        classes = list(class_to_idx.keys())
        print(f"Found classes: {classes}")
        return classes, class_to_idx

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):
        video_path = self.video_files[index].replace('\\', '/')
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")
            frames = []
            frame_count = 0
            max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if max_frames == self.frame_limit:
                increment = 1
            else:
                increment = int(max_frames / self.frame_limit)
            while True:
                ret, frame = cap.read()
                if not ret or (self.frame_limit and len(frames) >= self.frame_limit):
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
                frame_count += increment
            cap.release()
            # print(f"{video_path}: {frame_count}")
            waveform, sample_rate = self._extract_audio(video_path)
            if self.audio_transform:
                waveform = self.audio_transform(waveform)
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return None
        if len(frames) == 0:
            print(f"Warning: No frames loaded for video: {video_path}")
            return None
        # Padding
        fixed_num_frames = 6
        if len(frames) < fixed_num_frames:
            padding = [torch.zeros_like(frames[0]) for _ in range(fixed_num_frames - len(frames))]
            frames.extend(padding)
        elif len(frames) > fixed_num_frames:
            frames = frames[:fixed_num_frames]
        frames = torch.stack(frames)
        folder_name = os.path.basename(os.path.dirname(video_path))
        if folder_name not in self.class_to_idx:
            print(f"Warning: Folder name '{folder_name}' not found in class_to_idx. Skipping video: {video_path}")
            return None

        label = self.class_to_idx[folder_name]
        # print(f"Frames shape: {frames.shape}")
        # print(f"Waveform shape: {waveform.shape}")
        return {'video': frames, 'audio': waveform, 'label': label}

    def _extract_audio(self, video_path, fixed_length=80000):
        """
        Extract audio from the video file using pydub and return it as a waveform tensor (in memory).

        Args:
            video_path (str): Path to the video file.
            fixed_length (int): Desired length of the waveform after padding/truncation.

        Returns:
            waveform (torch.Tensor): Audio waveform as a tensor (shape: [1, T]).
            sample_rate (int): Sampling rate of the audio.
        """
        audio_segment = AudioSegment.from_file(video_path)

        with BytesIO() as audio_bytes:
            audio_segment.export(
                audio_bytes,
                format='wav',
                codec='pcm_s16le',
                parameters=['-ar', '16000']
            )
            audio_bytes.seek(0)

            audio_segment = AudioSegment.from_file(audio_bytes, format='wav')

        waveform = np.array(audio_segment.get_array_of_samples())
        sample_rate = audio_segment.frame_rate
        waveform = waveform.astype(np.float32) / 32768.0

        if audio_segment.channels > 1:
            waveform = waveform.reshape(-1, audio_segment.channels).mean(axis=1)

        # Pad/truncate
        if len(waveform) < fixed_length:
            padding = np.zeros(fixed_length - len(waveform), dtype=np.float32)
            waveform = np.concatenate([waveform, padding])
        elif len(waveform) > fixed_length:
            waveform = waveform[:fixed_length]

        return torch.from_numpy(waveform).unsqueeze(0), sample_rate
