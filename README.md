Multimodal Deepfake Detection using Frequency Analysis
===============
This repository contains the code to my Masters thesis on "Multimodal Learning for DeepFake Detection using
Frequency Analysis"


## Installation
First, download/clone this repository to your local machine, then download the fairseq zip file from the [SLSforASVspoof GitHub](https://github.com/QiShanZhang/SLSforASVspoof-2021-DF). The fairseq folder is needed for the audio deep-fake detection base method. Unzip the fairseq file and begin running the following commands:
```
conda create -n MMDFD python=3.7
conda activate MMDFD
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
cd fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
pip install --editable ./
cd ..
conda config --append channels conda-forge
conda config --append channels nvidia
conda config --append channels pytorch
conda install tensorboardX OpenCV scipy moviepy pydub scikit-learn torchaudio chardet
conda install pytorch==1.13.1 torchvision==0.14.1
```
You will also need to download the pre-trained wav2vec 2.0 XLS-R model.
This should allow you to start training and testing the code.

## Pre-trained wav2vec 2.0 XLS-R (300M)
Download the XLS-R models from [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec/xlsr)

## Dataset
The dataset used for training and testing this model was the FakeAVCeleb dataset, which can be obtained by filling out the access form [here](https://github.com/DASH-Lab/FakeAVCeleb). When training and testing, ensure the file structure is as follows:
```
dataset
| test
| | videos
| | | FakeVideo-FakeAudio
| | | FakeVideo-RealAudio
| | | RealVideo-FakeAudio
| | | RealVideo-RealAudio
| train
| | FakeVideo-FakeAudio
| | FakeVideo-RealAudio
| | RealVideo-FakeAudio
| | RealVideo-RealAudio
| val
| | FakeVideo-FakeAudio
| | FakeVideo-RealAudio
| | RealVideo-FakeAudio
| | RealVideo-RealAudio
```

## Training
```
python unified_trainer.py --name MMDFD --batch_size 32 --delr_freq 10 --lr 0.001 --niter 85
```
## Testing/Validation
```
python validate.py --model_path ./checkpoints/{path to latest model}
```