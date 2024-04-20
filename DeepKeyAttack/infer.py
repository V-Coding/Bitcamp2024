import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
import os
from scipy.io import wavfile
from CoAtNet import CoAtNet as CoAtNet

# assuming model and transform functions are already defined
# and 'MODEL_PATH' contains the path to the trained model 
MODEL_PATH = '/Models/model.pt'
AUDIO_DIR = '../Keystroke-Datasets/MBPWavs'

class ToMelSpectrogram:
    def __call__(self, samples):
        return librosa.feature.melspectrogram(samples, n_mels=64, length=1024, hop_length=225)

class PredictDataset(torch.utils.data.Dataset):
    def __init__(self, audio_paths, transform=None):
        self.audio_paths = audio_paths
        self.transform = transform

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        audio_clip = load_audio_clip(audio_path)
        # audio_clip = wavfile.read(audio_path)   # Modified above line since load_audio_clip was undefined

        if self.transform:
            audio_clip = self.transform(audio_clip)

        return audio_clip


def load_model(path):
    model = CoAtNet()  # should match the architecture of the trained model
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def predict(audio_paths):
    model = load_model(MODEL_PATH)

    transform = transforms.Compose([
        # add transforms that were used while training
        ToMelSpectrogram(), ToTensor()
    ])

    dataset = PredictDataset(audio_paths, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    predictions = []

    for batch in data_loader:
        batch = batch.cuda()
        outputs = model(batch)
        _, predicted = torch.max(outputs.data, 1)  # change if multi-label classification
        predictions.append(predicted.item())

    return predictions

def main():
    audio_paths = ["audio1.wav", "audio2.wav", "audio3.wav"]  # replace with actual paths
    # actual paths
    audio_paths = [[f"{AUDIO_DIR}/{j}.wav" for j in range(10)]]
    for i in list(range(26)):
        audio_paths.append(f"{AUDIO_DIR}/{chr(ord('A') + i)}.wav")
    
    predictions = predict(audio_paths)
    print(predictions)

if __name__ == "__main__":
    main()
