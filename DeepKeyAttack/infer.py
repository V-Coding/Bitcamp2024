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
        return librosa.feature.melspectrogram(y=samples, n_mels=64, win_length=1024, hop_length=225)

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = sorted(os.listdir(self.data_dir))
        print(self.file_list)
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        waveform, _ = librosa.load(os.path.join(self.data_dir, self.file_list[idx]),
                                   sr=None,
                                   duration=1.0,
                                   mono=True)

        label = self.file_list[idx].split("_")[0]  # Assuming the file name is 'label_otherInfo.wav'
        print(self.file_list[idx])
        if self.transform:
            waveform = self.transform(waveform)
        print(waveform.shape)

        return waveform

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


def load_model(model_path):
    model = CoAtNet()
    save_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(save_dict)
    model.eval()
    return model

def number_to_letter(number):
    if isinstance(number, int) and number >= 0:
        if number < 10:
            return str(number)
        elif number >= 10 and number <= 35:
            return chr(number - 10 + ord('A'))
    return None

def predict(model_path="DeepKeyAttack/Models/model.pt", audio_path="DeepKeyAttack/Keystrokes/"):
    model = load_model(model_path)
    transform = Compose([ToMelSpectrogram(), ToTensor()])
    dataset = TestDataset(audio_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    predictions = []

    # model = model.cuda()
    for batch in data_loader:
        # batch = batch.cuda()
        outputs = model(batch)
        _, predicted = torch.max(outputs.data, 1)  # change if multi-label classification

        print(number_to_letter(predicted.cpu()[0].item()))

        predictions.append(predicted.item())
    keystrokeString = ""
    for i in predictions:
        keystrokeString += number_to_letter(i)
    return keystrokeString

# def main():
#     audio_paths = ["audio1.wav", "audio2.wav", "audio3.wav"]  # replace with actual paths
#     # actual paths
#     audio_paths = [[f"{AUDIO_DIR}/{j}.wav" for j in range(10)]]
#     for i in list(range(26)):
#         audio_paths.append(f"{AUDIO_DIR}/{chr(ord('A') + i)}.wav")
    
#     predictions = predict(audio_paths)
#     print(predictions)

# if __name__ == "__main__":
#     main()