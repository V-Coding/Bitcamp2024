import matplotlib.pyplot as plt
import torch
import numpy as np
import librosa
import pandas as pd
import soundfile as sf

def isolator(signal, sample_rate, size, scan, before, after, threshold, show=False, keyName=""):
    strokes = []
    # -- signal'
    if show:
        plt.figure(figsize=(7, 2))
        librosa.display.waveshow(signal, sr=sample_rate)
    fft = librosa.stft(signal, n_fft=size, hop_length=scan)
    energy = np.abs(np.sum(fft, axis=0)).astype(float)
    # norm = np.linalg.norm(energy)
    # energy = energy/norm
    # -- energy'
    if show:
        plt.figure(figsize=(7, 2))
        librosa.display.waveshow(energy)
    threshed = energy > threshold
    # -- peaks'
    if show:
        plt.figure(figsize=(7, 2))
        librosa.display.waveshow(threshed.astype(float))
    peaks = np.where(threshed == True)[0]
    peak_count = len(peaks)
    prev_end = sample_rate*0.1*(-1)
    # '-- isolating keystrokes'
    counter=0
    for i in range(peak_count):
        this_peak = peaks[i]
        timestamp = (this_peak*scan) + size//2
        if timestamp > prev_end + (0.1*sample_rate):
            keystroke = signal[timestamp-before:timestamp+after]
            sf.write(f"DeepKeyAttack/Processed/{keyName}_{counter}.wav", keystroke, sample_rate)
            counter += 1
            strokes.append(torch.tensor(keystroke)[None, :])
            if show:
                plt.figure(figsize=(7, 2))
                librosa.display.waveshow(keystroke, sr=sample_rate)
            prev_end = timestamp+after
    return strokes

AUDIO_FILE = 'Keystroke-Datasets/MBPWavs/'
keys_s = '1234567890qwertyuiopasdfghjklzxcvbnm'
labels = list(keys_s)
keys = [f"{k}.wav" for k in labels]
data_dict = {'Key':[], 'File':[]}

for i, File in enumerate(keys):
    loc = AUDIO_FILE + File
    samples, sample_rate = librosa.load(loc, sr=None)
    #samples = samples[round(1*sample_rate):]
    strokes = []
    prom = 0.06
    step = 0.005
    while not len(strokes) == 25:
        strokes = isolator(samples[1*sample_rate:], sample_rate, 48, 24, 2400, 12000, prom, False, keyName=File.split(".")[0])
        if len(strokes) < 25:
            prom -= step
        if len(strokes) > 25:
            prom += step
        if prom <= 0:
            print('-- not possible for: ',File)
            break
        step = step*0.99
    label = [labels[i]]*len(strokes)
    print("Label:", label, "\n", "KeyStrokes:", strokes)
    print("\n\n")
    data_dict['Key'] += label
    data_dict['File'] += strokes

df = pd.DataFrame(data_dict)
mapper = {}
counter = 0
for l in df['Key']:
    if not l in mapper:
        mapper[l] = counter
        counter += 1
df.replace({'Key': mapper}, inplace=True)

print(df)