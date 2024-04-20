import librosa
import numpy as np
import soundfile as sf
import torch
import matplotlib.pyplot as plt

audio_file = "DeepKeyAttack/UploadedAudio/playingSoccer.wav"
audio_data, sample_rate = librosa.load(audio_file, sr=None)
energy_env = np.abs(audio_data)
thresh = np.percentile(energy_env, 90)

def isolator(signal, sample_rate, size, scan, before, after, threshold, show=False):
    strokes = []
    # -- signal'
    fft = librosa.stft(signal, n_fft=size, hop_length=scan)
    energy = np.abs(np.sum(fft, axis=0)).astype(float)
    # norm = np.linalg.norm(energy)
    # energy = energy/norm
    # -- energy'
    threshed = energy > threshold
    # -- peaks'
    peaks = np.where(threshed == True)[0]
    peak_count = len(peaks)
    prev_end = sample_rate*0.1*(-1)
    # '-- isolating keystrokes'
    counter=0
    for i in range(peak_count):
        this_peak = peaks[i]
        timestamp = (this_peak*scan) + size//2
        if timestamp > prev_end + (0.1*sample_rate):
            keystroke = []
            if(timestamp-before < 0):
                keystroke = signal[0:timestamp+after]
            elif(timestamp+after > len(signal)):
                keystroke = signal[timestamp-before:-1]
            else:
                keystroke = signal[timestamp-before:timestamp+after]
            # sf.write(f"DeepKeyAttack/Processed/{keyName}_{counter}.wav", keystroke, sample_rate)
            counter += 1
            strokes.append(keystroke)
            prev_end = timestamp+after
    return strokes

def isolate_num_strokes(num_strokes):
    samples, sample_rate = librosa.load(audio_file, sr=None)
    strokes = []
    prom = 0.06
    step = 0.0025
    while not len(strokes) == num_strokes:
        strokes = isolator(samples[1*sample_rate:], sample_rate, 48, 24, int(2400/2), int(12000/2), prom, False)
        if len(strokes) < num_strokes:
            prom -= step
        if len(strokes) > num_strokes:
            prom += step
        if prom <= 0:
            print('-- not possible for: ', audio_file)
            break
        step = step*0.99

    # Extract keystroke segments
    print(len(strokes))
    print(strokes[0])
    for i, stroke in enumerate(strokes):
        keystroke_segment = stroke
        sf.write(f"DeepKeyAttack/Keystrokes/keystroke_{i}.wav", keystroke_segment, sample_rate)

isolate_num_strokes(26)