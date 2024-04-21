import librosa
import numpy as np
import soundfile as sf
import torch
import matplotlib.pyplot as plt
import noisereduce as nr
import os


class AudioFileParser():
    def __init__(self, audio_file):
        # self.audio_file = "DeepKeyAttack/UploadedAudio/ThisIsTheFirst.wav"
        self.audio_file = audio_file
        self.audio_data, self.sample_rate = librosa.load(self.audio_file, sr=None)
        self.reduced_noise = nr.reduce_noise(y=self.audio_data, sr=self.sample_rate, stationary=True)


    def isolator(self):
        samples = self.reduced_noise
        signal = samples[1*self.sample_rate:]
        size = 48
        scan = 24
        before = int(2400/4)
        after = int(12000/4)
        prom = 0.06
        threshold = prom/2
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
        prev_end = self.sample_rate*0.1*(-1)
        # '-- isolating keystrokes'
        counter=0
        for i in range(peak_count):
            this_peak = peaks[i]
            timestamp = (this_peak*scan) + size//2
            if timestamp > prev_end + (0.1*self.sample_rate):
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

def clearKeystrokesDir(directory="DeepKeyAttack/Keystrokes/"):
    
    try:
        # List all files in the directory
        files = os.listdir(directory)
        # Iterate through each file
        for file in files:
            # Check if the file is a WAV file
            # Construct the full path to the file
            file_path = os.path.join(directory, file)
            
            # Delete the file
            os.remove(file_path)
            print(f"Deleted: {file}")
    except Exception as e:
        print(f"An error occurred: {e}")

def isolate_num_strokes(audio_file):
    clearKeystrokesDir()
    parser = AudioFileParser(audio_file)
    strokes = parser.isolator()
    ''' 
    # step = 0.0025
    # while not len(strokes) == num_strokes:
    #     strokes = isolator(samples[1*sample_rate:], sample_rate, 48, 24, int(2400/2), int(12000/2), prom, False)
    #     if len(strokes) < num_strokes:
    #         prom -= step
    #     if len(strokes) > num_strokes:
    #         prom += step
    #     if prom <= 0.06:
    #         print('-- not possible for: ', audio_file)
    #         break
    #     step = step*0.99 
    # '''

    # Extract keystroke segments
    for i, stroke in enumerate(strokes):
        keystroke_segment = stroke
        sf.write(f"DeepKeyAttack/Keystrokes/keystroke_{i}.wav", keystroke_segment, parser.sample_rate)

# isolate_num_strokes("DeepKeyAttack/UploadedAudio/ThisIsTheFirst.wav")