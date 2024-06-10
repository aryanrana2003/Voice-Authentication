import sounddevice as sd
import numpy as np

def record_audio(duration=5, fs=44100):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
    sd.wait()
    print("Recording complete")
    return np.squeeze(audio)

# Example usage
audio_sample = record_audio()