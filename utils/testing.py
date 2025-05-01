import numpy as np
import torch


def get_dummy_audio(seconds=5, channels=2, sr=44100):
    length = seconds * sr
    return np.random.randn(channels, length).astype(np.float32)


def get_dummy_tensor(seconds=5, channels=2, sr=44100):
    audio = get_dummy_audio(seconds, channels, sr)
    return torch.tensor(audio)
