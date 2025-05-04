import numpy as np
from numpy import ndarray


def split_waveform_into_segments(waveform: np.ndarray, sr: int, segment_length: float = 5.0) -> np.ndarray:
    """
    Разбивает аудиосигнал на сегменты фиксированной длины.

    Args:
        waveform (np.ndarray): Входной сигнал формы [C, T], где
                               C — количество каналов (1 для моно, 2 для стерео),
                               T — количество семплов.
        sr (int): Частота дискретизации (samples per second).
        segment_length (float): Длина каждого сегмента в секундах.

    Returns:
        np.ndarray: Массив с сегментами формы [num_segments, C, segment_samples]
    """
    C, T = waveform.shape
    segment_samples = int(segment_length * sr)
    num_segments = (T + segment_samples - 1) // segment_samples
    segments = np.zeros((num_segments, C, segment_samples), dtype=waveform.dtype)

    for i in range(num_segments):
        start = i * segment_samples
        end = min(start + segment_samples, T)
        segment = waveform[:, start:end]
        segments[i, :, :end - start] = segment

    return segments

def load_file(path: str, sr: float, mono: bool = False, duration: float = None) -> tuple[ndarray, int | float]:
    import librosa
    return librosa.load(path, sr=sr, mono=mono, duration=duration)