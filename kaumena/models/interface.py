from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
from ..utils.audio import split_waveform_into_segments


class BaseSeparationModel(ABC):
    """
    Абстрактный базовый класс для всех моделей MSS.
    """

    def __init__(self, model_name: str, sources: List[str], device: str = "cpu", segment_length = 5,
        sample_rate: int = 44100, segment_callback=None, model: nn.Module = None):
        self.model_name = model_name
        self.device = device
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.segment_callback = segment_callback
        self.model = model
        self.sources = sources
        self._load_model()

    @abstractmethod
    def _load_model(self):
        """Загрузка модели"""
        pass

    def separate(self, waveform: np.ndarray) -> Dict[str, np.ndarray]:
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]  # [1, T] → моно
        C, T = waveform.shape
        results = {source: np.zeros((C, 0), dtype=np.float32) for source in self.sources}
        segments = split_waveform_into_segments(waveform, self.sample_rate, self.segment_length)
        num_segments, segment_samples = segments.shape[:2]

        for i in range(num_segments):
            if self.segment_callback is not None:
                self.segment_callback(i, num_segments - 1)
            segment = segments[i]

            out_segment = self.model(
                torch.tensor(segment).unsqueeze(0).to(self.device)
            )

            out_segment_np = out_segment.cpu().detach().numpy()[0]
            for idx, source in enumerate(self.sources):
                results[source] = np.concatenate(
                    [results[source], out_segment_np[idx]], axis=-1
                )

        return results

    @abstractmethod
    def get_supported_sources(self) -> List[str]:
        """Возвращает список поддерживаемых дорожек"""
        pass

    def to(self, device: str):
        """Перемещает модель на устройство (CPU/GPU)"""
        self.device = device
        self._move_to_device(device)

    def _move_to_device(self, device: str):
        """Дополнительная логика перемещения модели"""
        pass
