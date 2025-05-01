from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np


class BaseSeparationModel(ABC):
    """
    Абстрактный базовый класс для всех моделей MSS.
    """

    def __init__(self, model_name: str, device: str = "cpu", segment_length = 5,
        sample_rate: int = 44100, segment_callback=None):
        self.model_name = model_name
        self.device = device
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.segment_callback = segment_callback
        self._load_model()

    @abstractmethod
    def _load_model(self):
        """Загрузка модели"""
        pass

    @abstractmethod
    def separate(self, waveform: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Разделение аудио на источники.

        Args:
            waveform: Входное аудио без сжатия (в формате np.ndarray)

        Returns:
            Словарь в формате {"vocals": ..., "drums": ..., ...}
        """
        pass

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
