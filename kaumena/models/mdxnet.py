import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from .interface import BaseSeparationModel
from ._mdxnet import  Mixer
from omegaconf import OmegaConf


class MDXNetModel(BaseSeparationModel):
    """
    Адаптер для модели MDX-Net с поддержкой Hydra/OmegaConf.
    Использует отдельные ConvTDFNet + Mixer.
    """

    def __init__(
        self,
        sources: List[str] = None,
        config_paths: Optional[Dict[str, Union[str, Path]]] = None,
        ckpt_paths: Optional[Dict[str, Union[str, Path]]] = None,
        device: str = "auto",
        segment_length: int = 5,
        sample_rate: int = 44100,
        segment_callback=None,
        *args, **kwargs
    ):
        self.sources = sources or ["vocals", "drums", "bass", "other"]
        self.config_paths = {k: Path(v) for k, v in config_paths.items()}
        self.ckpt_paths = {k: Path(v) if v else None for k, v in ckpt_paths.items()}
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device

        super().__init__(
            model_name="mdxnet",
            sources=self.sources,
            device=self.device,
            segment_length=segment_length,
            sample_rate=sample_rate,
            segment_callback=segment_callback
        )

    def _load_model(self):
        """Загрузка всех 4-х моделей и Mixer'а"""

        # Проверяем наличие файлов
        required_sources = ["vocals", "drums", "bass", "other"]
        for source in required_sources:
            if not self.config_paths[source].exists():
                raise FileNotFoundError(f"Config для {source} не найден: {self.config_paths[source]}")
            if not self.ckpt_paths[source].exists():
                raise FileNotFoundError(f"Checkpoint для {source} не найден: {self.ckpt_paths[source]}")

        if not self.config_paths["mixer"].exists():
            raise FileNotFoundError(f"Mixer config не найден: {self.config_paths['mixer']}")
        if not self.ckpt_paths["mixer"].exists():
            raise FileNotFoundError(f"Mixer checkpoint не найден: {self.ckpt_paths['mixer']}")

        # Загружаем Mixer
        mixer_config = OmegaConf.load(self.config_paths["mixer"])
        # mixer_ckpt = torch.load(self.ckpt_paths["mixer"], map_location="cpu", weights_only=False)
        configs_copy = self.config_paths.copy()
        ckpt_copy = self.ckpt_paths.copy()
        configs_copy.pop("mixer")
        ckpt_copy.pop("mixer")
        self.model = Mixer(
            separator_configs=configs_copy,
            separator_ckpts=ckpt_copy,
            **{key: mixer_config[key] for key in dict(mixer_config) if key != "_target_"}
        )
        # self.mixer.load_state_dict(mixer_ckpt["state"])
        self.model.to(self.device).eval()

    def separate(self, waveform: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Разделение аудио на источники.

        Args:
            waveform: Входной сигнал формы [2, T], стерео

        Returns:
            Dict[str, np.ndarray]: {"vocals": ..., ...}
        """
        import torch

        audio_tensor = torch.from_numpy(waveform).float().unsqueeze(0).to(self.device)  # [1, 2, T]

        return self.model(audio_tensor)

        # result = mixer_output.squeeze(0).cpu().numpy()  # [S, C, T]
        # return {source: result[i] for i, source in enumerate(self.sources)}

    def get_supported_sources(self) -> List[str]:
        return self.sources

    def _move_to_device(self, device: str):
        self.device = device
        if hasattr(self, "mixer"):
            self.mixer.to(device)