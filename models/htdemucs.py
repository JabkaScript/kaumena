import numpy as np
import torch
from typing import Dict, List, Union
from pathlib import Path
from ._htdemucs.htdemucs import HTDemucs # noqa
from models.interface import BaseSeparationModel


class HTDemucsModel(BaseSeparationModel):
    def __init__(
        self,
        sources: List[str] = None,
        model_path: Union[str, Path] = "",
        model_included_in_path = False,
        device: str = "cpu",
        segment_length = 5,
        sample_rate: int = 44100,
        segment_callback=None,
        *args, **kwargs
    ):
        if sources is None:
            sources = ["vocals", "drums", "bass", "other"]
        self.sources = sources
        self.model_path = Path(model_path)
        self.device = device
        self.model_included_in_path = model_included_in_path
        self.args = args
        self.kwargs = kwargs
        super().__init__(model_name="htdemucs", device=device, segment_length=segment_length, sample_rate=sample_rate, segment_callback=segment_callback)

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        state = torch.load(self.model_path, map_location="cpu", weights_only=False)

        if self.model_included_in_path:
            klass = state["klass"]
            args = state["args"]
            kwargs = state["kwargs"]
            self.model = klass(*args, **kwargs)
        else:
            self.model = HTDemucs(sources=self.sources, *self.args, **self.kwargs)
        self.model.load_state_dict(state["state"])
        self.model.to(self.device).eval()

    def separate(self, waveform: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Разделение аудио на источники

        Args:
            waveform (np.ndarray): Входной сигнал формы [C, T] или [T]

        Returns:
            dict: {"source": np.ndarray}, где каждый массив формы [C, T_total]
        """
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]  # [1, T] → моно

        C, T = waveform.shape
        sr = self.sample_rate
        segment_samples = int(self.segment_length * sr)
        num_segments = (T + segment_samples - 1) // segment_samples
        results = {source: np.zeros((C, 0), dtype=np.float32) for source in self.sources}

        for i in range(num_segments):
            if self.segment_callback is not None:
                self.segment_callback(i, num_segments - 1)
            start = i * segment_samples
            end = min(start + segment_samples, T)
            segment = waveform[:, start:end]

            out_segment = self.model(
                torch.tensor(segment).unsqueeze(0).to(self.device)
            )

            out_segment_np = out_segment.cpu().detach().numpy()[0]
            for idx, source in enumerate(self.sources):
                results[source] = np.concatenate(
                    [results[source], out_segment_np[idx]], axis=-1
                )

        return results

    def get_supported_sources(self) -> List[str]:
        return ["vocals", "drums", "bass", "other"]

    def _move_to_device(self, device: str):
        self.device = device
        if self.model is not None:
            self.model.to(device)