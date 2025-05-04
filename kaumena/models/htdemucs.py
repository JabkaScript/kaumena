import torch
from typing import List, Union
from pathlib import Path
from ._htdemucs.htdemucs import HTDemucs # noqa
from .interface import BaseSeparationModel # noqa


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
        super().__init__(model_name="htdemucs", sources=sources, device=device, segment_length=segment_length, sample_rate=sample_rate, segment_callback=segment_callback)

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


    def get_supported_sources(self) -> List[str]:
        return ["vocals", "drums", "bass", "other"]

    def _move_to_device(self, device: str):
        self.device = device
        if self.model is not None:
            self.model.to(device)