from typing import  List
import openunmix
from .interface import BaseSeparationModel # noqa


class OpenUnmixModel(BaseSeparationModel):
    """
    Адаптер для модели Open-Unmix.
    Поддерживает разделение на: vocals, drums, bass, other.
    """

    def __init__(self, sources: List[str] = None, model_path: str = "" , model_type='umxl', device: str = "cpu", segment_length=5,
                 sample_rate: int = 44100, segment_callback=None, nb_channels=2, hidden_size=1024, nb_layers=3, unidirectional=False):
        self.sources = sources if sources is not None else ["vocals", "drums", "bass", "other"]
        if model_type == 'umxse' and sources is None:
            self.sources = ["speech", "noise"]
        self.nb_channels = nb_channels
        self.hidden_size = hidden_size
        self.nb_layers = nb_layers
        self.unidirectional = unidirectional
        self.model_path = model_path
        self.model_type = model_type
        super().__init__("openunmix", sources, device, segment_length, sample_rate, segment_callback)

    def _load_model(self):
        match self.model_type:
            case 'umxl':
                self.model = openunmix.umxl(self.sources)
            case 'umx':
                self.model = openunmix.umx(self.sources)
            case 'umxhq':
                self.model = openunmix.umxhq(self.sources)
            case 'umxse':
                self.model = openunmix.umxse(self.sources)
            case _:
                pass
        self.model = self.model.to(self.device)
        self.model.eval()


    def get_supported_sources(self) -> List[str]:
        """Возвращает список поддерживаемых дорожек"""
        return self.sources

    def _move_to_device(self, device: str):
        """Перемещает модель на заданное устройство"""
        pass