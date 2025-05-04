import numpy as np
import pytest
import torch
from ..models import HTDemucsModel
from ..utils.testing import get_dummy_audio


@pytest.fixture
def model():
    model = HTDemucsModel(["drums", "bass", "other", "vocals"], model_path="../../weights/htdemucs/75fc33f5-1941ce65.th", model_included_in_path=True, device="cuda")
    return model


def test_model_initialization(model):
    assert isinstance(model, HTDemucsModel)
    assert model.get_supported_sources() == ["vocals", "drums", "bass", "other"]

def test_model_to_cpu(model):
    model.to("cpu")
    for param in model.model.parameters():
        assert param.device.type == "cpu"

def test_model_to_cuda(model):
    if torch.cuda.is_available():
        model.to("cuda")
        for param in model.model.parameters():
            assert param.device.type == "cuda"

def test_mono_audio_input(model):
    waveform = get_dummy_audio(seconds=5, channels=1, sr=44100)
    result = model.separate(waveform)
    for source in result:
        assert result[source].shape[0] == 1  # должно быть [1, T]

def test_stereo_audio_input(model):
    waveform = get_dummy_audio(seconds=5, channels=2, sr=44100)
    result = model.separate(waveform)
    for source in result:
        assert result[source].shape[0] == 2  # [2, T]

def test_output_shape(model):
    waveform = get_dummy_audio(seconds=5, channels=2, sr=44100)
    result = model.separate(waveform)
    for source in result:
        assert result[source].shape[-1] == 44100 * 5

def test_long_audio(model):
    waveform = get_dummy_audio(seconds=60, channels=2, sr=44100)
    result = model.separate(waveform)
    for source in result:
        assert result[source].shape[-1] == 44100 * 60

def test_no_nan_in_output(model):
    waveform = get_dummy_audio(seconds=5, channels=2, sr=44100)
    result = model.separate(waveform)
    for source in result:
        assert not np.isnan(result[source]).any()
def test_multiple_calls(model):
    waveform = get_dummy_audio(seconds=5, channels=2, sr=44100)
    result1 = model.separate(waveform)
    result2 = model.separate(waveform)
    for source in result1:
        assert np.allclose(result1[source], result2[source], atol=1e-5)