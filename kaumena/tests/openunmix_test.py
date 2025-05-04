import pytest
import torch
import numpy as np
from models import OpenUnmixModel


# Тест 1: Проверка успешной инициализации модели
def test_model_initialization():
    model = OpenUnmixModel(sources=["vocals", "drums"])
    assert isinstance(model, OpenUnmixModel)


# Тест 2: Проверка поддерживаемых источников
def test_get_supported_sources():
    model = OpenUnmixModel(sources=["vocals", "drums"])
    sources = model.get_supported_sources()
    assert sources == ["vocals", "drums"]


# Тест 3: Проверка источников по умолчанию
def test_default_sources():
    model = OpenUnmixModel()
    sources = model.get_supported_sources()
    assert sources == ["vocals", "drums", "bass", "other"]


# Тест 4: Проверка umxse с другими источниками
def test_umxse_model_sources():
    model = OpenUnmixModel(model_type='umxse')
    sources = model.get_supported_sources()
    assert sources == ["speech", "noise"]


# Тест 5: Проверка загрузки модели (умолчание)
def test_load_model_default():
    model = OpenUnmixModel()
    assert hasattr(model, 'model')


# Тест 6: Проверка загрузки разных типов моделей
@pytest.mark.parametrize("model_type", ['umxl', 'umx', 'umxhq', 'umxse'])
def test_load_different_model_types(model_type):
    model = OpenUnmixModel(model_type=model_type)
    assert hasattr(model, 'model')


# Тест 7: Проверка переноса модели на устройство
def test_move_to_device():
    model = OpenUnmixModel(device="cpu")
    model.to("cuda")
    assert next(model.model.parameters()).device.type == "cuda" if torch.cuda.is_available() else "cpu"


# Тест 8: Проверка разделения аудио (mock waveform)
def test_separate_method():
    model = OpenUnmixModel(sources=["vocals", "drums"], device="cpu")
    dummy_audio = np.random.rand(2, 44100 * 5).astype(np.float32)  # стерео, 5 секунд
    result = model.separate(dummy_audio)
    assert isinstance(result, dict)
    for source in result:
        assert source in ["vocals", "drums"]
        assert isinstance(result[source], np.ndarray)
        assert result[source].shape[0] == 2  # стерео


# Тест 9: Проверка обработки моно-входа
def test_mono_input_handling():
    model = OpenUnmixModel(device="cpu")
    mono_audio = np.random.rand(44100 * 5).astype(np.float32)  # моно
    result = model.separate(mono_audio)
    assert isinstance(result, dict)
    for source in result:
        assert result[source].shape[0] == 2  # должно стать стерео


# Тест 10: Проверка ошибки при неверном формате входа
def test_invalid_input_format():
    model = OpenUnmixModel(device="cpu")
    invalid_audio = np.random.rand(3, 44100)  # 3 канала — не поддерживается
    with pytest.raises(TypeError):
        model.separate(invalid_audio)
