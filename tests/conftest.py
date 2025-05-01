import pytest
import torch


@pytest.fixture(autouse=True)
def no_grad():
    with torch.no_grad():
        yield
@pytest.fixture(autouse=True)
def set_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    yield