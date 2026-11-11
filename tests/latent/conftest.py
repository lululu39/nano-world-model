import pytest
import torch


@pytest.fixture(autouse=True)
def reproducible_cpu_smoke():
    torch.set_num_threads(2)
    torch.manual_seed(7)
