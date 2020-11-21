import pytest
import torch

from scattering_compositional_learner import ScatteringCompositionalLearner


@pytest.mark.parametrize('image_size', [80, 160])
def test_forward(image_size):
    x = torch.rand(4, 16, image_size, image_size)
    scl = ScatteringCompositionalLearner(image_size=image_size)
    logits = scl(x)
    assert logits.shape == (4, 8)
