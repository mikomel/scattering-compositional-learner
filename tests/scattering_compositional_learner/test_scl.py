import pytest
import torch
import torch.nn.functional as F
import torchtest

from scattering_compositional_learner import ScatteringCompositionalLearner


@pytest.mark.parametrize('image_size', [80, 160])
def test_forward(image_size):
    x = torch.rand(4, 16, image_size, image_size)
    y = torch.randint(8, (4,), dtype=torch.long)
    scl = ScatteringCompositionalLearner(image_size=image_size)
    optimiser = torch.optim.Adam(scl.parameters())
    torchtest.test_suite(
        model=scl,
        loss_fn=F.cross_entropy,
        optim=optimiser,
        batch=[x, y],
        test_inf_vals=True,
        test_nan_vals=True,
        test_vars_change=True
    )
