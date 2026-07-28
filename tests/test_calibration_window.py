"""Tests that the differentiable loss scores only the calibration window.

The JAX losses used to slice off warmup and then score everything that
remained. With a calibration/evaluation split configured that span covers
both windows, so gradient-based optimizers were trained on the held-out
evaluation period and reported a score computed over a different window
than the Calib_* metrics written at final evaluation.

``cal_slice`` narrows the loss to the calibration period. These tests pin
that it is honoured: a slice covering the whole post-warmup record must be
a no-op, and a narrower slice must actually change the score.
"""

import numpy as np
import pytest

from jhechms.losses import kge_loss, nse_loss
from jhechms.parameters import DEFAULT_PARAMS

WARMUP = 100
N = 600


def _forcing():
    """Synthetic daily forcing with a seasonal cycle and a wet second half."""
    rng = np.random.default_rng(0)
    t = np.arange(N)
    precip = rng.gamma(0.6, 4.0, N)
    # Make the back half wetter so the two windows are genuinely different.
    precip[N // 2:] *= 3.0
    temp = 10.0 + 12.0 * np.sin(t * 2 * np.pi / 365.0)
    pet = np.clip(2.5 + 2.0 * np.sin(t * 2 * np.pi / 365.0), 0.01, None)
    obs = np.clip(0.35 * precip + rng.normal(0, 0.2, N), 0.01, None)
    return precip, temp, pet, obs


def _loss(loss_fn, cal_slice):
    precip, temp, pet, obs = _forcing()
    return float(loss_fn(
        dict(DEFAULT_PARAMS), precip, temp, pet, obs,
        WARMUP, use_jax=False, cal_slice=cal_slice,
    ))



@pytest.mark.parametrize("loss_fn", [kge_loss, nse_loss])
def test_full_span_slice_is_a_noop(loss_fn):
    """A slice covering the whole post-warmup record must not change the loss."""
    unsliced = _loss(loss_fn, None)
    full = _loss(loss_fn, (0, N - WARMUP))
    assert full == pytest.approx(unsliced, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("loss_fn", [kge_loss, nse_loss])
def test_narrower_slice_changes_the_score(loss_fn):
    """Restricting to a sub-window must actually be applied.

    Without this, the parameter is accepted and silently ignored — which is
    exactly the failure being guarded against.
    """
    unsliced = _loss(loss_fn, None)
    half = _loss(loss_fn, (0, (N - WARMUP) // 2))
    assert not np.isnan(half)
    assert abs(half - unsliced) > 1e-6


@pytest.mark.parametrize("loss_fn", [kge_loss, nse_loss])
def test_disjoint_windows_score_differently(loss_fn):
    """The two halves of the record must not collapse to the same number."""
    span = N - WARMUP
    first = _loss(loss_fn, (0, span // 2))
    second = _loss(loss_fn, (span // 2, span))
    assert abs(first - second) > 1e-6
