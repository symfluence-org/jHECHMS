"""The calibration slice must work on symfluence releases without the shared helper.

``InMemoryModelWorker.get_calibration_slice()`` landed in symfluence after
the 0.9.2 release on PyPI. This package calls it from the differentiable
loss path, where an AttributeError would be swallowed by the surrounding
``except Exception`` in compute_gradient / evaluate_with_gradient — so an
older core would not raise, it would quietly stop restricting the loss to
the calibration period. That is the bug this whole change exists to fix.

These tests pin that the fallback produces the same window as the shared
implementation, and degrades to None rather than to "score everything".
"""

import pandas as pd
import pytest

import symfluence.optimization.workers.inmemory_worker as _iw

from jhechms.calibration.worker import _calibration_slice

WARMUP = 365
PERIOD = "2003-06-01, 2003-08-31"


class _Worker:
    """Only the attributes the slice logic reads."""

    def __init__(self, idx, period=PERIOD, warmup=WARMUP):
        self._time_index = idx
        self._cfg_map = {"CALIBRATION_PERIOD": period}
        self._warmup = warmup

    def _cfg(self, key, default=None):
        return self._cfg_map.get(key, default)

    def warmup_steps(self):
        return self._warmup

    @property
    def warmup_days(self):
        return self._warmup


@pytest.fixture
def idx():
    return pd.date_range("2002-01-01", periods=365 * 3, freq="D")


@pytest.fixture
def without_shared_helper(monkeypatch):
    """Simulate a symfluence predating get_calibration_slice."""
    if hasattr(_iw.InMemoryModelWorker, "get_calibration_slice"):
        monkeypatch.delattr(_iw.InMemoryModelWorker, "get_calibration_slice")


def _with_shared(worker):
    """Bind the real shared implementation, when this symfluence has one."""
    shared = getattr(_iw.InMemoryModelWorker, "get_calibration_slice", None)
    if shared is not None:
        worker.get_calibration_slice = shared.__get__(worker)
    return worker


def test_fallback_selects_the_calibration_window(idx, without_shared_helper):
    start, end = _calibration_slice(_Worker(idx))
    after_warmup = idx[WARMUP:]
    assert after_warmup[start] == pd.Timestamp("2003-06-01")
    assert after_warmup[end - 1] == pd.Timestamp("2003-08-31")
    assert end - start == 92  # Jun 1 -> Aug 31 inclusive


def test_fallback_matches_the_shared_implementation(idx):
    """Both paths must agree, or upgrading symfluence would move results."""
    shared = getattr(_iw.InMemoryModelWorker, "get_calibration_slice", None)
    if shared is None:
        pytest.skip("this symfluence has no shared implementation to compare against")
    assert _calibration_slice(_with_shared(_Worker(idx))) == _calibration_slice(_Worker(idx))


@pytest.mark.parametrize("period", ["", "not-a-date", "2003-06-01"])
def test_unusable_period_returns_none(idx, period, without_shared_helper):
    assert _calibration_slice(_Worker(idx, period=period)) is None


def test_non_overlapping_period_returns_none(idx, without_shared_helper):
    assert _calibration_slice(_Worker(idx, period="1999-01-01, 1999-02-01")) is None


def test_missing_time_index_returns_none(without_shared_helper):
    assert _calibration_slice(_Worker(None)) is None
