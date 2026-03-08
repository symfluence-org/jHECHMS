"""Tests for jHECHMS plugin registration."""

import pytest


def test_register_function_exists():
    """jhechms module should have a register() function."""
    import jhechms
    assert hasattr(jhechms, 'register')
    assert callable(jhechms.register)


def test_entry_point_discoverable():
    """The hechms entry point should be discoverable."""
    from importlib.metadata import entry_points
    eps = entry_points(group='symfluence.plugins')
    names = [ep.name for ep in eps]
    assert 'hechms' in names


def test_register_creates_config_adapter():
    """Calling register() should register HECHMS config adapter."""
    import jhechms
    jhechms.register()

    from symfluence.core.registries import R
    assert 'HECHMS' in R.config_adapters
