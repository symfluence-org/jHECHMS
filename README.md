# jHECHMS

[![PyPI version](https://img.shields.io/pypi/v/jhechms.svg)](https://pypi.org/project/jhechms/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Dual-backend (JAX + NumPy) implementation of native HEC-HMS hydrological algorithms — usable standalone or as a SYMFLUENCE plugin.

Part of the SYMFLUENCE JAX-native model family — self-contained packages that
run standalone (NumPy fallback, no JAX required) and register automatically with
[SYMFLUENCE](https://github.com/symfluence-org/SYMFLUENCE) when installed alongside it.

## Features

- **Differentiable**: automatic differentiation through the full simulation (JAX)
- **Fast**: JIT compilation via `lax.scan`; `vmap` for ensembles; GPU-capable
- **Dependency-light**: pure-NumPy fallback when JAX is not installed
- **Plugin architecture**: auto-registers with SYMFLUENCE via entry points

## Installation

```bash
pip install jhechms          # NumPy backend
pip install 'jhechms[jax]'    # with JAX (differentiable, JIT)
```

## Quickstart

```python
from jhechms.model import simulate

flow, state = simulate(precip, temp, pet)                    # default parameters
flow, state = simulate(precip, temp, pet, params={"CN": 75}) # override any subset
```

## Gradient-based calibration

The JAX backend makes the full simulation differentiable end-to-end, so model
parameters can be calibrated with gradient descent:

```python
import jax
from jhechms.losses import kge_loss, get_kge_gradient_fn

grad_fn = get_kge_gradient_fn(precip, temp, pet, observed)
value, grads = grad_fn(params)          # dKGE/dparam for every parameter
```

`nse_loss` / `kge_loss` and their gradient factories are JIT-compatible and work
with any `optax` optimizer. Within SYMFLUENCE the same interface powers the
ADAM and L-BFGS calibration options.

## Use with SYMFLUENCE

jhechms registers with [SYMFLUENCE](https://github.com/symfluence-org/SYMFLUENCE)
through the `symfluence.plugins` entry point — installation is the integration:

```bash
pip install symfluence jhechms
```

```yaml
# config.yaml (excerpt)
model:
  hydrological_model: HECHMS
```

SYMFLUENCE then handles forcing preparation, calibration, evaluation, and
benchmarking for the model with no further wiring.

## Model structure

The implementation follows the HEC-HMS Technical Reference Manual
(US Army Corps of Engineers, 2000):

1. **Snow** — ATI-based temperature-index snow model
2. **Loss** — SCS Curve Number continuous method
3. **Transform** — Clark unit hydrograph (linear reservoir)
4. **Baseflow** — linear-reservoir groundwater

14 calibration parameters (`jhechms.parameters.PARAM_BOUNDS`).

## Testing

```bash
pip install -e '.[dev]'
pytest
```

## How to cite

If you use jHECHMS in your research, please cite the SYMFLUENCE companion papers,
which describe the design of the JAX-native model family (registry integration,
differentiability, and the calibration experiments they enable):

> Eythorsson, D., et al. (2026). The registry as social contract: Architectural patterns
> for community hydrological modeling. *Water Resources Research* (submitted).
>
> Eythorsson, D., et al. (2026). From configuration to prediction: Multi-model,
> multi-basin experiments with SYMFLUENCE. *Water Resources Research* (submitted).

Citation metadata for this package is provided in [`CITATION.cff`](CITATION.cff);
a version-specific DOI is minted via Zenodo for each GitHub release.
<!-- After the first Zenodo release, add the concept-DOI badge here. -->

## References

- Feldman, A. D. (2000). Hydrologic Modeling System HEC-HMS: Technical Reference Manual. Report CPD-74B, U.S. Army Corps of Engineers, Hydrologic Engineering Center, Davis, CA.

## License

Apache-2.0. See [LICENSE](LICENSE).
