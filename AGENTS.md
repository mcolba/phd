---
applyTo: "**/*.py"
---
## Overview

This file is the primary repository instruction for any AI assistants working in this repo.

Use this file as the high-level overview and, for more details, refer to:

- [docs/architecture.md](docs/architecture.md) for architecture notes.
- [docs/style.md](docs/style.md) for style notes.
- [docs/testing.py](docs/testing.md) for testing notes.


## Prefered Tech Stack
- Numerics: NumPy, SciPy.
- Testing: pytest.
- Formatting: ruff.
- Type cheking: mypy.

## Repo Structure

```
├── vol_risk/          # core Python package
│   └── _lib/          # DLL loaded by the Heston wrapper
│   └── calibration/   # data loading and calibration orchestration 
│   └── models/        # pricing models (Black76, Heston, etc.)
│   └── vol_surface/   # static surface interpolation and extrapolation
│   └── vol_dynamics/  # surface dynamic modeling (GARCH, HAR, etc.)
│   └── risk_engine/   # tail risk quantification and orchestration
│   └── protocols.py   # shared interfaces and data contracts
├── tests/             # unit and integration tests
├── scripts/           # calibration, analysis, and test-data generators
├── data/              # raw data and derived from scripts
├── examples/          # stand alone notebooks and demos
├── docs/              # architecture, style, testing, and decisions
├── cpp/               # C++ sources, bindings, build files
├── third_party/
│   ├── jaeckel/       # letsberational SWIG extension (Jaeckel, 2015)
│   └── heston/        # Heston C++ pricer/calibrator (Cui et al., 2015)
```

## Python Environment
- Use the virtual environment in `.venv\`.
- Run Python code through `.venv\Scripts\python.exe`.
- Install Python packages through `.venv\Scripts\python.exe -m pip install ...`.
- Run tests through `.venv\Scripts\python.exe -m pytest`.
- Run static-type checking `.venv\Scripts\python.exe -m mypy .\vol_risk`
- Runtime dependencies are listed in `requirements.txt`.
- Development dependencies are listed in `requirements-dev.txt`.

## Architecture
See [docs/architecture.md](docs/architecture.md) for core principles and design patterns.

## Coding Style
See [docs/style.md](docs/style.md) for code standards and naming details.

## Testing
See [docs/testing.py](docs/testing.py) for detailed testing guidance.

## Numerical Standards
- Public API: accept `ArrayLike`, validate, broadcast, and returns a normalised scalar/array output.
- Numerical kernel: accepts already-validated arrays and returns arrays (including 0-d arrays).
- Avoid undocumented implicit broadcasting or silent reshaping.
- Prefer numpy ndrarray to dataframens when possible. 

## Error Handling and Logging
- Fail fast on invalid market data, unsupported conventions, or impossible parameter regimes.
- Surface actionable `ValueError`, `TypeError`, or `NotImplementedError` messages.
- Do not bury assumptions inside low-level kernels; make them explicit in the wrapper or contract.