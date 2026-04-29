# Architecture Guidance

## Core Principles

- Define model, calibration, and risk APIs through [vol_risk/protocols.py](../vol_risk/protocols.py).
- Keep core numerical kernels pure, deterministic, vectorized, and side-effect free.
- Keep wrappers thin: validation, shape normalization, logging, and orchestration only.
- Add or update tests for every behavior change.

## Design Patterns

- Use a **layered architecture** with clear separation of concerns:
  - *Functional core*: write pure, single-purpose, vectorized functions.
  - *Wrapper/interface layer*: perform input validation, shape and dtype normalization, parameter transforms, and logging. Delegate calculations to the core functions.
  - *Adapter layer*: convert external objects into repository conventions.

  ```python
    @dataclass(frozen=True)
    class MarketData:
        spot: np.ndarray
        ...

    def _price(spot: np.ndarray, ...) -> np.ndarray:
        ...

    def price(x: MarketData) -> np.ndarray:
        # Validate and normalize shapes/dtypes, then delegate.
        return _price(spot=spot, ...)
  ```

- **Prefer composition over inheritance.** Inject the callable object (function or class) as a parameter or a class attribute:
  - If you expect object creation to be customizable, inject the callable through `__init__()` and store it as an instance attribute.

    ```python
    class JSONDecoder(object):
        ...
        def __init__(self, ... parse_float=None, ...):
            ...
            self.parse_float = parse_float or float
            ...
    ```

  - If customization is expected to be extremely rare, use a class attribute.
