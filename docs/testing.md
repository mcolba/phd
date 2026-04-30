---
applyTo: "tests/**/*.py"
---
# Testing Guidance

Use `pytest` with focused unit tests for all numerical kernels and integration tests for wrapper behavior.

## Unit Tests

- **Determinism**: Keep unit tests deterministic by fixing random generator seeds and mocking external dependencies such as databases, APIs, file systems, and third-party services.

- **Scope**: Cover invalid inputs, domain boundaries, and regression-sensitive edge cases. Test numerical stability for extreme but valid inputs, including parameter combinations near singularities or discontinuities.

- **Example-based tests**: Validate complex or numerically sensitive behavior against (a) an independent challenger implementation, (b) a comparable benchmark method, or (c) a trusted external reference. For example:
  - Compare pricing functions against a third-party reference library such as QuantLib or against numerical examples from published references.
  - Compare numerical algorithms against an alternative method, for example Fourier-based versus Monte Carlo pricing.

- **Property-based tests**: Use property-based tests when exact expected values are difficult to specify or when the input space is too large to cover programmatically.

- **Assertion methods**: Use `numpy.testing` utilities such as `assert_allclose` for numerical assertions.

- **Coverage**: Maintain 100% test coverage for core functionality. Exempt helper functions such as plotting utilities unless they implement business-critical behavior.

# Integration Tests
- Add integration tests for wrapper behavior.

# Test data

- When data generation is complex or time-consuming, store a data-generation script in `scripts/make_test_data/` and save the output in `tests/data`.
- Produce test data independently from the core implementation in `./vol_risk`.

