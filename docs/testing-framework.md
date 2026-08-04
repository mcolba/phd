---
applyTo: "tests/**/*.py"
---
# Testing Guidance

Use `pytest` with focused unit tests for numerical kernels and integration tests for wrapper behavior, calibration workflows, and backend interoperability.

## Unit Tests

### Requirements

- **Determinism**: Fix random generator seeds and mock external dependencies such as databases, APIs, remote services, and non-deterministic file-system interactions.
- **Assertion methods**: Use `numpy.testing` utilities such as `assert_allclose` array assertions.
- **Coverage**: Maintain full coverage for core public functionality and critical numerical paths.
- **Tolerances**: Use explicit numerical tolerances. Prefer tight tolerances for algebraic identities and looser tolerances oyherwise.

### Minimum required test Suite

1. **Contract tests**
   - Invalid input rejection.
   - Scalar versus vector input behavior.
   - `dtype`, shape, and finite-value behavior.
   - Boundary cases and exact-limit branches.

2. **Example-based tests**
   - Validate against a trusted external implementation, such as QuantLib for pricing or SciPy for numerical routines. If the full methodology is not available externally, validate a reduced or limiting case that has an trusted external implementation.
   - Compare against a challenger method, such as, Fourier pricing versus Monte Carlo pricing, or analytical Greeks versus finite-difference Greeks. Expensive Monte Carlo simulations should be run once with a fixed seed; store generated reference data in `tests/data/` and the data-generation script in `scripts/make_test_data/`.
   - For methods taken from peer-reviewed literature, validate against numerical examples from the paper (when available). Specify the paper, page, table, or figure number in the test docstring.
   - Prefer small, representative examples with explicit numerical tolerances.

3. **Property-based tests**
   - Use when exact expected values are difficult to specify or when the input space is too large to cover with fixed examples.
   - Examples for option pricing include no-arbitrage bounds, monotonicity and convexity, homogeneity, martingale conditions, put-call parity, asymptotic regimes (e.g., zero vol, gausian limit), cumulant time-additivity, characteristic-function identities.

4. **Stress tests**
   - Exercise difficult but valid numerical regimes, including near-singularities, extreme moneyness, short maturities, high volatilities, and near-admissibility boundaries.
   - Keep valid stress cases separate from invalid or expected-failure cases.
   - Stress tests may use example-based assertions, property assertions, or backend comparisons depending on the available oracle.

5. **Backend-consistency tests**
   - If multiple backends exist, verify that they produce consistent results within explicit tolerances.
   - Examples include quadrature versus FFT, NumPy versus JAX, analytical versus autodiff Greeks, or CPU versus accelerated implementations.
   - Keep backend-specific constraints explicit, especially fixed shapes, dtype behavior, and JIT/compiled-path assumptions.

6. **Optimisation routine tests**
   - Test parameter transforms independently: transforms must map to the valid domain, and inverse or round-trip behavior should be tested when available.
   - Test objective functions independently: objectives must be deterministic, lower at true synthetic parameters than at perturbed parameters.
   - Test weighting schemes, masks, bounds, and parameter constraints explicitly.

## Integration Tests

1. **Wrapper integration tests**
   - Test public wrappers from validated inputs through core numerical kernels.
   - Verify shape normalization, scalar/vector behavior, error propagation.

2. **Optimisation and calibration routine integration tests**
   - Use synthetic recovery tests: generate prices or data from known parameters, calibrate back from fixed initial guesses, and assert parameter recovery, objective improvement, and optimiser diagnostics with explicit tolerances.
   - Mark expensive calibration, multistart, Monte Carlo, or large-surface tests as `@pytest.mark.slow`.

## Test Data

- When data generation is complex or time-consuming, store the data-generation script in `scripts/make_test_data/` and save the generated output in `tests/data/`.
- Produce test data independently from the core implementation in `./vol_risk`.
- Store enough metadata to reproduce generated data, including random seeds, model parameters, numerical tolerances, and external library versions when relevant.