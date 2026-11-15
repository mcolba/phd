---
applyTo: "tests/**/*.py"
---
# Testing Guidance

Tests should be written using `pytest` and meet the following requirements:

- *Tolerances*: Use explicit tolerances based on the expected numerical error. Explain unusually large tolerances.
- *Assertion methods*: Use `numpy.testing.assert_allclose` with explicit `rtol` and `atol` for floating-point results.
- *Coverage*: Maintain 100% test coverage for core public functionality and critical numerical kernels.
- *Runtime*: Mark expensive tests as `@pytest.mark.slow`.

Tests can be categorised along three independent axes, each addressing a different question: (i) *level*, how many components are tested together?; (ii) *strategy*, how is correctness verified?; and (iii) *scenario*, what type of input data is used?

## Test Levels

- *Unit tests*: Verify one behavioural unit in isolation through a fast and deterministic execution path.
- *Integration tests*: Verify the interaction between two or more components, such as an end-to-end calibration workflow.

## Test Strategies

- **Contract tests**: Verify that a function behaves according to its specified interface. This includes:

   - Invalid input and error handling.
   - `dtype`, shape, and `NaN` handling.
   - Broadcasting and scalar/vector behaviour.
   - Input mutation guarantees. For example:

      ```python
      x0 = x.copy()
      func(x)
      np.testing.assert_array_equal(x, x0)
      ```

   - Fixed-seed reproducibility.

- **Reference-value tests**: Use fixed inputs and compare the results with precomputed expected outputs. Reference-value strategies, ordered from the strongest to the weakest oracle, include:

  1. *Analytical solution*: Compare the output with a closed-form solution, such as Black--Scholes prices or Greeks.

  2. *Trusted implementation benchmark*: Compare the output with a trusted external implementation, such as QuantLib for pricing or SciPy for numerical routines.

  3. *Challenger model benchmark*: Compare the output with an independent method, such as Fourier versus Monte Carlo pricing or analytical versus finite-difference Greeks.

  4. *Published reference*: For methods taken from peer-reviewed literature, validate against numerical examples from the paper when available. Specify the paper, page, table, or figure number in the test docstring.

  Multiple strategies may be appropriate for the same function. For example, a degenerate case may admit an analytical solution, while nominal and boundary scenarios are validated against a challenger method.

- **Mathematical-property tests**: Verify relationships and constraints implied by theory across representative inputs. Some examples include:

   - *Invariance/equivariance*: permutation invariance (OLS regression), homogeneity of option prices and VaR.

   - *Identities*: put-call parity, $\phi(0) = 1$ for characteristic functions, quantile inversion for continuous distributions.

   - *Bounds and ordering*: positive semidefiniteness of covariance matrices, no-arbitrage bounds, monotonicity, convexity.

   - *Structural properties*: cumulant additivity for independent increments, martingale conditions.

   - *Limiting behaviour*: convergence to a zero-volatility case, a Gaussian limit, or another known limiting model.

   - *Algorithmic properties*: nonincreasing objective values when guaranteed by the optimisation algorithm.

- **Cross-implementation tests**: If multiple backends exist, verify that they produce consistent results within explicit tolerances. Examples include quadrature versus FFT, NumPy versus JAX, analytical versus autodiff Greeks, or CPU versus accelerated implementations.

- **Round-trip tests**: Test serialisation/deserialisation, parameter transformations, and coordinate changes for correct round-trip behaviour.

- **Synthetic recovery and statistical validation**: Generate data from a known model and run the full calibration, fitting, or inference workflow. For noiseless data, test reconstruction and recovery of identifiable parameters. For stochastic data, assess bias, RMSE, interval coverage, or predictive calibration across fixed random streams using uncertainty-aware thresholds. For non-identifiable models, compare observables, likelihoods, predictive distributions, or equivalence classes rather than raw parameters. Public examples include Pyro's [NUTS recovery tests](https://github.com/pyro-ppl/pyro/blob/dev/tests/infer/mcmc/test_nuts.py) and QuantLib's [Heston calibration tests](https://github.com/lballabio/QuantLib/blob/master/test-suite/hestonmodel.cpp).

- **Regression tests**: Preserve minimal cases for previously diagnosed failures.

## Test Scenarios

- *Nominal*: Typical, well-conditioned inputs.
  
- *Boundary*: Valid inputs at or near documented domain limits, including minimum sizes, endpoints, and exact branch thresholds.
  
- *Numerically difficult*: Difficult but valid numerical regimes, including near-singularities, extreme magnitudes, ill-conditioning, cancellation, and near-admissibility boundaries.
  
- *Degenerate*: Cases in which the model simplifies, often allowing comparison with a closed-form solution or another implementation; for example, the Heston model approaches Black--Scholes under an appropriate constant-variance limit.
  
- *Invalid*: Inputs outside the supported domain that should be rejected according to the contract.

## Test Data

- When data generation is complex or time-consuming (e.g., Monte Carlo), store the data-generation script in `scripts/make_test_data/` and save the generated output in `tests/data/`.
- Produce test data independently of the core implementation in `./vol_risk`.
- Store enough metadata to reproduce generated data, including random seeds, model parameters, numerical tolerances, and external library versions when relevant.
