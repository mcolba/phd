---
applyTo: "**/*.py"
---
# Style Guidance

## Python Lenguage Rules
- Use type hints for all functions' inputs and outputs.
- Do not use type hints for variables.
- Avoid mutable global states.
  

## Style and Documentation Rules
- Use concise Google-style docstrings:
  - Modules: describe the contents in no more than three lines.
  - Simple functions/methods: use one-line description.
  - Complex functions/methods: provide short description, `Args:`, and `Returns:` sections. 
- Use limited inline comments, only for complex operations.
- When calling a function with arguments that are not self-explanatory (e.g., boleans), use named arguments in the same order as the function signature.
- For mathematically-heavy code replicating a reference paper, use the same notation for variable names and cite the paper in the docstring.
- Use a single leading underscore (_) for protecting internal module variables and functions.

## Naming Conventions

- Use `spot` for thte spot price.
- Use `fwd` for the forward price.
- Use `strike` for the strike price.
- Use `maturity` for the contract maturity.
- Use `tau` for the time to maturity in year fraction.
- Use `r` for the risk-free zero-coupon rate.
- Use `q` for the dividend yield.
- Use `disc` for the discount factor.
- Use `sigma` for the volatility parameter.
- Use `mu` for the drift parameter.
