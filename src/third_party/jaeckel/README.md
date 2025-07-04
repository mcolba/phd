# LetsBeRational Python Extension

Simplified Python bindings for the LetsBeRational library by Peter Jäckel.

## Overview

The LetsBeRational library implements advanced option pricing algorithms with high numerical precision. This Python extension makes these algorithms accessible from Python while maintaining the performance of the underlying C++ implementation.

## Prerequisites

- Python 3.11 or higher
- setuptools and build tools
- MSVC compiler
- SWIG 4.0 or higher (for regenerating wrappers)

## Build Instructions

### 1. Generate SWIG Wrappers (if interface file changes)

As simplified version of the SWIG interface file, [`LetsBeRationalSimplified.i`](LetsBeRationalSimplified.i), was used to generate the wrapper files in [`letsberational`](letsberational). Generally, there  is no need to regenerate the wrappers, unless changes are made to the interface file or to [`src/lets_be_rational.h`](src/lets_be_rational.h).

To regenerate the wrappers, run the batch script (it requires SWIG):

```cmd
cd src\third_party\jaeckel
generate_wrappers.bat

REM Generating SWIG wrappers for LetsBeRational...
REM SWIG wrapper generation completed successfully!
REM Generated files:
REM   - letsberational\LetsBeRational_wrap.cpp
REM   - letsberational\letsberational.py
REM Press any key to continue . . .
```

### 2. Build and Install

#### Create Wheel Package
For development and testing, build the extension in-place:

```cmd
cd cpp\third_party\jaeckel
python -m build
```

This creates wheel files in the `dist` directory.

#### Install from Wheel

```cmd
cd src\third_party\jaeckel
pip install dist/letsberational-1.0.0-cp313-cp313-win_amd64.whl --force-reinstall
```
**Note:** Replace the wheel filename with the actual generated file name, which varies by Python version and platform.

#### Install from Source
```cmd
cd src\third_party\jaeckel
python setup.py install
```

#### Development Installation
For development, install in editable mode:

```cmd
cd src\third_party\jaeckel
pip install -e .
```

## Usage

After installation, import and use the library:

```python
import letsberational

from letsberational import black_price, implied_black_vol

forward = 100.0  # forward price
strike = 105.0  # strike price (OTM call)
original_sigma = 0.25  # 25% volatility
maturity = 0.5  # 6 months to expiry
option_type = 1.0  # call option

price = black_price(forward, strike, original_sigma, maturity, option_type)
implied_sigma = implied_black_vol(price, forward, strike, maturity, option_type)
```

## Module Structure

```
src/third_party/jaeckel/
├── setup.py                    # Build configuration
├── pyproject.toml              # Project metadata
├── MANIFEST.in                 # Package manifest
├── generate_wrappers.bat       # SWIG wrapper generation script
├── LetsBeRationalSimplified.i  # SWIG interface file
├── README.md                   # This file
├── letsberational/             # Python package directory
│   ├── __init__.py             # Package initialization
│   ├── letsberational.py       # SWIG-generated Python module
│   ├── LetsBeRational_wrap.cpp # SWIG-generated C++ wrapper
│   ...
└── src/                        # C++ source files
    ├── lets_be_rational.cpp    # Core implementation
    ├── lets_be_rational.h      # Core header file
    ...
