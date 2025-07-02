"""Setup script for building the LetsBeRational Python extension.

This script builds a simplified Python extension from C++ source and SWIG-generated files.
The extension provides Python bindings for the LetsBeRational library's core option pricing functions.

Usage:
    python setup.py build_ext --inplace  # Build extension in-place for development
    python setup.py build                # Build extension in build directory
    python setup.py bdist_wheel          # Create wheel package
    python setup.py install              # Install the package

Note:
    Run generate_wrappers.bat first to generate the SWIG wrapper files.
"""

import os
from pathlib import Path
from setuptools import setup, Extension, find_packages
import platform

# Version configuration
VERSION = "1.0.0"  # Update this to match your versioning scheme

# Source files for the extension
SOURCE_FILES = [
    "letsberational/LetsBeRational_wrap.cpp",  # SWIG-generated wrapper
    "src/lets_be_rational.cpp",  # Core library implementation
    "src/erf_cody.cpp",  # Error function implementation
    "src/normaldistribution.cpp",  # Normal distribution utilities
    "src/rationalcubic.cpp",  # Rational cubic interpolation
]

# Preprocessor macro definitions
DEFINE_MACROS = [
    ("NO_XL_API", "1"),  # Disable Excel-specific API
    ("DLL_EXPORT", ""),  # Handle Windows DLL exports (defined as empty)
    ("NDEBUG", "1"),  # Release mode (from original -DNDEBUG)
]

# Platform-specific compiler arguments - mimiking the original Makefile setup
if platform.system() == "Windows":
    # MSVC equivalents of original g++ flags only
    COMPILE_ARGS = [
        "/O2",  # Equivalent to -O3 (optimize for speed)
        "/fp:fast",  # Equivalent to -Ofast/-ffast-math
        "/utf-8",  # Equivalent to -finput-charset=UTF-8 (for Greek characters)
    ]
    LINK_ARGS = []  # No special linker args in original
else:
    msg = "Unsupported platform."
    raise RuntimeError(msg)


# Define the extension module
letsberational_module = Extension(
    name="letsberational._letsberational",
    sources=SOURCE_FILES,
    include_dirs=["src"],
    language="c++",
    define_macros=DEFINE_MACROS,
    extra_compile_args=COMPILE_ARGS,
    extra_link_args=LINK_ARGS,
)

setup(
    name="letsberational",
    version=VERSION,
    description="Simplified Python bindings for LetsBeRational option pricing library",
    # Package configuration
    packages=find_packages(include=["letsberational"]),
    package_dir={"letsberational": "letsberational"},
    ext_modules=[letsberational_module],  # Extension object
    # Requirements
    python_requires=">=3.11",
    install_requires=[],
    # Development requirements
    extras_require={
        "dev": [],
        "test": [],
    },
    # Build configuration
    zip_safe=False,
    include_package_data=True,
)
