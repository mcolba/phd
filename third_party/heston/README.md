Heston Calibration
================================

This directory vendors the original C++ implementation of the Heston model
pricer and calibrator by Yiran Cui, together with a small term‑structure extension and Windows DLL bindings. See the paper Cui et al. (2015), "Full and fast calibration of the Heston stochastic volatility model" for more details on the methodology. 

Dependencies
------------

This implementation relies on a small set of external numeric libraries:

- **LAPACK** or equivalent implementation
  - The following libraries are compatible: 
    - **Intel MKL**: recommended on Intel CPUs.
    - **CLAPACK**: can be installed without a Fortran compiler as explained in the [Easy Windows Build](https://icl.utk.edu/lapack-for-windows/clapack/index.html) guide.

- **LEVMAR**
  - Levenberg-Marquardt optimization algorithm.
  - Download: http://www.ics.forth.gr/~lourakis/levmar/.

- **Eigen**
  - Header-only linear algebra library.
  - Download: https://eigen.tuxfamily.org/.

License
-------

The original author distributes this code under the GNU General Public License (GPL), which applies to any redistribution of binaries or source code. The relevant headers are preserved in `HestonCalibrator.cpp`.



