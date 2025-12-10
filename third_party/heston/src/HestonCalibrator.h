// Declares Gauss–Legendre quadrature state and integrands for the Heston pricer
// and its Jacobian. This header was added on top of the original vendored
// code in third_party/heston/src/HestonCalibrator.cpp to make the integration
// interfaces explicit.

#ifndef HESTON_CORE_H
#define HESTON_CORE_H

// Container for Gauss–Legendre weights / nodes
typedef struct tagGLAW {
    int numgrid;  // # of nodes
    double* u;    // nodes
    double* w;    // weights
} GLAW;

// Global Gauss–Legendre quadrature rule
extern GLAW glaw;

// Global constants used in integrals
extern const double pi;
extern const double Q;

// Integrands for Heston pricer
struct tagMN {
    double M1;
    double N1;
    double M2;
    double N2;
};

// Integrands for Jacobian
struct tagMNJac {
    double pa1s;
    double pa2s;

    double pb1s;
    double pb2s;

    double pc1s;
    double pc2s;

    double prho1s;
    double prho2s;

    double pv01s;
    double pv02s;
};

// Heston pricer integrand
tagMN HesIntMN(double u, double a, double b, double c, double rho, double v0,
               double K, double T, double S, double r);

// Jacobian integrand
tagMNJac HesIntJac(double u, double a, double b, double c, double rho,
                   double v0, double K, double T, double S, double r);

#endif  // HESTON_CORE_H