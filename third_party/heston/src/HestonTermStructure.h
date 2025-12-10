#ifndef HESTON_TERM_STRUCTURE_H
#define HESTON_TERM_STRUCTURE_H

// Extension of the Heston implementation of Cui et al. to include
// interest-rate term structure and dividend-adjusted spot prices.

#include <Eigen/Dense>

    // market parameters for term-structure version
    struct mktpara_term
{
    Eigen::ArrayXd T;
    Eigen::ArrayXd K;
    Eigen::ArrayXd r; // interest rate term structure
    Eigen::ArrayXd S_adj;   // Adjusted Spot = S_0^{-q_i*dt_i}

    mktpara_term() {}             // default
    explicit mktpara_term(int n); // parametrized
};

// term-structure Heston pricer / Jacobian / Delta
void fHesTerm(double *p, double *x, int m, int n, void *data);
void JacHesTerm(double *p, double *jac, int m, int n, void *data);
void dHesTerm(double *p, double *delta, int m, int n, void *data, double *q_termS);
void hestonCalibTerm(const mktpara_term &market, double *price, double *par, double *statistics, int n, int m);

#endif  // HESTON_TERM_STRUCTURE_H
