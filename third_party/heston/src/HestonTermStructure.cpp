#include <cmath>
#include <Eigen/Dense>
#include <levmar.h>
#include "HestonCalibrator.h"
#include "HestonTermStructure.h"

mktpara_term::mktpara_term(int n)
{
    (this->T).resize(n);
    (this->K).resize(n);
    (this->r).resize(n);
    (this->S_adj).resize(n);
}

// Heston pricer: (parameter, observation, dim_p, dim_x, arguments)
void fHesTerm(double *p, double *x, int m, int n, void *data)

{
    int l;

    // retrieve market parameters
    struct mktpara_term *dptr;
    dptr = (struct mktpara_term *)data;

    // retrieve model parameters
    double a = p[0];
    double b = p[1];
    double c = p[2];
    double rho = p[3];
    double v0 = p[4];

    // numerical integral settings
    int NumGrids = glaw.numgrid;
    NumGrids = (NumGrids + 1) >> 1;
    double *u = glaw.u;
    double *w = glaw.w;

    for (l = 0; l < n; ++l)
    {
        double S = dptr->S_adj[l];
        double r = dptr->r[l];
        double K = dptr->K[l];
        double T = dptr->T[l];
        double disc = exp(-r * T);
        double tmp = 0.5 * (S - K * disc);
        disc = disc / pi;
        double Y1 = 0.0, Y2 = 0.0;

        for (int j = 0; j < NumGrids; j++)
        {

            tagMN MN = HesIntMN(u[j], a, b, c, rho, v0, K, T, S, r);

            double M1 = MN.M1;
            double N1 = MN.N1;
            double M2 = MN.M2;
            double N2 = MN.N2;

            Y1 += w[j] * (M1 + N1);
            Y2 += w[j] * (M2 + N2);
        }

        double Qv1 = Q * Y1;
        double Qv2 = Q * Y2;
        double pv = tmp + disc * (Qv1 - K * Qv2);
        x[l] = pv;
    }
}

// Jacobian (parameter, observation, dim_p, dim_x, arguments)
void JacHesTerm(double *p, double *jac, int m, int n, void *data)
{

    int l, k;

    // retrieve market parameters
    struct mktpara_term *dptr;
    dptr = (struct mktpara_term *)data;

    // retrieve model parameters
    double a = p[0];
    double b = p[1];
    double c = p[2];
    double rho = p[3];
    double v0 = p[4];

    // numerical integration settings
    int NumGrids = glaw.numgrid;
    NumGrids = (NumGrids + 1) >> 1;
    double *u = glaw.u;
    double *w = glaw.w;

    for (l = k = 0; l < n; ++l)
    {
        double S = dptr->S_adj[l];
        double r = dptr->r[l];
        double K = dptr->K[l];
        double T = dptr->T[l];
        double discpi = exp(-r * T) / pi;
        double pa1 = 0.0, pa2 = 0.0, pb1 = 0.0, pb2 = 0.0, pc1 = 0.0, pc2 = 0.0, prho1 = 0.0, prho2 = 0.0, pv01 = 0.0, pv02 = 0.0;

        // integrate
        for (int j = 0; j < NumGrids; j++)
        {
            tagMNJac jacint = HesIntJac(u[j], a, b, c, rho, v0, K, T, S, r);

            pa1 += w[j] * jacint.pa1s;
            pa2 += w[j] * jacint.pa2s;

            pb1 += w[j] * jacint.pb1s;
            pb2 += w[j] * jacint.pb2s;

            pc1 += w[j] * jacint.pc1s;
            pc2 += w[j] * jacint.pc2s;

            prho1 += w[j] * jacint.prho1s;
            prho2 += w[j] * jacint.prho2s;

            pv01 += w[j] * jacint.pv01s;
            pv02 += w[j] * jacint.pv02s;
        }

        double Qv1 = Q * pa1;
        double Qv2 = Q * pa2;
        jac[k++] = discpi * (Qv1 - K * Qv2);

        Qv1 = Q * pb1;
        Qv2 = Q * pb2;
        jac[k++] = discpi * (Qv1 - K * Qv2);

        Qv1 = Q * pc1;
        Qv2 = Q * pc2;
        jac[k++] = discpi * (Qv1 - K * Qv2);

        Qv1 = Q * prho1;
        Qv2 = Q * prho2;
        jac[k++] = discpi * (Qv1 - K * Qv2);

        Qv1 = Q * pv01;
        Qv2 = Q * pv02;
        jac[k++] = discpi * (Qv1 - K * Qv2);
    }
}

// Heston Delta: (parameter, delta, dim_p, dim_x, arguments, dividendYield_curve)
void dHesTerm(double *p, double *delta, int m, int n, void *data, double *q_termS)

{
    int l;

    // retrieve market parameters
    struct mktpara_term *dptr;
    dptr = (struct mktpara_term *)data;

    // retrieve model parameters
    double a = p[0];
    double b = p[1];
    double c = p[2];
    double rho = p[3];
    double v0 = p[4];

    // numerical integral settings
    int NumGrids = glaw.numgrid;
    NumGrids = (NumGrids + 1) >> 1;
    double *u = glaw.u;
    double *w = glaw.w;

    for (l = 0; l < n; ++l)
    {
        double S = dptr->S_adj[l];
        double r = dptr->r[l];
        double K = dptr->K[l];
        double T = dptr->T[l];
        double tmp = exp(-r * T) / pi;
        double Y1 = 0.0, Y2 = 0.0;
        double q = q_termS[l];

        for (int j = 0; j < NumGrids; j++)
        {

            tagMN MN = HesIntMN(u[j], a, b, c, rho, v0, K, T, S, r);

            double M1 = MN.M1;
            double N1 = MN.N1;

            Y1 += w[j] * (M1 + N1);
        }

        double Qv1 = Q * Y1;
        double P1 = (0.5 + tmp * Qv1 / S);
        delta[l] = P1 * exp(-q * T);
    }
}

// Term-structure Heston calibrator taking mktpara as input.
void hestonCalibTerm(const mktpara_term &market, double *price, double *par, double *statistics, int n, int m)
{
    double opts[LM_OPTS_SZ];
    double info[LM_INFO_SZ];

    for (int i = 0; i < LM_OPTS_SZ; ++i)
        opts[i] = 0.0;

    opts[0] = LM_INIT_MU; // initial damping
    opts[1] = 1e-10;      // ||J^T e||_inf
    opts[2] = 1e-10;      // ||Dp||_2
    opts[3] = 1e-10;      // ||e||_2
    opts[4] = LM_DIFF_DELTA;

    int itmax = 500;

    dlevmar_der(fHesTerm, JacHesTerm, par, price, m, n, itmax, opts, info, NULL, NULL, (void*)&market);

    Eigen::Map<Eigen::ArrayXd>(statistics, 7) = Eigen::Map<Eigen::ArrayXd>(info, 7);
}