// Windows DLL bindings for the Heston pricing, Greeks, Jacobian, and calibration routines.
// The code is an extension of the code from Yiran Cui (see copitight details in HestonCalibrator.cpp).
//
// Exported functions:
//   hestonPricer     – option prices for a given term structure.
//   hestonDelta      – deltas for the same market configuration.
//   hestonJac        – Jacobian of prices w.r.t. model parameters.
//   hestonCalibrator – calibrates `par` to observed option prices.


#include <Eigen/Dense>
#include <cmath>
#include "HestonTermStructure.h"

#define WIN32_LEAN_AND_MEAN // Exclude rarely-used stuff from Windows headers
// Windows Header Files:
#include <windows.h>

using namespace std;

// Heston Pricer
extern "C" __declspec(dllexport) void hestonPricer(double *S_adj, double *r, double *K,
												   double *mat, double *par, int *n, int *m, double *p)
{
	// m = # of parameters
	// n = # of observations

	mktpara_term market(*n);
	market.K = Eigen::Map<Eigen::ArrayXd>(K, *n);
	market.T = Eigen::Map<Eigen::ArrayXd>(mat, *n);
	market.r = Eigen::Map<Eigen::ArrayXd>(r, *n);
	market.S_adj = Eigen::Map<Eigen::ArrayXd>(S_adj, *n);

	fHesTerm(par, p, *m, *n, (void *)&market);
}

// Heston Delta
extern "C" __declspec(dllexport) void hestonDelta(double *S_adj, double *r, double *q, double *K,
												  double *mat, double *par, int *n, int *m, double *delta)
{
	// m = # of parameters
	// n = # of observations

	mktpara_term market(*n);
	market.K = Eigen::Map<Eigen::ArrayXd>(K, *n);
	market.T = Eigen::Map<Eigen::ArrayXd>(mat, *n);
	market.r = Eigen::Map<Eigen::ArrayXd>(r, *n);
	market.S_adj = Eigen::Map<Eigen::ArrayXd>(S_adj, *n);

	dHesTerm(par, delta, *m, *n, (void *)&market, q);
}

// Heston Jacobian Matrix
extern "C" __declspec(dllexport) void hestonJac(double *S_adj, double *r, double *K,
												double *mat, double *par, int *n, int *m, double *jac)
{
	// m = # of parameters
	// n = # of observations

	mktpara_term market(*n);
	market.K = Eigen::Map<Eigen::ArrayXd>(K, *n);
	market.T = Eigen::Map<Eigen::ArrayXd>(mat, *n);
	market.r = Eigen::Map<Eigen::ArrayXd>(r, *n);
	market.S_adj = Eigen::Map<Eigen::ArrayXd>(S_adj, *n);

	JacHesTerm(par, jac, *m, *n, (void *)&market);
}

// Heston Calibrator
extern "C" __declspec(dllexport) void hestonCalibrator(double *S_adj, double *r, double *K,
													   double *mat, double *price, double *par,
													   double *statistics, int *n, int *m)
{

	mktpara_term market(*n);
	market.K = Eigen::Map<Eigen::ArrayXd>(K, *n);
	market.T = Eigen::Map<Eigen::ArrayXd>(mat, *n);
	market.r = Eigen::Map<Eigen::ArrayXd>(r, *n);
	market.S_adj = Eigen::Map<Eigen::ArrayXd>(S_adj, *n);

	hestonCalibTerm(market, price, par, statistics, *n, *m);
}