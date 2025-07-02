//
// To compile, find the xll that you wish to link against, e.g., x64/LetsBeRational.xll, and build with
//
//   g++ -shared lets_be_rational_in_gnuplot.cpp -o x64/letsberational_in_gnuplot.dll -Wl,-rpath=. -L./x64 -l:LetsBeRational.xll -s
//
// or
//
//   g++ -shared -fPIC lets_be_rational_in_gnuplot.cpp -o Linux/letsberational_in_gnuplot.so -Wl,-rpath=. -L./Linux -l:LetsBeRational.so -s
//

#include "gnuplot_plugin.h"
#include <math.h>

#include "lets_be_rational.h"

#include <array>
#include <limits>
#include <cfloat>

#define ENSURE_NARGS(nargs, ACTUAL_NARGS) if (nargs != ACTUAL_NARGS) return { INVALID_VALUE }

#define ENSURE_ISNUMERIC(arg) if (arg.type != CMPLX && arg.type != INTGR) return { INVALID_VALUE }

#include <errno.h>
//
// gnuplot checks errno for EDOM ('Numerical argument out of domain') and ERANGE ('Numerical result out of range') after each and every evaluation.
// In gnuplot version 5.4.2, for example, you find this check in gnuplot's source code at
//   eval.c:684:    if (errno == EDOM || errno == ERANGE)
//                      undefined = TRUE;
// subsequently leading to the error message 'undefined value' immediately upon the return from any plugin function.
// Alas, ERANGE is easily triggered by conventional underflow that we simply want to appear as 0.
// We clear both EDOM and ERANGE if they are inadvertently and unintentionally raised by our functions.
//
inline void clear_edom_and_erange() { if (errno == EDOM || errno == ERANGE) errno = 0; }
inline struct value gint(int i){            clear_edom_and_erange(); return { INTGR, (int64_t)i }; }
inline struct value gdouble(double x){      clear_edom_and_erange(); return { CMPLX, { .cmplx_val = { x, 0.0 } } }; }
inline struct value gstring(const char* s){ clear_edom_and_erange(); struct value r { STRING }; r.v.string_val = const_cast<char*>(s); return r;}
// CAREFUL: the inline function RVAL() in gnuplot_plugin.h can lead to erroneous floating point value conversion when in gnuplot a variable is assigned as a negative int like 'x=-4'.
inline double gdouble(const struct value& v){
  if (v.type == CMPLX) return v.v.cmplx_val.real;
  if (v.type == INTGR) return (double)(int)v.v.int_val; // The extra cast to (int) is needed to enable the correct transport of negative integers.
  return std::numeric_limits<double>::quiet_NaN() ;
}

template<int N> std::array<double,N> get_doubles(struct value *argp){
  std::array<double,N> args;
  for (int i=0;i<N;++i) args[i] = gdouble(argp[i]);
  return args;
}

#define GET_DOUBLES(ARGS,NARGS,ARGP,ACTUAL_NARGS) ENSURE_NARGS(NARGS,ACTUAL_NARGS); { for (int i=0;i<NARGS;++i) ENSURE_ISNUMERIC(ARGP[i]); } const auto ARGS = get_doubles<NARGS>(ARGP)

extern "C" DLLEXPORT struct value cpuname(int nargs, struct value *arg, void *p) { return gstring(CPUName()); }
extern "C" DLLEXPORT struct value dllname(int nargs, struct value *arg, void *p) { return gstring(DLLName()); }
extern "C" DLLEXPORT struct value builddate(int nargs, struct value *arg, void *p) { return gstring(BuildDate()); }
extern "C" DLLEXPORT struct value dlldirectory(int nargs, struct value *arg, void *p) { return gstring(DLLDirectory()); }
extern "C" DLLEXPORT struct value compilerversion(int nargs, struct value *arg, void *p) { return gstring(CompilerVersion()); }
extern "C" DLLEXPORT struct value buildconfiguration(int nargs, struct value *arg, void *p) { return gstring(BuildConfiguration()); }

extern "C" DLLEXPORT struct value revision(int nargs, struct value *arg, void *p) { return gint(Revision()); }
extern "C" DLLEXPORT struct value bitness(int nargs, struct value *arg, void *p) { return gint(Bitness()); }

extern "C" DLLEXPORT struct value dblepsilon(int nargs, struct value *arg, void *p) { return gdouble(DblEpsilon()); }
extern "C" DLLEXPORT struct value dblmin(int nargs, struct value *arg, void *p) { return gdouble(DblMin()); }
extern "C" DLLEXPORT struct value dblmax(int nargs, struct value *arg, void *p) { return gdouble(DblMax()); }

extern "C" DLLEXPORT struct value black(int nargs, struct value *argp, void *p) {
  GET_DOUBLES(args,5,argp,nargs);
  return gdouble(Black(args[0],args[1],args[2],args[3],args[4]));
}

extern "C" DLLEXPORT struct value impliedblackvolatility(int nargs, struct value *argp, void *p) {
  GET_DOUBLES(args,5,argp,nargs);
  return gdouble(ImpliedBlackVolatility(args[0],args[1],args[2],args[3],args[4]));
}

extern "C" DLLEXPORT struct value impliedvolatilityattainableaccuracy(int nargs, struct value *argp, void *p) {
  GET_DOUBLES(args,3,argp,nargs);
  return gdouble(ImpliedVolatilityAttainableAccuracy(args[0],args[1],args[2]));
}

extern "C" DLLEXPORT struct value normalisedblack(int nargs, struct value *argp, void *p) {
  GET_DOUBLES(args,3,argp,nargs);
  return gdouble(NormalisedBlack(args[0],args[1],args[2]));
}

extern "C" double lets_be_rational(double 𝛽, double θx, int N);

extern "C" DLLEXPORT struct value letsberational(int nargs, struct value *argp, void *p) {
  GET_DOUBLES(args,3,argp,nargs);
  return gdouble(lets_be_rational(args[0],args[1],(int)args[2]));
}
