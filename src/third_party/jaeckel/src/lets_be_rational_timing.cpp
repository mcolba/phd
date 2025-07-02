#include <math.h>
#include <climits>
#include <float.h>
#include <vector>
#include <time.h>
#include <stdio.h>

#include "lets_be_rational.h"

#if defined(_MSC_VER)
# pragma warning (disable : 26451)
#endif

#include <cstring> // for strrchr()

namespace { const char* skip_path(const char* file_name) { return (const char*)std::max((uintptr_t)file_name, std::max((uintptr_t)strrchr(file_name, '/'), (uintptr_t)strrchr(file_name, '\\')) + 1); } }

// Compile with:
//                f=lets_be_rational_timing; g++ -DNDEBUG -Ofast $f.cpp -o Linux/$f -s -Wl,-rpath=. -L./Linux -l:LetsBeRational.so
// or
//                f=lets_be_rational_timing; g++ -DNDEBUG -Ofast $f.cpp -o x64/$f   -s -Wl,-rpath=. -L./Linux -l:LetsBeRational.xll

//
//  x = ln(F/K)
//
//  β = price / √(F·K)
//
//  Thus, when F = 1, we have K = exp(-x) and
//
//     β = price · exp(x/2)
//     price = β · exp(-x/2)
//     price = β · √K
//
// Invocation example:
//
//   ./letsbe_rational_timing  2097152  0 1E-14 1E-8 1E-4 0.001 0.01 0.05 0.1 0.25 0.5 1 2 4 8 16 32 64 128 256 512
//

#define NO_PREMAPPING_OF_VOLATILITIES

#if defined( HAVE_BPROF )
extern "C" void bprof_start();
extern "C" void bprof_stop();
#else
#define bprof_start()
#define bprof_stop()
#endif

#if defined(_POSIX_C_SOURCE) && (_POSIX_C_SOURCE >= 199309L) && !defined( USE_ELAPSED_CLOCK_TIME_FOR_TIMING )
#define GET_CLOCK( CLOCK_VAR_NAME ) timespec CLOCK_VAR_NAME; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &CLOCK_VAR_NAME)
#define MICROSECONDS_BETWEEN_CLOCKS( START, END ) ((END.tv_sec - START.tv_sec) * 1E6 + (END.tv_nsec - START.tv_nsec) * 1E-3)
#else
#define GET_CLOCK( CLOCK_VAR_NAME ) const clock_t CLOCK_VAR_NAME = clock();
#define MICROSECONDS_BETWEEN_CLOCKS( START, END ) ((END - START) / ((double)CLOCKS_PER_SEC) * 1000000)
#endif

int main(int argc, char** argv) {
  bprof_stop();
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <number_of_points> <x> [ <x2> <x3> ...]\n", skip_path(argv[0]));
    fprintf(stderr, "\nInvocation example:\n\n   %s   2097152  0 1E-14 1E-8 1E-4 0.001 0.01 0.05 0.1 0.25 0.5 1 2 4 8 16 32 64 128 256 512\n\n", skip_path(argv[0]));
    fflush(stderr);
    return -1;
  }
  const int m = std::max((int)strtod(argv[1], 0), 4), m1 = m / 2;
  std::vector<double> x_values(argc - 2), s(m), price(m), sigma_input(m), sigma_output(m);
  for (int i_arg = 2; i_arg < argc; ++i_arg) x_values[(size_t)i_arg - 2] = strtod(argv[i_arg], 0);
  printf("\n%s\t%d\t%g", skip_path(argv[0]), m, x_values[0]);
  for (int i_arg = 3; i_arg < argc; ++i_arg) printf(" %g", x_values[(size_t)i_arg - 2]);
  printf("\n");
  printf("\nLoaded %s revision %d [%d bit %s running on %s] compiled with %s on %s from %s .\n\n", DLLName(), Revision(), Bitness(), BuildConfiguration(), CPUName(), CompilerVersion(), BuildDate(), DLLDirectory());
  fflush(stdout);
  // When denormalized (aka 'subnormal) numbers are automatically 'flushed to zero', the minimum price that we can reliably reproduce (e.g., for x=-1E-8) is 2·DBL_MIN due to the first iteration (from DBL_MIN or nearby)
  // slightly going too far, i.e., ever so slightly below (normalised_black = ) DBL_MIN, and then converges in a much smaller step back. This works fine if the platform is happy with denormalized (subnormal) numbers,
  // i.e., numbers between DBL_TRUE_MIN and DBL_MIN. The gcc/g++ compiler, however, is best optimised with settings that map all numbers below DBL_MIN to zero, whence, we cannot use DBL_MIN as an input price with g++.
  // When compiling with MSVC with the settings in this 'solution' ['LetsBeRational (Visual Studio 2022).sln'], where we do have access to numbers between DBL_TRUE_MIN and DBL_MIN, then we can use DBL_MIN as a target
  // (normalised) Black function price, and then this test here works fine (e.g., when x=-1E-8), but not with g++. Note: DBL_TRUE_MIN = DBL_MIN·DBL_EPSILON. To cater for all cases, we set 𝛽_min := 2·DBL_MIN.
  double total_elapsed_μsec = 0, total_rel_sum = 0;
  size_t total_evaluation_count = 0, total_success_count = 0;
  for (const auto x_val : x_values) {
    const double x = -fabs(x_val), F = 1, K = F * exp(-x), sqrt_FK = sqrt(F) * sqrt(K), T = 1, sqrt_T = sqrt(T), call_put = 1;
    const double s_mid = sqrt(std::max(2 * fabs(x), DBL_EPSILON)), beta_mid = NormalisedBlack(x, s_mid, call_put);
    double beta_min, min_price = 0, beta_max, max_price, s_min = 0, s_max = DBL_MAX;
    for (unsigned int i = 1; i < USHRT_MAX && (s_min <= DBL_MIN || min_price < DBL_MIN); ++i) {
      beta_min = i * DBL_MIN;
      s_min = ImpliedBlackVolatility(beta_min * sqrt_FK, F, K, T, call_put) * sqrt_T;
      min_price = Black(F, K, s_min / sqrt_T, T, call_put);
      if (min_price >= DBL_MIN && s_min > DBL_MIN)
        s_min = ImpliedBlackVolatility(min_price, F, K, T, call_put) * sqrt_T;
    }
    for (unsigned int i = 1; i < INT_MAX && s_max >= DBL_MAX; ++i){
      beta_max = exp(x / 2) * (1 - i * DBL_EPSILON);
      max_price = beta_max * sqrt_FK;
      s_max = ImpliedBlackVolatility(max_price, F, K, T, call_put) * sqrt_T;
    }
    const double log_beta_min = log(beta_min), d_logb = (log(beta_mid) - log_beta_min) / (m1 - 1), ds = (s_max - s_mid) / (m - m1);
    for (int i = 0; i < m1; ++i)
      // We ensure that the very first price used here is exactly equal to beta_min since we previously asserted that beta_min leads to a positive s_min.
      s[i] = ImpliedBlackVolatility(beta_min * exp(i * d_logb) * sqrt_FK, F, K, T, call_put) * sqrt_T;
    for (int i = m1; i < m; ++i)
      s[i] = s_mid + (i - m1 + 1) * ds;
    for (int i = 0; i < m; ++i) {
      price[i] = Black(F, K, sigma_input[i] = s[i] / sqrt_T, T, call_put);
      if (price[i] <= 0) {
        printf("%d: Black(%.17g,%.17g,%.17g,%.17g,%.17g) returned %g.\n", i,  F, K, sigma_input[i], T, call_put, price[i]);
        fflush(stdout);
      }
#if !defined( NO_PREMAPPING_OF_VOLATILITIES )
      sigma_input[i] = ImpliedBlackVolatility(price[i], F, K, T, call);
#endif
    }
    printf("#\n# |x|=%.16g\tnumber_of_points=%d\tbeta_min=%g\tbeta_mid=%g\tbeta_max=%s%G\ts_min=%g\ts_mid=%g\ts_max=%g\n", fabs(x), m, beta_min, beta_mid, (fabs(x)<1E-4?(beta_max<1?"1-":beta_max>1?"1+":""):""),(fabs(x)<1E-4?(1==beta_max?beta_max:fabs(1-beta_max)):beta_max), s[0], s[(size_t)m1 - 1], s[(size_t)m - 1]);
    fflush(stdout);
    int ibegin, iend;
    for (ibegin = 0; ibegin < m && (price[ibegin] <= 0 || sigma_input[ibegin] <= 0); ++ibegin);
    for (iend = m - 1; iend >= 0 && (price[iend] >= F || sigma_input[iend] >= DBL_MAX); --iend);

    GET_CLOCK(start_clock);
    bprof_start();
    for (int i = ibegin; i <= iend; ++i)
      sigma_output[i] = ImpliedBlackVolatility(price[i], F, K, T, call_put);
    bprof_stop();
    GET_CLOCK(end_clock);

    const double elapsed_μsec = MICROSECONDS_BETWEEN_CLOCKS(start_clock, end_clock);

    double sum = 0, rel_sum = 0;
    int evaluation_count = 0, success_count = 0;
    for (int i = ibegin; i <= iend; ++i) {
      ++evaluation_count;
      if (sigma_output[i] > -DBL_MAX && sigma_output[i] < DBL_MAX) {
        ++success_count;
        sum += sigma_output[i];
        double rel_diff = fabs(sigma_output[i] / sigma_input[i] - 1);
#if defined( NO_PREMAPPING_OF_VOLATILITIES ) // Express relative (residual) difference as a proportion of what is theoretically attainable. Numbers lower than 1 suggest that the numerics is, net, more accurate than DBL_EPSILON, which is the nominal granularity.
        rel_diff /= (DBL_EPSILON * (1 + fabs(1 / BlackAccuracyFactor(x, sigma_input[i] * sqrt_T, 1))));
#if defined( LOG_OUTLIERS )
        if (rel_diff > 2) {
          printf("%d: ImpliedBlackVolatility(%.17g,%.17g,%.17g,%.17g,%.17g) returned %.17g.\n\tsigma: %.17g, rel_diff: %g, attainable: %g, rel_diff/attainable: %g.\n", i, price[i], F, K, T, call_put, sigma_output[i], sigma_input[i], fabs(sigma_output[i] / sigma_input[i] - 1), (DBL_EPSILON * (1 + fabs(1 / BlackAccuracyFactor(x, sigma_input[i] * sqrt_T, 1)))), rel_diff);  fflush(stdout);
        }
#endif
#endif
        rel_sum += rel_diff;
      } else {
        printf("%d: ImpliedBlackVolatility(%.17g,%.17g,%.17g,%.17g,%.17g) returned %g.\n", i, price[i], F, K, T, call_put, sigma_output[i]);  fflush(stdout);
      }
    }

#if defined( NO_PREMAPPING_OF_VOLATILITIES )
    //
    // NOTE: when this number is below 1, the algorithm does better than what is theoretically attainable according to error propagation analysis (1st order).
    // See lets_be_rational.cpp for a derivation.
    //
#define RELATIVE_ACCURACY_OUTPUT "|(attained accuracy)/(theoretically attainable accuracy)|"
#else
#define RELATIVE_ACCURACY_OUTPUT "|(relative volatility difference)|"
#endif
    printf("#\n# Total implied volatility evaluations: %d\tAverage time: %g microseconds\tAverage volatility [%d]: %g\tAverage " RELATIVE_ACCURACY_OUTPUT " : %g\n", evaluation_count, elapsed_μsec / evaluation_count, success_count, sum / success_count, rel_sum / success_count);
    fflush(stdout);
    total_elapsed_μsec += elapsed_μsec;
    total_evaluation_count += evaluation_count;
    total_rel_sum += rel_sum;
    total_success_count += success_count;
  }
  printf("\nAverage over %zu different values for x:  %g  microseconds per evaluation (%zu successful evaluations in total).\n\nTotal average " RELATIVE_ACCURACY_OUTPUT " : %g.\n", x_values.size(), total_elapsed_μsec / total_evaluation_count, total_evaluation_count, total_rel_sum / total_success_count);
#if defined( NO_PREMAPPING_OF_VOLATILITIES )
  printf("This means the method is on average %g times more accurate than what one could\nhope for on the basis of first order error propagation analysis.\n", floor(total_success_count / total_rel_sum * 10 + 0.5) / 10);
#endif
  printf("\n");
  fflush(stdout);
  return 0;
}

/*

Output on 12th Gen Intel(R) Core(TM) i5-12500H under Windows 11 with Visual-Studio-2022-generated 64 bit LetsBeRational.xll:

  letsberational_timing.exe	2097152	0 1e-14 1e-08 0.0001 0.001 0.01 0.05 0.1 0.25 0.5 1 2 4 8 16 32 64 128 256 512

  Loaded LetsBeRational.xll revision 1520 [64 bit RELEASE running on 12th Gen Intel(R) Core(TM) i5-12500H] compiled with MSVC 1938 on Thu Feb 15 18:38:28 2024 from ./x64 .

  #
  # |x|=0	number_of_points=2097152	beta_min=2.22507e-308	beta_mid=5.9447e-09	beta_max=1-2.22045E-16	s_min=5.57743e-308	s_mid=1.49012e-08	s_max=16.4191
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.0162125 microseconds	Average volatility [2097152]: 4.1048	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.107859
  #
  # |x|=1e-14	number_of_points=2097152	beta_min=4.45015e-308	beta_mid=5.6419e-08	beta_max=1-5.21805E-15	s_min=2.74241e-16	s_mid=1.41421e-07	s_max=16.4191
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.206947 microseconds	Average volatility [2097152]: 4.10487	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.199918
  #
  # |x|=1e-08	number_of_points=2097152	beta_min=2.22507e-308	beta_mid=5.6414e-05	beta_max=1-5E-09	s_min=2.71516e-10	s_mid=0.000141421	s_max=16.4191
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.206947 microseconds	Average volatility [2097152]: 4.10491	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.176822
  #
  # |x|=0.0001	number_of_points=2097152	beta_min=2.22507e-308	beta_mid=0.00559199	beta_max=0.99995	s_min=2.69694e-06	s_mid=0.0141421	s_max=16.4191
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.221252 microseconds	Average volatility [2097152]: 4.10842	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.181558
  #
  # |x|=0.001	number_of_points=2097152	beta_min=2.22507e-308	beta_mid=0.0173442	beta_max=0.9995	s_min=2.69245e-05	s_mid=0.0447214	s_max=16.5847
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.220776 microseconds	Average volatility [2097152]: 4.15711	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.186806
  #
  # |x|=0.01	number_of_points=2097152	beta_min=2.22507e-308	beta_mid=0.0515133	beta_max=0.995012	s_min=0.000268798	s_mid=0.141421	s_max=16.4191
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.220776 microseconds	Average volatility [2097152]: 4.14062	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.167185
  #
  # |x|=0.05	number_of_points=2097152	beta_min=2.22507e-308	beta_mid=0.102224	beta_max=0.97531	s_min=0.00134243	s_mid=0.316228	s_max=16.4191
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.220776 microseconds	Average volatility [2097152]: 4.18558	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.182049
  #
  # |x|=0.1	number_of_points=2097152	beta_min=2.22507e-308	beta_mid=0.13147	beta_max=0.951229	s_min=0.00268352	s_mid=0.447214	s_max=16.4191
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.217915 microseconds	Average volatility [2097152]: 4.21981	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.182782
  #
  # |x|=0.25	number_of_points=2097152	beta_min=2.22507e-308	beta_mid=0.169576	beta_max=0.882497	s_min=0.0067044	s_mid=0.707107	s_max=16.419
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.220776 microseconds	Average volatility [2097152]: 4.28905	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.191009
  #
  # |x|=0.5	number_of_points=2097152	beta_min=2.22507e-308	beta_mid=0.185683	beta_max=0.778801	s_min=0.0134021	s_mid=1	s_max=16.419
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.225067 microseconds	Average volatility [2097152]: 4.36917	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.19412
  #
  # |x|=1	number_of_points=2097152	beta_min=2.22507e-308	beta_mid=0.173594	beta_max=0.606531	s_min=0.026791	s_mid=1.41421	s_max=16.5843
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.227451 microseconds	Average volatility [2097152]: 4.52796	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.207222
  #
  # |x|=2	number_of_points=2097152	beta_min=2.22507e-308	beta_mid=0.122098	beta_max=0.367879	s_min=0.0535553	s_mid=2	s_max=16.7471
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.228405 microseconds	Average volatility [2097152]: 4.74133	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.190898
  #
  # |x|=4	number_of_points=2097152	beta_min=2.22507e-308	beta_mid=0.0503856	beta_max=0.135335	s_min=0.107058	s_mid=2.82843	s_max=16.9049
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.226021 microseconds	Average volatility [2097152]: 5.03963	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.210302
  #
  # |x|=8	number_of_points=2097152	beta_min=2.22507e-308	beta_mid=0.00742863	beta_max=0.0183156	s_min=0.21401	s_mid=4	s_max=17.3664
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.228405 microseconds	Average volatility [2097152]: 5.54767	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.199564
  #
  # |x|=16	number_of_points=2097152	beta_min=2.22507e-308	beta_mid=0.000144752	beta_max=0.000335463	s_min=0.427814	s_mid=5.65685	s_max=18.2291
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.226498 microseconds	Average volatility [2097152]: 6.36912	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.202945
  #
  # |x|=32	number_of_points=2097152	beta_min=2.22507e-308	beta_mid=5.07396e-08	beta_max=1.12535E-07	s_min=0.855246	s_mid=8	s_max=19.6294
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.236511 microseconds	Average volatility [2097152]: 7.6691	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.194758
  #
  # |x|=64	number_of_points=2097152	beta_min=2.22507e-308	beta_mid=5.88893e-15	beta_max=1.26642E-14	s_min=1.70998	s_mid=11.3137	s_max=22.1223
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.232697 microseconds	Average volatility [2097152]: 9.80438	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.223058
  #
  # |x|=128	number_of_points=2097152	beta_min=2.22507e-308	beta_mid=7.62071e-29	beta_max=1.60381E-28	s_min=3.42094	s_mid=16	s_max=26.097
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.23365 microseconds	Average volatility [2097152]: 13.2325	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.235364
  #
  # |x|=256	number_of_points=2097152	beta_min=2.22507e-308	beta_mid=1.24084e-56	beta_max=2.57221E-56	s_min=6.8601	s_mid=22.6274	s_max=32.2536
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.249386 microseconds	Average volatility [2097152]: 18.7257	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.274997
  #
  # |x|=512	number_of_points=2097152	beta_min=2.22507e-308	beta_mid=3.22573e-112	beta_max=6.61626E-112	s_min=13.8959	s_mid=32	s_max=41.1406
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.23365 microseconds	Average volatility [2097152]: 27.4546	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.309155

  Average over 20 different values for x:  0.215006  microseconds per evaluation (41943040 successful evaluations in total).

  Total average |(attained accuracy)/(theoretically attainable accuracy)| : 0.200918.
  This means the method is on average 5 times more accurate than what one could
  hope for on the basis of first order error propagation analysis.

Same output on the same 12th Gen Intel(R) Core(TM) i5-12500H under Windows-Subsystem-for-Linux ('WSL') with g++-generated LetsBeRational.so:

  letsberational_timing	2097152	0 1e-14 1e-08 0.0001 0.001 0.01 0.05 0.1 0.25 0.5 1 2 4 8 16 32 64 128 256 512

  Loaded LetsBeRational.so revision 1520 [64 bit RELEASE running on 12th Gen Intel(R) Core(TM) i5-12500H] compiled with GCC 11.4.0 on Feb 15 2024 18:47:33 from ./Linux/. .

  #
  # |x|=0	number_of_points=2097152	beta_min=6.67522e-308	beta_mid=5.9447e-09	beta_max=1-2.22045E-16	s_min=1.67323e-307	s_mid=1.49012e-08	s_max=16.4191
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.0134365 microseconds	Average volatility [2097152]: 4.1048	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.112388
  #
  # |x|=1e-14	number_of_points=2097152	beta_min=4.45015e-308	beta_mid=5.6419e-08	beta_max=1-5.44009E-15	s_min=2.74241e-16	s_mid=1.41421e-07	s_max=16.4191
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.171134 microseconds	Average volatility [2097152]: 4.10487	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.200132
  #
  # |x|=1e-08	number_of_points=2097152	beta_min=4.45015e-308	beta_mid=5.6414e-05	beta_max=1-5E-09	s_min=2.71654e-10	s_mid=0.000141421	s_max=16.4191
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.17194 microseconds	Average volatility [2097152]: 4.10491	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.177151
  #
  # |x|=0.0001	number_of_points=2097152	beta_min=2.22507e-308	beta_mid=0.00559199	beta_max=0.99995	s_min=2.69694e-06	s_mid=0.0141421	s_max=16.4191
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.182781 microseconds	Average volatility [2097152]: 4.10842	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.181952
  #
  # |x|=0.001	number_of_points=2097152	beta_min=2.22507e-308	beta_mid=0.0173442	beta_max=0.9995	s_min=2.69245e-05	s_mid=0.0447214	s_max=16.5847
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.182239 microseconds	Average volatility [2097152]: 4.15711	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.185877
  #
  # |x|=0.01	number_of_points=2097152	beta_min=2.22507e-308	beta_mid=0.0515133	beta_max=0.995012	s_min=0.000268798	s_mid=0.141421	s_max=16.4191
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.182754 microseconds	Average volatility [2097152]: 4.14062	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.165852
  #
  # |x|=0.05	number_of_points=2097152	beta_min=4.45015e-308	beta_mid=0.102224	beta_max=0.97531	s_min=0.0013431	s_mid=0.316228	s_max=16.4191
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.183087 microseconds	Average volatility [2097152]: 4.18558	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.180615
  #
  # |x|=0.1	number_of_points=2097152	beta_min=4.45015e-308	beta_mid=0.13147	beta_max=0.951229	s_min=0.00268486	s_mid=0.447214	s_max=16.4191
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.184021 microseconds	Average volatility [2097152]: 4.21981	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.181276
  #
  # |x|=0.25	number_of_points=2097152	beta_min=2.22507e-308	beta_mid=0.169576	beta_max=0.882497	s_min=0.0067044	s_mid=0.707107	s_max=16.419
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.185245 microseconds	Average volatility [2097152]: 4.28905	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.189685
  #
  # |x|=0.5	number_of_points=2097152	beta_min=2.22507e-308	beta_mid=0.185683	beta_max=0.778801	s_min=0.0134021	s_mid=1	s_max=16.419
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.190011 microseconds	Average volatility [2097152]: 4.36917	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.192689
  #
  # |x|=1	number_of_points=2097152	beta_min=4.45015e-308	beta_mid=0.173594	beta_max=0.606531	s_min=0.0268043	s_mid=1.41421	s_max=16.5843
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.189848 microseconds	Average volatility [2097152]: 4.52798	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.204945
  #
  # |x|=2	number_of_points=2097152	beta_min=4.45015e-308	beta_mid=0.122098	beta_max=0.367879	s_min=0.0535819	s_mid=2	s_max=16.7471
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.190014 microseconds	Average volatility [2097152]: 4.74136	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.188231
  #
  # |x|=4	number_of_points=2097152	beta_min=2.22507e-308	beta_mid=0.0503856	beta_max=0.135335	s_min=0.107058	s_mid=2.82843	s_max=16.9049
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.189387 microseconds	Average volatility [2097152]: 5.03963	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.207811
  #
  # |x|=8	number_of_points=2097152	beta_min=2.22507e-308	beta_mid=0.00742863	beta_max=0.0183156	s_min=0.21401	s_mid=4	s_max=17.3664
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.190648 microseconds	Average volatility [2097152]: 5.54767	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.197195
  #
  # |x|=16	number_of_points=2097152	beta_min=4.45015e-308	beta_mid=0.000144752	beta_max=0.000335463	s_min=0.428025	s_mid=5.65685	s_max=18.2291
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.191559 microseconds	Average volatility [2097152]: 6.3693	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.195542
  #
  # |x|=32	number_of_points=2097152	beta_min=4.45015e-308	beta_mid=5.07396e-08	beta_max=1.12535E-07	s_min=0.855669	s_mid=8	s_max=19.6294
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.198529 microseconds	Average volatility [2097152]: 7.66943	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.187576
  #
  # |x|=64	number_of_points=2097152	beta_min=4.45015e-308	beta_mid=5.88893e-15	beta_max=1.26642E-14	s_min=1.71083	s_mid=11.3137	s_max=22.1223
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.197074 microseconds	Average volatility [2097152]: 9.80499	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.205469
  #
  # |x|=128	number_of_points=2097152	beta_min=2.22507e-308	beta_mid=7.62071e-29	beta_max=1.60381E-28	s_min=3.42094	s_mid=16	s_max=26.097
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.198478 microseconds	Average volatility [2097152]: 13.2325	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.218411
  #
  # |x|=256	number_of_points=2097152	beta_min=4.45015e-308	beta_mid=1.24084e-56	beta_max=2.57221E-56	s_min=6.86354	s_mid=22.6274	s_max=32.2536
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.202566 microseconds	Average volatility [2097152]: 18.7276	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.250375
  #
  # |x|=512	number_of_points=2097152	beta_min=2.22507e-308	beta_mid=3.22573e-112	beta_max=6.61626E-112	s_min=13.8959	s_mid=32	s_max=41.1406
  #
  # Total implied volatility evaluations: 2097152	Average time: 0.195823 microseconds	Average volatility [2097152]: 27.4546	Average |(attained accuracy)/(theoretically attainable accuracy)| : 0.287964

  Average over 20 different values for x:  0.179529  microseconds per evaluation (41943040 successful evaluations in total).

  Total average |(attained accuracy)/(theoretically attainable accuracy)| : 0.195557.
  This means the method is on average 5.1 times more accurate than what one could
  hope for on the basis of first order error propagation analysis.

*/
