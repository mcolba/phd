//
// This is an interface definition file for the "Simplified Wrapper and Interface Generator" (https://www.swig.org/)
// to enable the use of the functions
//    Black(double F, double K, double sigma, double T, double q /* q=±1 */);
//    NormalisedBlack(double x, double s, double q /* q=±1 */);
//    ImpliedBlackVolatility(double price, double F, double K, double T, double q /* q=±1 */);
//    NormalisedImpliedBlackVolatility(double beta, double x, double q /* q=±1 */);
// in any of the SWIG-supported target languages.
//
// Depending on your settings or preferences, this may be based on the use of the precompiled Dll "LetsBeRational.xll"
// (same as what can be loaded directly into Excel) or on a recompilation that includes the source files 
//



// For GNU Octave, note:
//     this requires SWIG version 4.1.1. or higher, and the two files octruntime.swg and octrun.swg
//     (usually residing in Lib\octave\ from the swig installation directory)
//     may have to be manually updated according to https://github.com/swig/swig/pull/2512/files.
//     You can find accordingly patched versions of octruntime.swg and octrun.swg in the directory "patched SWIG files".
//     They can be used directly as shown below.
//
//   swig -c++ -octave -I"patched SWIG files" -o lets_be_rational_in_octave_via_swig.cc lets_be_rational_in_octave_via_swig.i
//
// To compile, find the xll that you wish to link against, e.g., Release/x64/LetsBeRational.xll, and build the 'oct' file with
//
//   mkoctfile lets_be_rational_in_octave_via_swig.cc -o x64/lets_be_rational_in_octave_via_swig.oct   -Wl,-rpath=. -L./x64   -l:LetsBeRational.xll -s
//
// or
//
//   mkoctfile lets_be_rational_in_octave_via_swig.cc -o Linux/lets_be_rational_in_octave_via_swig.oct -Wl,-rpath=. -L./Linux -l:LetsBeRational.so  -s
//
// Note: on Windows, the easiest way to invoke 'mkoctfile' may be to open up an Octave-specific command shell,
//       e.g., by running C:\Program Files\GNU Octave\Octave-8.3.0\cmdshell.bat for Octave version 8.3.0.
//

%module letsberational

#define DLL_EXPORT

%{
#ifdef HAVE_SYS_SELECT_H
# undef HAVE_SYS_SELECT_H
#endif /* !HAVE_SYS_SELECT_H */
#include "lets_be_rational.h"
%}
%include "lets_be_rational.h"

extern "C" const char* CPUName();
extern "C" int Revision();

%{ namespace { inline const char* skip_path(const char* file_name) { return (const char*)std::max((uintptr_t)file_name, std::max((uintptr_t)strrchr(file_name, '/'), (uintptr_t)strrchr(file_name, '\\')) + 1); } } %}

%init %{

#ifdef SWIGOCTAVE
#if SWIG_VERSION < 0x040101
#error "AT LEAST SWIG VERSION 4.1.1 IS REQUIRED!"
#endif
#if SWIG_OCTAVE_PREREQ(8,0,0)
  std::string oct_file = octave::interpreter::the_interpreter()->get_evaluator().current_function()->fcn_file_name();
#elif SWIG_OCTAVE_PREREQ(6,0,0)
  std::string oct_file = octave::interpreter::the_interpreter()->get_evaluator().get_call_stack().current_function()->fcn_file_name();
#elif SWIG_OCTAVE_PREREQ(4,4,0)
  std::string oct_file = octave::interpreter::the_interpreter()->get_call_stack().current()->fcn_file_name();
#else
  std::string oct_file = octave_call_stack::current()->fcn_file_name();
#endif
  oct_file = skip_path(oct_file.c_str());
  if (module_ns->swig_members_begin() != module_ns->swig_members_end()) {
#endif

	// Common to all target languages (e.g., PYTHON and OCTAVE).
 //   printf("\nLoaded dynamic load library %s revision %d [%d bit %s running on %s] compiled with %s on %s from %s .\n\n", DLLName(), Revision(), Bitness(), BuildConfiguration(), CPUName(), CompilerVersion(), BuildDate(), DLLDirectory());
//	fflush(stdout);
	
#ifdef SWIGOCTAVE
	printf("Registered the following functions for autoloading from %s by Octave:\n\n",oct_file.c_str());
    for (octave_swig_type::swig_member_const_iterator mb = module_ns->swig_members_begin(); mb != module_ns->swig_members_end(); ++mb)
      if (mb->second.first && mb->second.first->method && mb->second.first->name)
        printf("\t%s()\n",mb->second.first->name);
    printf("\n");
  } else
    printf("\nNothing to be (auto-)loaded from %s.\n\n",oct_file.c_str());
#endif

%}

#ifdef SWIGPYTHON
%pythonbegin %{	import os; saved_cwd = os.getcwd();	os.chdir(os.path.dirname(os.path.realpath(__file__))) %}
%pythoncode  %{ os.chdir(saved_cwd); saved_cwd %}
#endif

%{
#ifdef HAVE_SYS_SELECT_H
//#error ERROR
#endif /* !HAVE_SYS_SELECT_H */
%}
