//
// To compile, find the xll that you wish to link against, e.g., x64/LetsBeRational.xll, and build the 'oct' file with
//
//   mkoctfile lets_be_rational_in_octave.cpp -o x64/lets_be_rational_in_octave.oct   -Wl,-rpath=. -L./x64   -l:LetsBeRational.xll -s
//
// or
//
//   mkoctfile lets_be_rational_in_octave.cpp -o Linux/lets_be_rational_in_octave.oct -Wl,-rpath=. -L./Linux -l:LetsBeRational.so  -s
//
// Note: on Windows, the easiest way to invoke 'mkoctfile' may be to open up an Octave-specific command shell,
//       e.g., by running C:\Program Files\GNU Octave\Octave-8.3.0\cmdshell.bat for Octave version 8.3.0.
//

#include <octave/oct.h>
#include <octave/version.h>

// Borrowed from SWIG.
#define OCTAVE_PREREQ(major, minor, patch) \
  ( (OCTAVE_MAJOR_VERSION<<16) + (OCTAVE_MINOR_VERSION<<8) + (OCTAVE_PATCH_VERSION + 0) >= ((major)<<16) + ((minor)<<8) + (patch) )

#include <octave/parse.h>
#if OCTAVE_PREREQ(4,2,0)
#include <octave/interpreter.h>
#include <octave/call-stack.h>
#else
#include <octave/toplev.h>
#endif

#include "lets_be_rational.h"

#define  TOSTRING_AUX(x) #x
#define  TOSTRING(x)     TOSTRING_AUX(x)

namespace {

  std::vector<std::string>& my_function_names() {
    static std::vector<std::string> the_function_names;
    return the_function_names;
  }

  size_t add_function(const char* f) {
    my_function_names().push_back(f);
    return my_function_names().size();
  }

  const char* skip_path(const char* file_name) { return (const char*)std::max((uintptr_t)file_name, std::max((uintptr_t)strrchr(file_name, '/'), (uintptr_t)strrchr(file_name, '\\')) + 1); }

}

#define MY_DEFUN_DLD(name, args_name, nargout_name, doc) static size_t reg_##name = add_function(#name); DEFUN_DLD(name, args_name, nargout_name, doc)

#if OPTIONAL

MY_DEFUN_DLD(BuildDate, /* args : ignored */, /* nargout : ignored */, "The build date of the used dynamic load library.") { return octave_value(BuildDate()); }

MY_DEFUN_DLD(DLLName, /* args : ignored */, /* nargout : ignored */, "The file name of the used dynamic load library.") { return octave_value(DLLName()); }

MY_DEFUN_DLD(CompilerVersion, /* args : ignored */, /* nargout : ignored */, "The compiler name and version number that the used dynamic load library was built with.") { return octave_value(CompilerVersion()); }

MY_DEFUN_DLD(BuildConfiguration, /* args : ignored */, /* nargout : ignored */, "The build configuration, i.e., DEBUG or RELEASE, of the used dynamic load library.") { return octave_value(BuildConfiguration()); }

MY_DEFUN_DLD(Bitness, /* args : ignored */, /* nargout : ignored */, "The Bitness of the used dynamic load library.") { return octave_value(Bitness()); }

MY_DEFUN_DLD(Revision, /* args : ignored */, /* nargout : ignored */, "The revision number of the used dynamic load library.") { return octave_value(Revision()); }

#endif

MY_DEFUN_DLD(Black, args,  /* nargout : ignored */, "Black(F,K,sigma,T,q=±1 for calls/puts)") {
  if (args.length () != 5)
	print_usage ();	
  const double F = args(0).double_value(), K = args(1).double_value(), sigma = args(2).double_value(), T = args(3).double_value(), q = args(4).double_value();
  return octave_value(Black(F,K,sigma,T,q));
}

MY_DEFUN_DLD(CPUName, /* args : ignored */, /* nargout : ignored */, "The CPU model name of the executing hardware.") { return octave_value(CPUName()); }

MY_DEFUN_DLD (ImpliedBlackVolatility, args,  /* nargout : ignored */, "ImpliedBlackVolatility(value,F,K,T,q=±1 for calls/puts)") {
  if (args.length () != 5)
	print_usage ();	
  const double value = args(0).double_value(), F = args(1).double_value(), K = args(2).double_value(), T = args(3).double_value(), q = args(4).double_value();
  return octave_value(ImpliedBlackVolatility(value,F,K,T,q));
}

#ifndef TARGET
# define TARGET letsberational
#endif

DEFUN_DLD(TARGET, /* args : ignored */, /* nargout : ignored */, "The entry point into " TOSTRING(TARGET) ".oct.") {
#if OCTAVE_PREREQ(8,0,0)
  std::string oct_file = octave::interpreter::the_interpreter()->get_evaluator().current_function()->fcn_file_name();
#elif OCTAVE_PREREQ(6,0,0)
  std::string oct_file = octave::interpreter::the_interpreter()->get_evaluator().get_call_stack().current_function()->fcn_file_name();
#elif OCTAVE_PREREQ(4,4,0)
  std::string oct_file = octave::interpreter::the_interpreter()->get_call_stack().current()->fcn_file_name();
#else
  std::string oct_file = octave_call_stack::current()->fcn_file_name();
#endif  
  if (my_function_names().size() > 0) {
    octave_value_list auto_load_item(2);
	auto_load_item(1) = oct_file;
    for (const auto& f : my_function_names()){
      auto_load_item(0) = f;
#if OCTAVE_PREREQ(4,4,0)
      octave::
#endif
      feval("autoload", auto_load_item, 0);
    }
    printf("\nLoaded dynamic load library %s revision %d [%d bit %s running on %s] compiled with %s on %s from %s .\n\n", DLLName(), Revision(), Bitness(), BuildConfiguration(), CPUName(), CompilerVersion(), BuildDate(), DLLDirectory());
    printf("Registered the following functions for autoloading from %s by Octave:\n\n", skip_path(oct_file.c_str()));
    for (const auto& f : my_function_names())
      printf("\t%s()\n",f.c_str());
    printf("\n");
  } else
    printf("\nNothing to be (auto-)loaded from %s.\n\n", skip_path(oct_file.c_str()));
  fflush(stdout);
  return octave_value();
}
