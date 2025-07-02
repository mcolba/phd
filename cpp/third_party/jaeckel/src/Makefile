
Default:

CURDIRBASE := $(shell basename "$(CURDIR)")
TARGET := $(subst -,_,$(subst .,_,$(subst XLL,,$(CURDIRBASE))))

SWIG=swig
#SWIG=~/bin/swigwin-4.1.1/swig.exe

MIN_SWIG_VERSION_FOR_OCTAVE=4.1.1
MIN_SWIG_VERSION_FOR_PYTHON=3.0.0

BPROF=bprof

#PYTHON=python3 # will be autodetected if not defined.

# DEFINES += VERBOSE_DEBUG_LOG

# For WIN32 cross-compilation (automatically defines C preprocessor macros WIN32 and _WIN32, works with Cygwin's g++ version 7.4.0):
#  CROSS_PLATFORM_PREFIX=i686-w64-mingw32-
# or
#  CXX=i686-w64-mingw32-g++

# For WIN64 cross-compilation (automatically defines C preprocessor macro WIN32, _WIN32, WIN64, and _WIN64, works with Cygwin's g++ version 7.4.0):
#  CROSS_PLATFORM_PREFIX=x86_64-w64-mingw32-
# or
#  CXX=x86_64-w64-mingw32-g++

lower_case = $(shell echo $(1) | tr '[:upper:]' '[:lower:]')

target = $(call lower_case,$(TARGET))

ifeq ($(origin CXX),default)
 ifneq ($(PLATFORM),)
  ifeq ($(PLATFORM),win32)
   CROSS_PLATFORM_PREFIX=i686-w64-mingw32-
  else
   ifeq ($(PLATFORM),win64)
    CROSS_PLATFORM_PREFIX=x86_64-w64-mingw32-
   else
    $(error Bad PLATFORM specification ('PLATFORM = $(PLATFORM)') - valid are only 'win32' and 'win64' for cross-platform compilation.)
   endif
  endif
 endif
 ifneq ($(CROSS_PLATFORM_PREFIX),)
  CXX := $(CROSS_PLATFORM_PREFIX)$(CXX)
 endif
else
 ifeq ($(CROSS_PLATFORM_PREFIX),)
  CROSS_PLATFORM_PREFIX := $(patsubst %-g++,%-,$(filter %-g++,$(CXX)))
 endif
endif

ifeq ($(OS),)
 OS = $(shell uname -s)
endif

ifeq ($(OS),Windows_NT)
 XLL_EXT = xll
 DLL_EXT = dll
 PY_DLL_EXT = pyd
 ALTERNATIVE_PYTHON_NAME = py
#   Regarding the dependency on libwinpthread-1.dll, see, e.g., https://github.com/wkhtmltopdf/wkhtmltopdf/issues/1723 or https://groups.google.com/g/fltkgeneral/c/PZezs2tAo2o
#   It can be obtained, e.g., from https://packages.msys2.org/package/mingw-w64-x86_64-libwinpthread-git.
#   The issue is essentially a gcc/g++ build-time configuration feature.
 WHEEL_OPTIONAL_BINARIES = libwinpthread-1.dll
else
 DLL_EXT = so
 XLL_EXT = $(DLL_EXT)
 PY_DLL_EXT = $(DLL_EXT)
 ALTERNATIVE_PYTHON_NAME = python3
endif

ifeq ($(findstring w32,$(CXX)),w32)
 XLL_EXT=xll
endif

# If XL_CATEGORY_NAME is not defined, the Excel function category is determined dynamically as the XLL's base file name (without path and without extension).
DEFINES += XL_CATEGORY_NAME=$(TARGET)

# You can use this syntax to convert Microsoft "BOM"-marked UTF8 files into files that preserve UTF8 but are also understood by g++.
Example_in_utf8_without_bom.cpp:	Example_in_utf8_with_bom.cpp
	uconv --to-code UTF-8 --remove-signature  $< | dos2unix > $@

# Include directories
INCDIRS := .

DEFFLAGS := $(DEFINES:%=-D%)

INCFLAGS := $(INCDIRS:%=-I%)

LIBFLAGS    =

CXXFLAGS  = -Wall -finput-charset=UTF-8 -DNDEBUG -fextended-identifiers -ffp-contract=fast -march=native -O3 $(EXTRA_CXXFLAGS)
CXXFLAGS += -Ofast # Careful: small changes to the code logic can cause excessive roundoff errors with g++ for extreme values, e.g., when |x| >= 256. This optimisation relies on there being no underflow etc. by design algorithm design.
#  '-Ofast' implies '-fast-math' which in turn invokes -funsafe-math-optimizations, and therein lies the crux!

# CAUTION: when compiling with the MINGW compiler that comes with GNU Octave for Windows, we are always left with a dependency on libwinpthread-1.dll.
#          There are many online discussions about this - and it is not unique to the MINGW g++ version that comes with Octave.
#          See, e.g., https://github.com/wkhtmltopdf/wkhtmltopdf/issues/1723 or https://groups.google.com/g/fltkgeneral/c/PZezs2tAo2o
#          It can be obtained, e.g., from https://packages.msys2.org/package/mingw-w64-x86_64-libwinpthread-git.
#          The issue is essentially a gcc/g++ build-time configuration feature.
#          The default cross compilers that come with Ubuntu, however, i.e., i686-w64-mingw32-g++ for 32 bit and x86_64-w64-mingw32-g++ for 64 bit,
#          do not appear to have the same issue. In essence, we can build *.oct files with the Windows version of GNU Octave to be used under its own
#          runtime (where the path includes, e.g., C:\Program Files\GNU Octave\Octave-8.3.0\mingw64\bin, the location of libwinpthread-1.dll), but
#          any DLLs built with that compiler may not work outside GNU Octave (on Windows) for any other Windows programmes, e.g., Excel.
#          Unless you copy all required DLLs over to x64/ or Win32/, obviously.
MINGWLDFLAGS    = -static-libgcc -static-libstdc++
PROFILINGFLAGS  = -fno-omit-frame-pointer -g -pg -fprofile-arcs
BPROFILINGFLAGS = -g -DHAVE_BPROF
SOLDFLAGS       = -shared
#
# Regarding the default C++ standard used by your g++, you can try
#   g++ -x c++  -E -dM -< /dev/null | grep __cplusplus
# to find out.
#

ifeq ($(XLL_EXT),xll)
 EXE = .exe
 LDFLAGS += $(MINGWLDFLAGS)
 ifeq ($(CXX_TARGET_BITNESS),)
	CXX_TARGET_BITNESS = $(shell d=`mktemp -d --tmpdir`; echo "#include <stdio.h>" > $$d/bitness.cpp; echo "int main(){ printf(\"%d\",(int)sizeof(void*)); return 0; }" >> $$d/bitness.cpp; $(CXX) $$d/bitness.cpp $(LDFLAGS) -o $$d/bitness$(EXE); $$d/bitness$(EXE); rm -rf $$d)
 endif
 ifeq ($(CXX_TARGET_BITNESS),8)
	OUTDIR = ./x64
 else
	OUTDIR = ./Win32
 endif
 # See https://github.com/msys2/MINGW-packages/issues/8503 about how -no-pie seems to fix the reported issue of empty gmon.out output.
 # This works with the full MINGW environment (as it comes with GNU Octave on Windows), but not when cross-compiling, alas.
 PROFILINGFLAGS += -no-pie
else
 EXE = 
 OUTDIR = ./$(OS)
 LDFLAGS += -ldl
 SOLDFLAGS += -fPIC
endif

SOFLAGS = -Wl,-rpath=. -L$(OUTDIR) -l:$(TARGET).$(XLL_EXT)

SWIG_VERSION = $(eval SWIG_VERSION := $$(shell $(SWIG) -version 2>&1 | sed -n '/Version/{ s/.* //g; p }'))$(SWIG_VERSION)

major_minor_micro_to_long = $(shell echo $(1) | awk -F . '{ printf "%03d%03d%03d",$$1,$$2,$$3 }' )

BPROF_VERSION = $(eval BPROF_VERSION := $$(shell $(BPROF) -V 2>/dev/null))$(BPROF_VERSION)
REQUIRED_BPROF_VERSION = 2.1.1
ifneq ($(BPROF_VERSION)),)
 BPROF_OK = $(eval BPROF_OK := $$(shell [ $(call major_minor_micro_to_long,$(BPROF_VERSION)) -ge $(call major_minor_micro_to_long,$(REQUIRED_BPROF_VERSION)) ] && echo TRUE || echo FALSE))$(BPROF_OK)
 ifeq ($(BPROF_OK),TRUE)
  BMON_OBJ := $(shell dirname `which "$(BPROF)" 2>/dev/null` 2>/dev/null)/../lib/bmon.o
 endif
endif

ifeq ($(origin CXX),default)
 OCTAVE_VERSION = $(shell octave --eval 'disp(version())' 2>/dev/null)
 GNUPLOT_VERSION = $(shell gnuplot -e 'print GPVAL_VERSION' 2>/dev/null)
 # Only include Octave targets and test when *not* cross-compiling.
 # $(info origin SWIG is $(origin SWIG))
 ifneq ($(OCTAVE_VERSION),)
  # $(info Octave version is $(OCTAVE_VERSION))
  OCTAVE_TARGETS = octave
  OCTAVE_TESTS = octave_test
  ifeq ($(shell [ $(call major_minor_micro_to_long,$(SWIG_VERSION)) -ge $(call major_minor_micro_to_long,$(MIN_SWIG_VERSION_FOR_OCTAVE)) ] && echo TRUE),TRUE)
   OCTAVE_TARGETS += octave_via_swig
   OCTAVE_TESTS   += octave_test_via_swig
   TEST_OUTPUTS   += octave_tests_output
  endif
 endif
 GNUPLOT_VERSION = $(shell gnuplot -e 'set print "-"; print GPVAL_VERSION' 2>/dev/null)
 # $(info Gnuplot version is $(GNUPLOT_VERSION))
 # Only include gnuplot plugin target and test when *not* cross-compiling.
 ifneq ($(GNUPLOT_VERSION),)
  MIN_GNUPLOT_VERSION = 5.0
  ifeq ($(shell [ $(call major_minor_micro_to_long,$(GNUPLOT_VERSION)) -ge $(call major_minor_micro_to_long,$(MIN_GNUPLOT_VERSION)) ] && echo TRUE),TRUE)
   GNUPLOT_TARGETS = gnuplot
   GNUPLOT_TESTS   = gnuplot_test
   TEST_OUTPUTS   += gnuplot_tests_output
   DEMOS          += gnuplot_demo
  endif
 endif
 # Only include profiling targets when *not* cross-compiling.
 PROFILING_TARGETS = profiling
 ifeq ($(BPROF_OK),TRUE)
  PROFILING_TARGETS += bprofiling
 endif
 # Only include Python targets and tests when *not* cross-compiling.
 ifeq ($(origin PYTHON),undefined)
  # Attempt to autodetect python 3, or whichever is the highest available number.
  get_python_version_as_long = $(call major_minor_micro_to_long,$(shell $(1) --version 2>/dev/null | sed 's/.* //'))
  DETECT_PYTHON = $(shell py_ver=$(call get_python_version_as_long,python); py3_ver=$(call get_python_version_as_long,$(ALTERNATIVE_PYTHON_NAME)); [ $$py_ver -ge $$py3_ver -a $$py_ver -gt 0 ] && echo python || { [ $$py3_ver -gt 0 ] && echo $(ALTERNATIVE_PYTHON_NAME); })
  # Windows versions of Python, if invoked as 'py', can show the strangest of errors, e.g., 'child_copy: cygheap read copy failed, [...], Win32 error 6', unless we resolve to the executable that is actually to be invoked.
  PYTHON = $(eval PYTHON := $$(subst \,/,$$(shell $$(DETECT_PYTHON) -c 'import sys; print(sys.executable)')))$(PYTHON)
 endif
 ifneq ($(PYTHON),)
  # $(info Python is $(PYTHON))
  PYTHON_VERSION = $(shell "$(PYTHON)" --version | sed 's/.* //')
  # $(info Python version is $(PYTHON_VERSION))
  PYTHON_TARGETS = python
  PYTHON_TESTS  = python_test
  ifeq ($(shell [ $(call major_minor_micro_to_long,$(SWIG_VERSION)) -ge $(call major_minor_micro_to_long,$(MIN_SWIG_VERSION_FOR_PYTHON)) ] && echo TRUE),TRUE)
   PYTHON_TARGETS += python_wheel
   PYTHON_TESTS   += python_wheel_test
  endif  
  TEST_OUTPUTS   += python_tests_output
  # $(info Python major version is $(firstword $(subst ., ,$(PYTHON_VERSION))))
  PYTHON_INCLUDE = $(eval PYTHON_INCLUDE := $$(subst \,/,$$(shell "$(PYTHON)" -c "import sysconfig; print(sysconfig.get_paths()['include'])")))$(PYTHON_INCLUDE)
  # $(info PYTHON_INCLUDE is $(PYTHON_INCLUDE))
  SWIG_PY3_FLAG = $(eval SWIG_PY3_FLAG := $$(shell [ $(firstword $(subst ., ,$(PYTHON_VERSION))) -ge 3 -a $(call major_minor_micro_to_long,$(SWIG_VERSION)) -lt $(call major_minor_micro_to_long,$(MIN_SWIG_VERSION_FOR_OCTAVE)) ] && echo '-py3'))$(SWIG_PY3_FLAG)
  # $(info SWIG_PY3_FLAG is $(SWIG_PY3_FLAG))
  ifeq ($(XLL_EXT),xll)
   # This is a heuristic: under Windows, any required run time dlls are typically right next to the main executable, and the flag needed for linking is typically something like -lpython311.
   DETECT_PYLDFLAGS = -L"$(subst \,/,$(shell '$(PYTHON)' -c 'import os,sys; print(os.path.dirname(os.path.realpath(sys.executable)))'))" -lpython$(shell echo $(PYTHON_VERSION) | sed 's/[.]//;s/[.].*//')
   PYLDFLAGS = $(eval PYLDFLAGS := $$(DETECT_PYLDFLAGS))$(PYLDFLAGS)
   # $(info PYLDFLAGS is $(PYLDFLAGS))
  endif
 endif
endif

CXX_SUPPORTS_UTF8_IN_CODE = $(shell d=`mktemp -d --tmpdir`; f=utf8_code_check; { echo "int main(int ν, char** h₃){ int h²=ν-1; return h²; }" > $$d/$$f.cpp; $(CXX)  $(CXXFLAGS) $$d/$$f.cpp $(LDFLAGS) -o $$d/$$f$(EXE); $$d/$$f$(EXE); } 2>/dev/null && echo 1; rm -rf $$d)

ifeq ($(CXX_SUPPORTS_UTF8_IN_CODE),)
 LETS_BE_RATIONAL_CPP = $(OUTDIR)/lets_be_rational_without_utf8.cpp
else
 LETS_BE_RATIONAL_CPP = lets_be_rational.cpp
endif

REVISION = $(eval REVISION := $$(shell sed -n '/REVISION/{ s/.* //g; p }' version.h))$(REVISION)

HEADERS := lets_be_rational.h dllmain.h normaldistribution.h rationalcubic.h version.h XLCall.h XLFunctions.h XLOper.h
SOURCES := erf_cody.cpp $(LETS_BE_RATIONAL_CPP) dllmain.cpp normaldistribution.cpp rationalcubic.cpp XLFunctions.cpp XLOper.cpp

HEADERS += 
SOURCES += 

.SECONDEXPANSION:

$(OUTDIR):
	mkdir -p $@

$(OUTDIR)/lets_be_rational_without_utf8.cpp:	lets_be_rational.cpp Makefile | $(OUTDIR)
	sed '\:^[ \t]*//:!{ s/ν/nu/g;s/h²/h_sqr/g;s/t²/t_sqr/g;s/h₂/hh2/g;s/h₃/hh3/g;s/h₄/hh4/g;s/q₁/q1/g;s/q₂/q2/g;s/θ/theta/g;s/𝛽/beta/g;s/sₗ/s_l/g;s/bₗ/b_l/g;s/μ/mu/g;s/λ/lambda/g;s/rₗₗ/r_ll/g;s/rₗₘ/r_lm/g;s/vₗ/v_l/g;s/sᵤ/s_u/g;s/bᵤ/b_u/g;s/rᵤₘ/r_um/g;s/vᵤ/v_u/g;s/rᵤᵤ/r_uu/g;s/x²/x_sqr/g;s/s³/s_cube/g; }' $<  > $@

$(OUTDIR)/$(TARGET).$(XLL_EXT):	Makefile $(HEADERS) $(SOURCES) | $(OUTDIR)
	$(CXX) $(CXXFLAGS) $(DEFFLAGS) $(INCFLAGS) $(SOURCES) $(LDFLAGS) $(SOLDFLAGS) -o $@ -s

DLL:	$(OUTDIR)/$(TARGET).$(XLL_EXT)

$(OUTDIR)/$(target)_timing$(EXE):	lets_be_rational_timing.cpp $(OUTDIR)/$(TARGET).$(XLL_EXT) | $(OUTDIR)
	$(CXX) $(CXXFLAGS) $(DEFFLAGS) $(INCFLAGS) lets_be_rational_timing.cpp $(LIBFLAGS) $(LDFLAGS) $(SOFLAGS) -o $@ -s

timing_exe:	$(OUTDIR)/$(target)_timing$(EXE)

TIMING_ARGS = 2097152  0 1E-14 1E-8 1E-4 0.001 0.01 0.05 0.1 0.25 0.5 1 2 4 8 16 32 64 128 256 512

# See https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html for special make variables such as $(^F).

timing: $(OUTDIR)/$(target)_timing$(EXE) | $(OUTDIR)
	cd $(OUTDIR); ./$(^F)  $(TIMING_ARGS) | tee ./$(target)_$@_output.txt

$(OUTDIR)/$(target)_profiling$(EXE):	Makefile lets_be_rational_timing.cpp $(HEADERS) $(SOURCES) | $(OUTDIR)
	$(CXX) $(PROFILINGFLAGS) $(CXXFLAGS) $(DEFFLAGS) $(INCFLAGS) $(SOURCES) lets_be_rational_timing.cpp $(LIBFLAGS) $(LDFLAGS) -o $@

profiling_exe:	$(OUTDIR)/$(target)_profiling$(EXE)

profiling: $(OUTDIR)/$(target)_$$@$(EXE) | $(OUTDIR)
	cd $(OUTDIR); rm -f *.gcda; ./$(^F)  $(TIMING_ARGS) | tee ./$(target)_$@_output.txt
	grep -v ^# $(OUTDIR)/$(target)_$@_output.txt | uniq > $(OUTDIR)/$@_analysis.txt
	cd $(OUTDIR); $(CROSS_PLATFORM_PREFIX)gprof $(^F) gmon.out >> $@_analysis.txt
	@echo; echo The $@ analysis results are:-
	@head -n 25 $(OUTDIR)/$@_analysis.txt
	@echo ; echo ...; echo; echo Please inspect file $(OUTDIR)/$@_analysis.txt for the full $@ analysis.; echo

bprof_check:	Makefile
	@echo
ifeq ($(BPROF_VERSION),)
	@echo "You dont't have bprof."
else
	@echo Your bprof version is $(BPROF_VERSION).
endif
	@echo
ifneq ($(BPROF_OK),TRUE)
	@echo "The minimum required BPROF version is "$(REQUIRED_BPROF_VERSION).; echo; echo You can try specifying the bprof executable to use on the make invocation command line via 'BPROF=<path_to_bprof>'.; echo; false;
endif

$(OUTDIR)/$(target)_bprofiling$(EXE):	Makefile bprof_check lets_be_rational_timing.cpp $(HEADERS) $(SOURCES) | $(OUTDIR)
	$(CXX) $(BPROFILINGFLAGS) $(CXXFLAGS) $(DEFFLAGS) $(INCFLAGS) $(SOURCES) lets_be_rational_timing.cpp $(BMON_OBJ) $(LIBFLAGS) $(LDFLAGS) -o $@

bprofiling_exe:	$(OUTDIR)/$(target)_$$@$(EXE)

bprofiling: $(OUTDIR)/$(target)_$$@$(EXE) | $(OUTDIR)
	cd $(OUTDIR); rm -f  bmon.out*; BPROF_FREQUENCY=10000 BPROF_NO_AUTOSTART=1 ./$(^F)  $(TIMING_ARGS) | tee ./$(target)_$@_output.txt
	grep -v ^# $(OUTDIR)/$(target)_$@_output.txt | uniq > $(OUTDIR)/$@_analysis.txt
	cd $(OUTDIR); rm -f *.bprof; $(BPROF) -d .. -c $(^F) bmon.out.0 | sed 's|\(in source file \)[^ \t]*/|\1|' >> $@_analysis.txt
	cp -avf $(OUTDIR)/$@_analysis.txt $(OUTDIR)/$@_sorted_analysis.txt
	cd $(OUTDIR); for f in *.bprof; do echo; echo $$f; grep '%]' "$$f"               ; done >> $@_analysis.txt
	cd $(OUTDIR); for f in *.bprof; do echo; echo $$f; grep '%]' "$$f" | sort -shrk2 ; done >> $@_sorted_analysis.txt
	@echo; echo The $@ analysis results are:-
	@head -n 20 $(OUTDIR)/$@_analysis.txt
	@echo ; echo ...; echo; echo Please inspect file $(OUTDIR)/$@_analysis.txt or $(OUTDIR)/$@_sorted_analysis.txt for the full $@ analysis.; echo

ifeq ($(origin SWIG),command line)
# You can enforce rebuilding of lets_be_rational_in_octave_via_swig.cpp when SWIG was specified on the command line.
#.PHONY:	lets_be_rational_in_octave_via_swig.cpp
endif

swig_check:	Makefile
	@echo
ifeq ($(SWIG_VERSION),)
	@echo "You dont't have SWIG."
else
	@echo Your SWIG Version is $(SWIG_VERSION).
endif
	@echo
	@[ $(call major_minor_micro_to_long,$(SWIG_VERSION)) -ge $(call major_minor_micro_to_long,$(REQUIRED_SWIG_VERSION)) ] || { echo "The minimum required SWIG version is "$(REQUIRED_SWIG_VERSION).; echo; echo You can try specifying the swig executable to use on the make invocation command line via 'SWIG=<path_to_swig>'.; echo; false; }

lets_be_rational_in_octave_via_swig.cpp:	LetsBeRational.i $(HEADERS) Makefile | $(OUTDIR)
	$(MAKE) REQUIRED_SWIG_VERSION=$(MIN_SWIG_VERSION_FOR_OCTAVE) swig_check
	$(SWIG) -c++ -octave -module $(target)_via_swig -I"patched SWIG files" -o $@ $<
	@echo

lets_be_rational_in_python_via_swig.cpp $(target).py:	LetsBeRational.i $(HEADERS) Makefile | $(OUTDIR)
	$(MAKE) REQUIRED_SWIG_VERSION=$(MIN_SWIG_VERSION_FOR_PYTHON) swig_check
# See https://www.swig.org/Doc4.0/Python.html#Python_commandline:
#  -O  Enable the following optimization options: -fastdispatch -fastproxy -fvirtual
	$(SWIG) -c++ -python -module $(target) $(SWIG_PY3_FLAG) -O -o lets_be_rational_in_python_via_swig.cpp $<

$(OUTDIR)/$(target).py:	$(target).py | $(OUTDIR)
	cp -vf $< $(OUTDIR)

$(OUTDIR)/_$(target).$(PY_DLL_EXT):	lets_be_rational_in_python_via_swig.cpp $(OUTDIR)/$(TARGET).$(XLL_EXT) $(HEADERS) | $(OUTDIR)
	$(CXX) $(CXXFLAGS) $(DEFFLAGS) $(INCFLAGS) -I"$(PYTHON_INCLUDE)" $<  $(LIBFLAGS) $(LDFLAGS) $(SOLDFLAGS) $(PYLDFLAGS) -o $@ $(SOFLAGS) -s

python:	$(OUTDIR)/$(target).py $(OUTDIR)/_$(target).$(PY_DLL_EXT)

$(OUTDIR)/wheel_name:	setup.py $(OUTDIR)/$(target).py $(OUTDIR)/_$(target).$(PY_DLL_EXT) $(OUTDIR)/$(TARGET).$(XLL_EXT) | $(OUTDIR)
	"$(PYTHON)" -m pip install pip build wheel setuptools --upgrade --quiet --no-warn-script-location
	cd $(OUTDIR); rm -rf *.whl wheel_name; REVISION=$(REVISION) MODULE_NAME=$(target) WHEEL_MAIN_BINARY=_$(target).$(PY_DLL_EXT) WHEEL_DEPENDENCY_BINARIES=$(TARGET).$(XLL_EXT) WHEEL_OPTIONAL_BINARIES="$(WHEEL_OPTIONAL_BINARIES)" "$(PYTHON)" ../setup.py bdist_wheel --dist-dir .; echo *.whl > wheel_name;
	@echo Generated wheel file $(OUTDIR)/`cat $(OUTDIR)/wheel_name`.
	@rm -rf $(OUTDIR)/$(TARGET).egg-info $(OUTDIR)/__pycache__ $(OUTDIR)/build

wheel python_wheel:	$(OUTDIR)/wheel_name

PYTHON_TEST_COMMAND = 'from $(target) import *; sigma=ImpliedBlackVolatility(Black(1,1,1,1,1),1,1,1,1); print("\tImpliedBlackVolatility(Black(1,1,1,1,1),1,1,1,1) = %g\n" % sigma)'

python_test:	$(OUTDIR)/$(target).py $(OUTDIR)/_$(target).$(PY_DLL_EXT) $(OUTDIR)/$(TARGET).$(XLL_EXT) 
	cd $(if $(filter undefined,$(origin PYTHON_TEST_RUN_DIR)),$(OUTDIR),$(PYTHON_TEST_RUN_DIR)); "$(PYTHON)" -c $(PYTHON_TEST_COMMAND)

python_wheel_test:	python_wheel
	cd $(OUTDIR); "$(PYTHON)" -m pip install --force-reinstall `cat wheel_name` --quiet --no-warn-script-location
	d=`mktemp -d --tmpdir`;	$(MAKE) PYTHON_TEST_RUN_DIR=$$d python_test; rm -rf $$d
	"$(PYTHON)" -m pip uninstall $(target) -y --quiet

$(OUTDIR)/%.oct:	$$(subst $(target),lets_be_rational_in_octave,$$(*F)).cpp $(OUTDIR)/$(TARGET).$(XLL_EXT) $(HEADERS) | $(OUTDIR)
	mkoctfile -DTARGET=$(target) $< -o $@ $(SOFLAGS) -s

octave octave_via_swig:	$(OUTDIR)/$(target)$$(subst octave,,$$(@F)).oct | $(OUTDIR)

OCTAVE_TEST_COMMAND = sigma=ImpliedBlackVolatility(Black(1,1,1,1,1),1,1,1,1); printf("\tImpliedBlackVolatility(Black(1,1,1,1,1),1,1,1,1) = %g\n\n",sigma); assert( abs(sigma-1) < 1E-15 )

octave_test octave_test_via_swig:	$$(subst _test,,$$(@F))
	cd $(OUTDIR); octave --eval 'printf("\nThis is GNU Octave version %s.\n",version()); $(target)$(subst octave,,$<); $(OCTAVE_TEST_COMMAND)'

octave_tests:	$(OCTAVE_TESTS)

$(OUTDIR)/octave_tests_output.txt:	$(OCTAVE_TARGETS)
	$(MAKE) octave_tests | tee $@

octave_tests_output:	$(OUTDIR)/octave_tests_output.txt

python_tests:	$(PYTHON_TESTS)

$(OUTDIR)/python_tests_output.txt:	$(PYTHON_TARGETS)
	$(MAKE) python_tests | tee $@

python_tests_output:	$(OUTDIR)/python_tests_output.txt

$(OUTDIR)/letsberational.gnuplot:	letsberational.gnuplot Makefile | $(OUTDIR)
	cp -vf $< $@

$(OUTDIR)/letsberational_in_gnuplot.$(DLL_EXT):	lets_be_rational_in_gnuplot.cpp $(OUTDIR)/$(TARGET).$(XLL_EXT) $(HEADERS) | $(OUTDIR)
	$(CXX) $(CXXFLAGS) $(DEFFLAGS) $(INCFLAGS) $<  $(LIBFLAGS) $(LDFLAGS) $(SOLDFLAGS) -o $@ $(SOFLAGS) -s

gnuplot:	$(OUTDIR)/letsberational_in_gnuplot.$(DLL_EXT) $(OUTDIR)/letsberational.gnuplot

GNUPLOT_TEST_COMMAND = sigma=ImpliedBlackVolatility(Black(1,1,1,1,1),1,1,1,1); print "\tImpliedBlackVolatility(Black(1,1,1,1,1),1,1,1,1) = ",sigma,"\n"

gnuplot_test:	gnuplot
	cd $(OUTDIR); gnuplot -e 'set print "-"; print "\nThis is gnuplot version ",GPVAL_VERSION,"."; load "letsberational.gnuplot"; $(GNUPLOT_TEST_COMMAND)'

gnuplot_tests:	$(GNUPLOT_TESTS)

$(OUTDIR)/gnuplot_tests_output.txt:	$(GNUPLOT_TARGETS)
	$(MAKE) gnuplot_tests | tee $@

gnuplot_tests_output:	$(OUTDIR)/gnuplot_tests_output.txt

gnuplot_demo:	letsberational_demo.gnuplot gnuplot
	cd $(OUTDIR); gnuplot ../$<

clean:
	rm -f *.obj *.o *.exp *.lib *.pdb *.idb *.ndb *.ncb *~ */*.gcda

veryclean:	clean
	rm -f */$(target)_profiling */$(target)_profiling.exe
	rm -rf __pycache__ */__pycache__ */wheel_name

sterile:	veryclean
	rm -rf $(OUTDIR)

# See https://stackoverflow.com/questions/16467718/how-to-print-out-a-variable-in-makefile. Invoke as make print-VARIABLE.
print-% : ; $(info $* is a $(flavor $*) variable set to [$($*)]) @true

Default:	$(OUTDIR)/$(TARGET).$(XLL_EXT) $(OUTDIR)/$(target)_timing$(EXE) $(OCTAVE_TARGETS) $(PYTHON_TARGETS) $(GNUPLOT_TARGETS) | $(OUTDIR)

tests:	$(OCTAVE_TESTS) $(PYTHON_TESTS) $(GNUPLOT_TESTS)

all:	Default tests timing $(PROFILING_TARGETS) veryclean $(TEST_OUTPUTS) $(DEMOS)

cross_platforms:
	$(MAKE) CXX=i686-w64-mingw32-g++    all
	$(MAKE) CXX=x86_64-w64-mingw32-g++  all

everything:	clean cross_platforms all clean

update_from_msvc_build:
	@mkdir -p x64
	@cp -v build/x64/Release/$(TARGET).xll x64/
	@cp -v build/x64/Release/$(target)_timing.exe x64/
	@mkdir -p Win32
	@cp -v build/Win32/Release/$(TARGET).xll Win32/
	@cp -v build/Win32/Release/$(target)_timing.exe Win32/

archive:
	cd .. ; mkdir -p 7z; ARCHIVE=7z/"$(TARGET)".$(shell date +%F).7z; mv -f $$ARCHIVE /tmp; rm -f $$ARCHIVE; 7z a -l -t7z -m0=lzma -xr\!.svn -xr\!.vs -xr\!build -xr\!__pycache__ $$ARCHIVE "$(CURDIRBASE)"
