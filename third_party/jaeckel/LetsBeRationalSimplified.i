// Simplified interface SWIG (https://www.swig.org/).

%module letsberational

#define DLL_EXPORT // Ignore Windows DLL decorations

%{
#include "../src/lets_be_rational.h"
%}

%include "../src/lets_be_rational.h"

// extern "C" const char* CPUName();
// extern "C" int Revision();