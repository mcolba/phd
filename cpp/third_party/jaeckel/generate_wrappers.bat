@echo off
REM Batch file to generate SWIG wrappers for LetsBeRational
REM This script generates the Python wrapper files from the simplified SWIG interface

echo Generating SWIG wrappers for LetsBeRational...

REM Create output directory if it doesn't exist
if not exist "letsberational" mkdir letsberational

REM Run SWIG to generate the wrapper files
swig -c++ -python -outdir letsberational -o letsberational\LetsBeRational_wrap.cpp LetsBeRational_Simplified.i

if %ERRORLEVEL% neq 0 (
    echo ERROR: SWIG wrapper generation failed!
    pause
    exit /b 1
)

echo SWIG wrapper generation completed successfully!
echo Generated files:
echo   - letsberational\LetsBeRational_wrap.cpp
echo   - letsberational\letsberational.py

pause