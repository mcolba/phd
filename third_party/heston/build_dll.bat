@echo off
set SRC_DIR=%~dp0
set SRC_DIR=%SRC_DIR:~0,-1%
set BUILD_DIR=%SRC_DIR%\build

cmake -S "%SRC_DIR%" ^
    -B "%BUILD_DIR%" ^
    -G "Visual Studio 17 2022" ^
    -A x64 ^
    -DCMAKE_CXX_FLAGS_RELEASE="/O2 /Ob2 /DNDEBUG /arch:AVX2 /fp:precise" ^
    -DUSE_MKL=ON

cmake --build "%BUILD_DIR%" --config Debug --verbose
cmake --build "%BUILD_DIR%" --config Release --verbose
