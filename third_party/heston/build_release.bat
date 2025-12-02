@echo off
set "SRC_DIR=C:\dev\git\phd\third_party\cui_et_al_code"
set "BUILD_DIR=%SRC_DIR%\build"

cmake -S "%SRC_DIR%" -B "%BUILD_DIR%" -A x64
cmake --build "%BUILD_DIR%" --config Release --verbose --target heston -- /m 

echo Done.