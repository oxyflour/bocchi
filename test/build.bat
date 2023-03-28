where cl || call "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
dir build || mkdir build
nvcc test/main.cu deps/lodepng/lodepng.cpp ^
  -g ^
  -I"C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.6.0\include" -I. ^
  --extended-lambda --expt-relaxed-constexpr --std=c++17 ^
  -o build/main.exe && ^
echo build\main.exe
