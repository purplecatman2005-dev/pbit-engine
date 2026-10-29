@echo off
rem build.bat - build with g++ (MinGW-w64)
setlocal
cd /d "%~dp0"
set FLAGS=-O2 -std=c++23 -march=native -Wall -Wextra -static
echo [1/2] demo.exe
g++ %FLAGS% demo.cpp -o demo.exe || exit /b 1
echo [2/2] test_pbit.exe
g++ %FLAGS% test_pbit.cpp -o test_pbit.exe || exit /b 1
echo done.
