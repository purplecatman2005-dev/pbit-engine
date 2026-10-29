@echo off
rem build_qubo.bat - build QUBO solver tools with g++ (MinGW-w64)
setlocal
cd /d "%~dp0"
set FLAGS=-O2 -std=c++23 -march=native -Wall -Wextra -static
echo [1/2] qubo_portfolio.exe
g++ %FLAGS% qubo_portfolio.cpp -o qubo_portfolio.exe || exit /b 1
echo [2/2] test_qubo.exe
g++ %FLAGS% test_qubo.cpp -o test_qubo.exe || exit /b 1
echo done.
