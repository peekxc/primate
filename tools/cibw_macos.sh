#!/usr/bin/env bash

brew install libomp llvm openblas
export CC=/usr/local/opt/llvm/bin/clang
export CXX=/usr/local/opt/llvm/bin/clang++
export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
export CFLAGS="$CFLAGS -I$PREFIX/include"
export CXXFLAGS="$CXXFLAGS -I$PREFIX/include"
export LDFLAGS="$LDFLAGS -Wl,-rpath,$PREFIX/lib -L$PREFIX/lib -lomp"

if [[ $(uname -m) == "arm64" && "$CIBW_BUILD" == "cp38-macosx_arm64" ]]; then
  # Enables native building and testing for macosx arm on Python 3.8. For details see:
  # https://cibuildwheel.readthedocs.io/en/stable/faq/#macos-building-cpython-38-wheels-on-arm64
  curl -o /tmp/Python38.pkg https://www.python.org/ftp/python/3.8.10/python-3.8.10-macos11.pkg
  sudo installer -pkg /tmp/Python38.pkg -target /
  sh "/Applications/Python 3.8/Install Certificates.command"
fi

echo CXX VARIABLE: 
echo $CXX