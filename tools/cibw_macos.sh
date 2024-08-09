#!/usr/bin/env bash

rm -rf /usr/local/bin/2to3/*
brew install --force libomp llvm openblas

# export CC=/usr/local/opt/llvm/bin/clang
# export CXX=/usr/local/opt/llvm/bin/clang++
export CC=/opt/homebrew/opt/llvm/bin/clang
export CXX=/opt/homebrew/opt/llvm/bin/clang++
export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
export CFLAGS="$CFLAGS -Wno-implicit-function-declaration -I$PREFIX/include"
export CXXFLAGS="$CXXFLAGS -I$PREFIX/include"
export LDFLAGS="$LDFLAGS -Wl,-rpath,$PREFIX/lib -L$PREFIX/lib -lomp"
export LDFLAGS="$LDFLAGS -Wl,-S -Wl,-rpath,$PREFIX/lib -L$PREFIX/lib -lomp"
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export LDFLAGS="$LDFLAGS -L/opt/homebrew/opt/llvm/lib"
export CXXFLAGS="$CXXFLAGS -I/opt/homebrew/opt/llvm/include"

if [[ $(uname -m) == "arm64" && "$CIBW_BUILD" == "cp38-macosx_arm64" ]]; then
  # Enables native building and testing for macosx arm on Python 3.8. For details see:
  # https://cibuildwheel.readthedocs.io/en/stable/faq/#macos-building-cpython-38-wheels-on-arm64
  curl -o /tmp/Python38.pkg https://www.python.org/ftp/python/3.8.10/python-3.8.10-macos11.pkg
  sudo installer -pkg /tmp/Python38.pkg -target /
  sh "/Applications/Python 3.8/Install Certificates.command"
fi

echo CXX VARIABLE: $CXX
clang --version


#  # Make sure to use a libomp version binary compatible with the oldest
#   # supported version of the macos SDK as libomp will be vendored into
#   # the scikit-image wheels for macos. The list of binaries are in
#   # https://packages.macports.org/libomp/.  Currently, the oldest
#   # supported macos version is: High Sierra / 10.13. When upgrading
#   # this, be sure to update the MACOSX_DEPLOYMENT_TARGET environment
#   # variable accordingly. Note that Darwin_17 == High Sierra / 10.13.
#   #
#   # We need to set both MACOS_DEPLOYMENT_TARGET and MACOSX_DEPLOYMENT_TARGET
#   # until there is a new release with this commit:
#   # https://github.com/mesonbuild/meson-python/pull/309
#   if [[ "$CIBW_ARCHS_MACOS" == arm64 ]]; then
#       # SciPy requires 12.0 on arm to prevent kernel panics
#       # https://github.com/scipy/scipy/issues/14688
#       # so being conservative, we just do the same here
#       export MACOSX_DEPLOYMENT_TARGET=12.0
#       export MACOS_DEPLOYMENT_TARGET=12.0
#       OPENMP_URL="https://anaconda.org/conda-forge/llvm-openmp/11.1.0/download/osx-arm64/llvm-openmp-11.1.0-hf3c4609_1.tar.bz2"
#   else
#       export MACOSX_DEPLOYMENT_TARGET=10.9
#       export MACOS_DEPLOYMENT_TARGET=10.9
#       OPENMP_URL="https://anaconda.org/conda-forge/llvm-openmp/11.1.0/download/osx-64/llvm-openmp-11.1.0-hda6cdc1_1.tar.bz2"
#   fi
#   echo MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET}
#   echo MACOS_DEPLOYMENT_TARGET=${MACOS_DEPLOYMENT_TARGET}

#   # use conda to install llvm-openmp
#   # Note that we do NOT activate the conda environment, we just add the
#   # library install path to CFLAGS/CXXFLAGS/LDFLAGS below.
#   sudo conda create -n build $OPENMP_URL
#   PREFIX="/usr/local/miniconda/envs/build"
#   export CC=/usr/bin/clang
#   export CXX=/usr/bin/clang++
