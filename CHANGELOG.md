# Changelog

All notable changes to this project will be documented in this file. This project _loosely_ adheres to [semantic versioning](https://semver.org/spec/v2.0.0.html).

## v0.4.0 [Unreleased]
- Removed ability to specify scalar-valued matrix functions 
- Added ability to specify vector-valued matrix functions (strictly more general!)
- Added transform() function to MatrixFunction class to allow preprocessing of vector inputs (e.g. for deflation)
<!-- - Fixed issue related to NaN's being generated from traces of diagonal matrices  -->
- Fixed crashing bug related to auto-detection of number of threads to launch when <= 0
- Preliminary (undocumented) support added for native matrix functions, paving the way for other FFI bindings, e.g. Numba or Cython
- Updated CI linux wheel to manylinux2014 for improved compatibility for other systems
- Revamped Hutch++ implementation

## v0.3.6
- Updated hutch to correctly account for different confidence levels when use_CLT=True 
- Added t-scores to CLT computation to discourage stopping too early
- Added numerical rank function 'numrank' to fledgling functional API 
- Added initial Hutch++ code 

## v0.3.5
- Added initial python-pnly XTrace implementation 
- Added unit tests to increase coverage on matrix function API 

## v0.3.4
- Added introductory theory pages to the docs
- Improved doc installation pages

## v0.3.3
- First PyPI release 