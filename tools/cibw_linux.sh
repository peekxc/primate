#!/usr/bin/env bash

if [ -n "$(command -v yum)" ]; then 
  dnf check-update
  dnf update
  yum install llvm-toolset
  yum install openblas
  yum install -y python3.9
  yum install python39-devel
  alias python=python3.9
elif [ -n "$(command -v apt-get)" ]; then 
  apt-get update -y
  apt-get install -y clang libomp-dev
  apt-get update -y
  apt-get install -y libopenblas-dev
  apt-get install -y python3-dev
elif [ -n "$(command -v apk)" ]; then 
  apk update
  apk add clang # looks like Alpine only supports up to clang 10 ?
  # apk add openmp
  apk add openblas
  apk add python3-dev
  alias python=python3
  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  python get-pip.py
fi 