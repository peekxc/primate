#!/usr/bin/env bash

if [ -n "$(command -v yum)" ]; then
  cat /etc/*-release 
  # yum remove -y epel-release
  yum update -y 
  ulimit -n 4096
  yum install -y clang
  yum install -y openblas
  alias python=/opt/python/cp39-cp39/bin/python
  # yum install -y python3.9
  # yum install -y python39-devel
  # alias python=python3.9
elif [ -n "$(command -v apt-get)" ]; then 
  cat /etc/*-release
  sudo apt-get update -y
  sudo apt-get install -y clang libomp-dev
  sudo apt-get install -y libopenblas-dev
  sudo apt-get install -y python3-dev
  sudo wget https://apt.llvm.org/llvm.sh
  sudo chmod u+x llvm.sh 
  sudo ./llvm.sh 17 all 
elif [ -n "$(command -v apk)" ]; then 
  cat /etc/*-release
  apk update
  apk add clang # looks like Alpine only supports up to clang 10 ?
  # apk add openmp
  apk add openblas
  apk add python3-dev
  alias python=python3
  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  python get-pip.py
fi 