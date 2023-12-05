#!/usr/bin/env bash

if [ -n "$(command -v yum)" ]; then 
  dng check-update
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
fi 