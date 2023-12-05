#!/usr/bin/env bash

choco install rtools -y --no-progress --force --version=4.0.0.20220206"
echo "c:\rtools40\ucrt64\bin;" >> $PATH
