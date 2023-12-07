#!/usr/bin/env bash

# To avoid "LINK : fatal error LNK1158: cannot run 'rc.exe'"
# we explicitly add rc.exe to path using the method from:
# https://github.com/actions/virtual-environments/issues/294#issuecomment-588090582
# with additional -arch=x86 flag to vsdevcmd.bat
# https://github.com/actions/runner-images/issues/294
# function Invoke-VSDevEnvironment {
#   $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
#     $installationPath = & $vswhere -prerelease -legacy -latest -property installationPath
#     $Command = Join-Path $installationPath "Common7\Tools\vsdevcmd.bat"
#   & "${env:COMSPEC}" /s /c "`"$Command`" -arch=amd64 -no_logo && set" | Foreach-Object {
#         if ($_ -match '^([^=]+)=(.*)') {
#             [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2])
#         }
#     }
# }
# Invoke-VSDevEnvironment
# Get-Command rc.exe | Format-Table -AutoSize

choco install rtools -y --no-progress --force --version=4.0.0.20220206
set PATH=C:\rtools40\ucrt64\bin\;%PATH%
refreshenv