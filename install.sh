#!/usr/bin/env bash
set -e

scriptdir="$(dirname "$0")"
cd "$scriptdir"
if [ ! -d build ]; then
  ./compile/compile.libs.camp.sh
else
  echo "ERROR: CAMP build/ folder already exists. It is already installed? If want to install again, remove build/ folder"
fi