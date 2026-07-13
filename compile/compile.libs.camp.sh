#!/usr/bin/env bash
set -e

scriptdir="$(dirname "$0")"
cd "$scriptdir"
./get.libs.sh
./compile.json-fortran-6.1.0.sh
./compile.suiteSparse.sh
./compile.cvode.sh
./compile.camp.sh