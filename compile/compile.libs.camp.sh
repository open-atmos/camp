#!/usr/bin/env bash
set -e

scriptdir="$(dirname "$0")"
cd "$scriptdir"
./get.libs.sh
./compile.json-fortran-6.1.0.power9.sh
./compile.suiteSparse.power9.sh
./compile.cvode-3.4-alpha.power9.sh
./compile.camp.sh