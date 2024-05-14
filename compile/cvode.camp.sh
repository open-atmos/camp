#!/usr/bin/env bash
set -e
./compile.cvode-3.4-alpha.sh || exit 1
./make.camp.power9.sh