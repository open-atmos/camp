#!/usr/bin/env bash
set -e
./compile.cvode-3.4-alpha.power9.sh || exit 1
./make.camp.power9.sh