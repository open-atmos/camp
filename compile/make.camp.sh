#!/usr/bin/env bash
set -e
scriptdir="$(dirname "$0")"
cd "$scriptdir"
cd ../build
make -j 8
