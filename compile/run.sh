#!/usr/bin/env bash
set -e

scriptdir="$(dirname "$0")"
cd "$scriptdir"
if [ ! -d ../build ]; then
  ./compile.libs.camp.sh
fi
source load.modules.camp.sh
cd ../test/monarch
./run.sh