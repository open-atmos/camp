#!/bin/sh

# exit on error
set -e
# turn on command echoing
set -v

mkdir -p out

../../build/camp chamber.spec

# Now run ./2_process.sh to process the data
