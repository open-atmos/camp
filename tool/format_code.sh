#!/bin/bash

# turn on command echoing
set -v

# make sure that the current directory is the camp root dir
cd "${0%/*}/.."
find src/. -iname *.h -o -iname *.c -o -iname *.cpp -o -iname *.hpp -o -iname *.cu \
    | xargs clang-format -style=file -i -fallback-style=none
#find src/. -iname "*.cu" | xargs clang-format -i