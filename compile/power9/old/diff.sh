#!/usr/bin/env bash

cat log_cpu.txt | sed -e/"[100%] Built target mock_monarch"/\{ -e:1 -en\;b1 -e\} -ed > diff.txt
#diff log_cpu.txt log.txt | grep -E "^\+" > diff.txt
