#!/usr/bin/env bash
set -e
ln -rs -fL out ../../build/out
ln -rs -fL settings ../../build/settings
python checkGPU.py