#!/usr/bin/env bash

cd  ../../build
if ! make -j ${NUMPROC}; then
  exit
fi