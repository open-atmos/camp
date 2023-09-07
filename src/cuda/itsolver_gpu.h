/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#ifndef ITSOLVERGPU_H
#define ITSOLVERGPU_H

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<iostream>
#include"libsolv.h"
#include<cuda.h>
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>

extern "C" {
#include "../camp_solver.h"
}

#define ONECELL 1
#define MULTICELLS 2
#define  BLOCKCELLSN 3
#define BLOCKCELLS1 4
#define BLOCKCELLSNHALF 5
#define BCG_MAXIT 1000
#define BCG_TOLMAX 1.0E-30


#endif