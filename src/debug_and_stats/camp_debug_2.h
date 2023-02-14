/*
 * -----------------------------------------------------------------
 * Programmer(s): Christian G. Ruiz and Mario Acosta
 * -----------------------------------------------------------------
 * Copyright (C) 2022 Barcelona Supercomputing Center
 * SPDX-License-Identifier: MIT
 */

#ifndef CAMP_DEBUG_2_H
#define CAMP_DEBUG_2_H

#include "../camp_common.h"

#ifdef ENABLE_NETCDF
void cell_netcdf(SolverData *sd);
#endif

void check_isnand(double *x, int len, const char *s);
void print_int(int *x, int len, const char *s);
void print_double(double *x, int len, const char *s);
int compare_doubles(double *x, double *y, int len, const char *s);
void get_camp_config_variables(SolverData *sd);
void export_counters_open(SolverData *sd);


#endif  // CAMP_DEBUG_2_H
