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

void init_export_state(SolverData *sd);
void export_state(SolverData *sd);
void join_export_state();
void init_export_stats();
void export_stats(int ntimers,int ncounters, int* counters, double* times);
void print_double(double *x, int len, const char *s);
void print_int(int *x, int len, const char *s);
void get_camp_config_variables(SolverData *sd);

#endif  // CAMP_DEBUG_2_H
