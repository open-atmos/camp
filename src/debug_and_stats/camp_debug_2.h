//
// Created by cguzman on 10/24/19.
//

#ifndef CAMP_DEBUG_2_H
#define CAMP_DEBUG_2_H

#include "../camp_common.h"


void check_isnand(double *x, int len, char *s);
void print_int(int *x, int len, char *s);
void print_double(double *x, int len, char *s);
void get_camp_config_variables(SolverData *sd);
void export_counters_open(SolverData *sd);


#endif  // CAMP_DEBUG_2_H
