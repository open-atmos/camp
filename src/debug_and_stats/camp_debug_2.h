//
// Created by cguzman on 10/24/19.
//

#ifndef CAMP_DEBUG_2_H
#define CAMP_DEBUG_2_H

#include "../camp_common.h"

#ifdef CSR_MATRIX
void swapCSC_CSR(int n_row, int n_col, int* Ap, int* Aj, double* Ax, int* Bp, int* Bi, double* Bx);
#endif
void check_isnand(double *x, int len, const char *s);
void print_int(int *x, int len, const char *s);
void print_double(double *x, int len, const char *s);
int compare_doubles(double *x, double *y, int len, const char *s);
void get_camp_config_variables(SolverData *sd);
void export_counters_open(SolverData *sd);


#endif  // CAMP_DEBUG_2_H