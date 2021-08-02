/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 *
 * Header file for sub_model_solver.c
 *
 */
/** \file
 * \brief Header file for abstract sub model functions
 */
#ifndef SUB_MODEL_SOLVER_H
#define SUB_MODEL_SOLVER_H
#include "camp_common.h"

/** Public sub model functions **/

/* Solver functions */
void sub_model_get_used_jac_elem(ModelData *model_data, Jacobian *jac);
void sub_model_set_jac_map(ModelData *model_data, Jacobian jac);
void sub_model_update_ids(ModelData *model_data, int *deriv_ids, Jacobian jac);
void sub_model_update_env_state(ModelData *model_data);
void sub_model_calculate(ModelData *model_data);
#ifdef CAMP_USE_SUNDIALS
void sub_model_get_jac_contrib(ModelData *model_data, double *J_data,
                               realtype time_step);
#endif
void sub_model_print_data(void *solver_data);

/* Setup functions */
void sub_model_add_condensed_data(int sub_model_type, int n_int_param,
                                  int n_float_param, int n_env_param,
                                  int *int_param, double *float_param,
                                  void *solver_data);
void sub_model_update_data(int cell_id, int *sub_model_id,
                           int update_sub_model_type, void *update_data,
                           void *solver_data);

#endif
