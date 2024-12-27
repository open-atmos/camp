/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */
/** \file
 * \brief c function tests for the single particle aerosol representation with layers
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../test_common.h"
#include "../../src/aero_rep_solver.h"
#include "../../src/aero_reps.h"
#include "../../src/camp_common.h"

// index for the test aerosol representation
#define AERO_REP_IDX 0

// test computational particle
#define TEST_PARTICLE_1 2
#define TEST_PARTICLE_2 1

// number of computational particles in the test
#define N_COMP_PARTICLES 3

// number of aerosol phases per particle
#define NUM_AERO_PHASE 4

// index for the test phase (test-particle phase 2)
#define AERO_PHASE_IDX_1 ((TEST_PARTICLE_1-1)*NUM_AERO_PHASE+1)
#define AERO_PHASE_IDX_2 ((TEST_PARTICLE_2)*NUM_AERO_PHASE-1)

// number of Jacobian elements used for the test phase
#define N_JAC_ELEM 12

// Test concentrations (kg/m3)
#define CONC_wheat 1.0
#define CONC_water 2.0
#define CONC_salt 3.0
#define CONC_rasberry 4.0
#define CONC_honey 5.0
#define CONC_sugar 6.0
#define CONC_lemon 7.0
#define CONC_almonds 8.0
#define CONC_sugarB 9.0
#define CONC_wheatB 10.0
#define CONC_waterB 11.0
#define CONC_saltB 12.0

// Molecular weight (kg/mol) of test species (must match json file)
#define MW_wheat 1.0
#define MW_water 0.018
#define MW_salt 0.058
#define MW_rasberry 42.1
#define MW_honey 52.3
#define MW_sugar 623.2
#define MW_lemon 72.3
#define MV_almonds 72.3

// Density (kg/m3) of test species (must match json file)
#define DENSITY_wheat 1.0
#define DENSITY_water 1000.0
#define DENSITY_salt 2160.0
#define DENSITY_rasberry 4.0
#define DENSITY_honey 5.0
#define DENSITY_sugar 6.0
#define DENSITY_lemon 7.0
#define DENSITY_almonds 7.0

// Externally set properties
#define PART_NUM_CONC 1.23e3

/** \brief Test the effective radius function
 *
 * \param model_data Pointer to the model data
 * \param state Solver state
 */

#ifdef CAMP_USE_SUNDIALS

int test_effective_radius(ModelData * model_data, N_Vector state) {

  int ret_val = 0;
  double partial_deriv_1[N_JAC_ELEM+2];
  double partial_deriv_2[N_JAC_ELEM+2];
  double eff_rad_1 = -999.9;
  double eff_rad_2 = -999.9;

  for( int i = 0; i < N_JAC_ELEM+2; ++i ) partial_deriv_1[i] = 999.9;

  aero_rep_get_effective_radius__m(model_data, AERO_REP_IDX,
                                AERO_PHASE_IDX_1, &eff_rad_1, &(partial_deriv_1[1]));
  aero_rep_get_effective_radius__m(model_data, AERO_REP_IDX,
                                AERO_PHASE_IDX_2, &eff_rad_2, &(partial_deriv_2[1]));

  double volume_density = ( CONC_wheat / DENSITY_wheat +
                            CONC_water / DENSITY_water +
                            CONC_salt / DENSITY_salt +
                            CONC_rasberry / DENSITY_rasberry +
                            CONC_honey / DENSITY_honey +
                            CONC_sugar / DENSITY_sugar +
                            CONC_lemon / DENSITY_lemon +
                            CONC_almonds / DENSITY_almonds + 
                            CONC_sugarB / DENSITY_sugar +
                            CONC_wheatB / DENSITY_wheat +
                            CONC_waterB / DENSITY_water +
                            CONC_saltB / DENSITY_salt ); // volume density (m3/m3)
  
  printf("\neff_rad_1 : %f", eff_rad_1);
  printf("\neff_rad_2 : %f", eff_rad_2);
  double eff_rad_expected = pow( ( 3.0 / 4.0 / 3.14159265359 * volume_density ), 1.0/3.0 );
  printf("\neff_rad_expected %f", eff_rad_expected);
  ret_val += ASSERT_MSG(fabs(eff_rad_1-eff_rad_expected) < 1.0e-6*eff_rad_expected,
                        "Bad effective radius");
  ret_val += ASSERT_MSG(fabs(eff_rad_2-eff_rad_expected) < 1.0e-6*eff_rad_expected,
                        "Bad effective radius");

  ret_val += ASSERT_MSG(partial_deriv_1[0] = 999.9,
                        "Bad Jacobian (-1)");
  double d_eff_rad_dx = 1.0 / 4.0 / 3.14159265359 *
                        pow( 3.0 / 4.0 / 3.14159265359 * volume_density, -2.0/3.0 );

  ret_val += ASSERT_MSG(fabs(partial_deriv_1[1] - d_eff_rad_dx / DENSITY_wheat) <
                        1.0e-10 * partial_deriv_1[1], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[2] - d_eff_rad_dx / DENSITY_water) <
                        1.0e-10 * partial_deriv_1[2], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[3] - d_eff_rad_dx / DENSITY_salt) <
                        1.0e-10 * partial_deriv_1[3], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[4] - d_eff_rad_dx / DENSITY_rasberry) <
                        1.0e-10 * partial_deriv_1[4], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[5] - d_eff_rad_dx / DENSITY_honey) <
                        1.0e-10 * partial_deriv_1[5], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[6] - d_eff_rad_dx / DENSITY_sugar) <
                        1.0e-10 * partial_deriv_1[6], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[7] - d_eff_rad_dx / DENSITY_lemon) <
                        1.0e-10 * partial_deriv_1[7], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[8] - d_eff_rad_dx / DENSITY_almonds) <
                        1.0e-10 * partial_deriv_1[8], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[9] - d_eff_rad_dx / DENSITY_sugar) <
                        1.0e-10 * partial_deriv_1[9], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[10] - d_eff_rad_dx / DENSITY_wheat) <
                        1.0e-10 * partial_deriv_1[10], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[11] - d_eff_rad_dx / DENSITY_water) <
                        1.0e-10 * partial_deriv_1[11], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[12] - d_eff_rad_dx / DENSITY_salt) <
                        1.0e-10 * partial_deriv_1[12], "Bad Jacobian element");
  ret_val += ASSERT_MSG(partial_deriv_1[N_JAC_ELEM+1] = 999.9,
                        "Bad Jacobian (end+1)");

  return ret_val;
}

/** \brief Test the number concentration function
 *
 * \param model_data Pointer to the model data
 * \param state Solver state
 */

int test_number_concentration(ModelData * model_data, N_Vector state) {

  int ret_val = 0;
  double partial_deriv[N_JAC_ELEM+2];
  double num_conc = -999.9;
  
  for( int i = 0; i < N_JAC_ELEM+2; ++i ) partial_deriv[i] = 999.9;

  aero_rep_get_number_conc__n_m3(model_data, AERO_REP_IDX,
                           AERO_PHASE_IDX_2, &num_conc, &(partial_deriv[1]));

  printf("\nnumber conc : %f", num_conc);
  ret_val += ASSERT_MSG(fabs(num_conc-PART_NUM_CONC) < 1.0e-10*PART_NUM_CONC,
                        "Bad number concentration");

  ret_val += ASSERT_MSG(partial_deriv[0] = 999.9,
                        "Bad Jacobian (-1)");
  for( int i = 4; i < N_JAC_ELEM+1; ++i )
    ret_val += ASSERT_MSG(partial_deriv[i] == ZERO,
                          "Bad Jacobian element");
  ret_val += ASSERT_MSG(partial_deriv[N_JAC_ELEM+1] = 999.9,
                        "Bad Jacobian (end+1)");

  return ret_val;
}

/** \brief Test the total aerosol phase mass function
 *
 * \param model_data Pointer to the model data
 * \param state Solver state
 */

int test_aero_phase_mass(ModelData * model_data, N_Vector state) {

  int ret_val = 0;
  double partial_deriv_1[N_JAC_ELEM+2];
  double partial_deriv_2[N_JAC_ELEM+2];
  double phase_mass_1 = -999.9;
  double phase_mass_2 = -999.9;

  for( int i = 0; i < N_JAC_ELEM+2; ++i ) partial_deriv_1[i] = 999.9;
  for( int i = 0; i < N_JAC_ELEM+2; ++i ) partial_deriv_2[i] = 999.9;

  aero_rep_get_aero_phase_mass__kg_m3(model_data, AERO_REP_IDX, AERO_PHASE_IDX_1,
                               &phase_mass_1, &(partial_deriv_1[1]));
/*  aero_rep_get_aero_phase_mass__kg_m3(model_data, AERO_REP_IDX, AERO_PHASE_IDX_2,
*/                               &phase_mass_2, &(partial_deriv_2[1]));

  double mass_1 = CONC_rasberry + CONC_honey + CONC_sugar + CONC_lemon;
/*  double mass_2 = CONC_wheatB + CONC_waterB + CONC_saltB;
*/
  ret_val += ASSERT_MSG(fabs(phase_mass_1-mass_1) < 1.0e-10*mass_1,
                        "Bad aerosol phase mass");

  ret_val += ASSERT_MSG(partial_deriv_1[0] = 999.9,
                        "Bad Jacobian (-1)");
  for( int i = 1; i < 4; ++i )
    ret_val += ASSERT_MSG(partial_deriv_1[i] == ZERO,
                          "Bad Jacobian element");
  for( int i = 4; i < 8; ++i )
    ret_val += ASSERT_MSG(partial_deriv_1[i] == ONE,
                          "Bad Jacobian element");
  for( int i = 8; i < 10; ++i )
    ret_val += ASSERT_MSG(partial_deriv_1[i] == ZERO,
                          "Bad Jacobian element");
  for( int i = 10; i < N_JAC_ELEM+1; ++i )
    ret_val += ASSERT_MSG(partial_deriv_1[i] == ZERO,
                          "Bad Jacobian element");
  ret_val += ASSERT_MSG(partial_deriv_1[N_JAC_ELEM+1] = 999.9,
                        "Bad Jacobian (end+1)");
/**
  ret_val_2 += ASSERT_MSG(fabs(phase_mass_2-mass_2) < 1.0e-10*mass_2,
                        "Bad aerosol phase mass");
  ret_val_2 += ASSERT_MSG(partial_deriv_2[0] = 999.9,
                        "Bad Jacobian (-1)");
  for( int i = 1; i < 4; ++i )
    ret_val_2 += ASSERT_MSG(partial_deriv_2[i] == ZERO,
                          "Bad Jacobian element");
  for( int i = 4; i < 8; ++i )
    ret_val_2 += ASSERT_MSG(partial_deriv_2[i] == ZERO,
                          "Bad Jacobian element");
  for( int i = 8; i < 10; ++i )
    ret_val_2 += ASSERT_MSG(partial_deriv_2[i] == ZERO,
                          "Bad Jacobian element");
  for( int i = 10; i < N_JAC_ELEM+1; ++i )
    ret_val_2 += ASSERT_MSG(partial_deriv_2[i] == ONE,
                          "Bad Jacobian element");
  ret_val_2 += ASSERT_MSG(partial_deriv_2[N_JAC_ELEM+1] = 999.9,
                        "Bad Jacobian (end+1)");

  return ret_val_1;
  return ret_val_2;
*/
  return ret_val;
}

/** \brief Test the aerosol phase average molecular weight function
 *
 * \param model_data Pointer to the model data
 * \param state Solver state
 */

int test_aero_phase_avg_MW(ModelData * model_data, N_Vector state) {

  int ret_val = 0;
  double partial_deriv[N_JAC_ELEM+2];
  double avg_mw = -999.9;

  for( int i = 0; i < N_JAC_ELEM+2; ++i ) partial_deriv[i] = 999.9;

  aero_rep_get_aero_phase_avg_MW__kg_mol(model_data, AERO_REP_IDX, AERO_PHASE_IDX_1,
                                 &avg_mw, &(partial_deriv[1]));

  double mass = CONC_rasberry + CONC_honey + CONC_sugar + CONC_lemon;
  double moles = CONC_rasberry / MW_rasberry + CONC_honey / MW_honey + CONC_sugar / MW_sugar + CONC_lemon / MW_lemon;
  double avg_mw_real = mass / moles;
  double dMW_drasberry = ONE / moles - mass / (moles * moles * MW_rasberry);
  double dMW_dhoney = ONE / moles - mass / (moles * moles * MW_honey);
  double dMW_dsugar = ONE / moles - mass / (moles * moles * MW_sugar);
  double dMW_dlemon = ONE / moles - mass / (moles * moles * MW_lemon);

  ret_val += ASSERT_MSG(fabs(avg_mw-avg_mw_real) < 1.0e-10*avg_mw_real,
                        "Bad average MW");

  ret_val += ASSERT_MSG(partial_deriv[0] = 999.9,
                        "Bad Jacobian (-1)");
  for( int i = 1; i < 4; ++i )
    ret_val += ASSERT_MSG(partial_deriv[i] == ZERO,
                          "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv[4]-dMW_drasberry) < 1.0e-10*fabs(dMW_drasberry),
                        "Bad Jacobian (-1)");
  ret_val += ASSERT_MSG(fabs(partial_deriv[5]-dMW_dhoney) < 1.0e-10*fabs(dMW_dhoney),
                        "Bad Jacobian (-1)");
  ret_val += ASSERT_MSG(fabs(partial_deriv[6]-dMW_dsugar) < 1.0e-10*fabs(dMW_dsugar),
                        "Bad Jacobian (-1)");
  ret_val += ASSERT_MSG(fabs(partial_deriv[7]-dMW_dlemon) < 1.0e-10*fabs(dMW_dlemon),
                        "Bad Jacobian (-1)");
  for( int i = 8; i < N_JAC_ELEM+1; ++i )
    ret_val += ASSERT_MSG(partial_deriv[i] == ZERO,
                          "Bad Jacobian element");
  ret_val += ASSERT_MSG(partial_deriv[N_JAC_ELEM+1] = 999.9,
                        "Bad Jacobian (end+1)");

  return ret_val;
}

#endif

/** \brief Run c function tests
 *
 * \param solver_data Pointer to the solver data
 * \param state Pointer to the state array
 * \param env Pointer to the environmental state array
 * \return 0 if tests pass; otherwise number of test failures
 */

int run_aero_rep_single_particle_c_tests(void *solver_data, double *state, double *env) {

  int ret_val = 0;

#ifdef CAMP_USE_SUNDIALS
  SolverData *sd = (SolverData*) solver_data;
  ModelData * model_data = &(sd->model_data);
  int n_solver_var = NV_LENGTH_S(sd->y);
  N_Vector solver_state = N_VNew_Serial(n_solver_var);

  model_data->grid_cell_id = 0;
  model_data->total_state     = state;
  model_data->grid_cell_state = model_data->total_state;
  model_data->total_env       = env;
  model_data->grid_cell_env   = model_data->total_env;

  bool *jac_struct = malloc(sizeof(bool) * n_solver_var);
  ret_val += ASSERT_MSG(jac_struct!=NULL, "jac_struct not allocated");
  if (ret_val>0) return ret_val;

  for (int i_var=0; i_var<n_solver_var; ++i_var) jac_struct[i_var] = false;

  int aero_phase_idx = AERO_PHASE_IDX_1;   // phase 2
  int aero_rep_idx   = AERO_REP_IDX;     // only one aero rep in the test

  int n_jac_elem = aero_rep_get_used_jac_elem(model_data, aero_rep_idx,
                       aero_phase_idx, jac_struct);

  free(jac_struct);

  ret_val += ASSERT_MSG(n_jac_elem==N_JAC_ELEM, "Bad number of Jac elements");


  // set concentrations of test particle species
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*12+0] = state[(TEST_PARTICLE_1-1)*12+0] = CONC_wheat; // layer one, phase one 
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*12+1] = state[(TEST_PARTICLE_1-1)*12+1] = CONC_water; // layer one, phase one
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*12+2] = state[(TEST_PARTICLE_1-1)*12+2] = CONC_salt; // layer one, phase one
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*12+3] = state[(TEST_PARTICLE_1-1)*12+3] = CONC_rasberry; // layer two, phase one
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*12+4] = state[(TEST_PARTICLE_1-1)*12+4] = CONC_honey; // layer two, phase one
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*12+5] = state[(TEST_PARTICLE_1-1)*12+5] = CONC_sugar; // layer two, phase one
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*12+6] = state[(TEST_PARTICLE_1-1)*12+6] = CONC_lemon; // layer two, phase one
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*12+7] = state[(TEST_PARTICLE_1-1)*12+7] = CONC_almonds; // layer two, phase two
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*12+8] = state[(TEST_PARTICLE_1-1)*12+8] = CONC_sugarB; // layer two, phase two
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*12+9] = state[(TEST_PARTICLE_1-1)*12+9] = CONC_wheatB; // layer three, phase one
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*12+10] = state[(TEST_PARTICLE_1-1)*12+10] = CONC_waterB; // layer three, phase one
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*12+11] = state[(TEST_PARTICLE_1-1)*12+11] = CONC_saltB; // layer three, phase one

  NV_DATA_S(solver_state)[(TEST_PARTICLE_2-1)*12+0] = state[(TEST_PARTICLE_2-1)*12+0] = CONC_wheat; // layer one, phase one
  NV_DATA_S(solver_state)[(TEST_PARTICLE_2-1)*12+1] = state[(TEST_PARTICLE_2-1)*12+1] = CONC_water; // layer one, phase one
  NV_DATA_S(solver_state)[(TEST_PARTICLE_2-1)*12+2] = state[(TEST_PARTICLE_2-1)*12+2] = CONC_salt; // layer one, phase one
  NV_DATA_S(solver_state)[(TEST_PARTICLE_2-1)*12+3] = state[(TEST_PARTICLE_2-1)*12+3] = CONC_rasberry; // layer two, phase one
  NV_DATA_S(solver_state)[(TEST_PARTICLE_2-1)*12+4] = state[(TEST_PARTICLE_2-1)*12+4] = CONC_honey; // layer two, phase one
  NV_DATA_S(solver_state)[(TEST_PARTICLE_2-1)*12+5] = state[(TEST_PARTICLE_2-1)*12+5] = CONC_sugar; // layer two, phase one
  NV_DATA_S(solver_state)[(TEST_PARTICLE_2-1)*12+6] = state[(TEST_PARTICLE_2-1)*12+6] = CONC_lemon; // layer two, phase one
  NV_DATA_S(solver_state)[(TEST_PARTICLE_2-1)*12+7] = state[(TEST_PARTICLE_2-1)*12+7] = CONC_almonds; // layer two, phase two
  NV_DATA_S(solver_state)[(TEST_PARTICLE_2-1)*12+8] = state[(TEST_PARTICLE_2-1)*12+8] = CONC_sugarB; // layer two, phase two
  NV_DATA_S(solver_state)[(TEST_PARTICLE_2-1)*12+9] = state[(TEST_PARTICLE_2-1)*12+9] = CONC_wheatB; // layer three, phase one
  NV_DATA_S(solver_state)[(TEST_PARTICLE_2-1)*12+10] = state[(TEST_PARTICLE_2-1)*12+10] = CONC_waterB; // layer three, phase one
  NV_DATA_S(solver_state)[(TEST_PARTICLE_2-1)*12+11] = state[(TEST_PARTICLE_2-1)*12+11] = CONC_saltB; // layer three, phase one
  
  // Set the environment-dependent parameter pointer to the first grid cell
  model_data->grid_cell_aero_rep_env_data = model_data->aero_rep_env_data;

  // Update the environmental and concentration states
  aero_rep_update_env_state(model_data);
  aero_rep_update_state(model_data);

  // Run the property tests
  ret_val += test_effective_radius(model_data, solver_state);
  ret_val += test_aero_phase_mass(model_data, solver_state);
  ret_val += test_aero_phase_avg_MW(model_data, solver_state);
  ret_val += test_number_concentration(model_data, solver_state);

  N_VDestroy(solver_state);
#endif

  return ret_val;
}

