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

// number of computational particles in the test
#define N_COMP_PARTICLES 3

// number of aerosol phases per particle
#define NUM_AERO_PHASE 7

// index for the test phase
// (test-particle phase 2 : middle layer, jam)
#define AERO_PHASE_IDX_1 ((TEST_PARTICLE_1-1)*NUM_AERO_PHASE+2)
// (test-particle phase 4 : top layer, bread)
#define AERO_PHASE_IDX_2 ((TEST_PARTICLE_1-1)*NUM_AERO_PHASE+6)

// number of Jacobian elements used for the test phase
#define N_JAC_ELEM 19

// Test concentrations (kg/m3)
#define CONC_wheat 1.0
#define CONC_water 2.0
#define CONC_salt 3.0
#define CONC_rasberry 4.0
#define CONC_honey 5.0
#define CONC_sugar 6.0
#define CONC_lemon 7.0
#define CONC_almonds 8.0

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
                            CONC_almonds / DENSITY_almonds +
                            CONC_sugar / DENSITY_sugar +
                            CONC_rasberry / DENSITY_rasberry +
                            CONC_honey / DENSITY_honey +
                            CONC_sugar / DENSITY_sugar +
                            CONC_lemon / DENSITY_lemon +
                            CONC_almonds / DENSITY_almonds +
                            CONC_wheat / DENSITY_wheat +
                            CONC_water / DENSITY_water +
                            CONC_salt / DENSITY_salt + 
                            CONC_sugar / DENSITY_sugar +
                            CONC_almonds / DENSITY_almonds +
                            CONC_sugar / DENSITY_sugar +
                            CONC_wheat / DENSITY_wheat +
                            CONC_water / DENSITY_water +
                            CONC_salt / DENSITY_salt ); // volume density (m3/m3)
  
  double eff_rad_expected = pow( ( 3.0 / 4.0 / 3.14159265359 * volume_density ), 1.0/3.0 );
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
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[4] - d_eff_rad_dx / DENSITY_almonds) <
                        1.0e-10 * partial_deriv_1[4], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[5] - d_eff_rad_dx / DENSITY_sugar) <
                        1.0e-10 * partial_deriv_1[5], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[6] - d_eff_rad_dx / DENSITY_rasberry) <
                        1.0e-10 * partial_deriv_1[6], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[7] - d_eff_rad_dx / DENSITY_honey) <
                        1.0e-10 * partial_deriv_1[7], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[8] - d_eff_rad_dx / DENSITY_sugar) <
                        1.0e-10 * partial_deriv_1[8], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[9] - d_eff_rad_dx / DENSITY_lemon) <
                        1.0e-10 * partial_deriv_1[9], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[10] - d_eff_rad_dx / DENSITY_almonds) <
                        1.0e-10 * partial_deriv_1[10], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[11] - d_eff_rad_dx / DENSITY_sugar) <
                        1.0e-10 * partial_deriv_1[11], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[12] - d_eff_rad_dx / DENSITY_wheat) <
                        1.0e-10 * partial_deriv_1[12], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[13] - d_eff_rad_dx / DENSITY_water) <
                        1.0e-10 * partial_deriv_1[13], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[14] - d_eff_rad_dx / DENSITY_salt) <
                        1.0e-10 * partial_deriv_1[14], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[15] - d_eff_rad_dx / DENSITY_almonds) <
                        1.0e-10 * partial_deriv_1[15], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[16] - d_eff_rad_dx / DENSITY_sugar) <
                        1.0e-10 * partial_deriv_1[16], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[17] - d_eff_rad_dx / DENSITY_wheat) <
                        1.0e-10 * partial_deriv_1[17], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[18] - d_eff_rad_dx / DENSITY_water) <
                        1.0e-10 * partial_deriv_1[18], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[19] - d_eff_rad_dx / DENSITY_salt) <
                        1.0e-10 * partial_deriv_1[19], "Bad Jacobian element");
  ret_val += ASSERT_MSG(partial_deriv_1[N_JAC_ELEM+1] = 999.9,
                        "Bad Jacobian (end+1)");

  return ret_val;
}

/** \brief Test the surface area of interfacial layer function
 *
 * \param model_data Pointer to the model data
 * \param state Solver state
 */

int test_surface_area_layer(ModelData * model_data, N_Vector state) {

  int ret_val = 0;
  double partial_deriv[N_JAC_ELEM+2];
  double eff_sa = -999.9;

  for( int i = 0; i < N_JAC_ELEM+2; ++i ) partial_deriv[i] = 999.9;

  aero_rep_get_interface_surface_area__m2(model_data, AERO_REP_IDX,
                                AERO_PHASE_IDX_1 - NUM_AERO_PHASE, 
                                AERO_PHASE_IDX_2 - NUM_AERO_PHASE, 
                                &eff_sa, &(partial_deriv[1]));

  double volume_density = ( CONC_wheat / DENSITY_wheat +
                            CONC_water / DENSITY_water +
                            CONC_salt / DENSITY_salt +
                            CONC_almonds / DENSITY_almonds +
                            CONC_sugar / DENSITY_sugar +
                            CONC_rasberry / DENSITY_rasberry +
                            CONC_honey / DENSITY_honey +
                            CONC_sugar / DENSITY_sugar +
                            CONC_lemon / DENSITY_lemon +
                            CONC_almonds / DENSITY_almonds +
                            CONC_wheat / DENSITY_wheat +
                            CONC_water / DENSITY_water +
                            CONC_salt / DENSITY_salt +
                            CONC_sugar / DENSITY_sugar ); // volume density (m3/m3)

  double volume_density_jam = ( CONC_rasberry / DENSITY_rasberry +
                            CONC_honey / DENSITY_honey +
                            CONC_sugar / DENSITY_sugar +
                            CONC_lemon / DENSITY_lemon ); // volume density of jam (m3/m3)

  double volume_density_layer_2 = ( CONC_almonds / DENSITY_almonds +
                            CONC_sugar / DENSITY_sugar +
                            CONC_rasberry / DENSITY_rasberry +
                            CONC_honey / DENSITY_honey +
                            CONC_sugar / DENSITY_sugar +
                            CONC_lemon / DENSITY_lemon +
                            CONC_wheat / DENSITY_wheat +
                            CONC_water / DENSITY_water +
                            CONC_salt / DENSITY_salt ); // volume density of layer 2 (m3/m3)

  double volume_density_bread = ( CONC_wheat / DENSITY_wheat +
                            CONC_water / DENSITY_water +
                            CONC_salt / DENSITY_salt ); // volume density of bread (m3/m3)

  double volume_density_layer_3 = ( CONC_almonds / DENSITY_almonds +
                            CONC_sugar / DENSITY_sugar +
                            CONC_wheat / DENSITY_wheat +
                            CONC_water / DENSITY_water +
                            CONC_salt / DENSITY_salt ); // volume density of layer 3 (m3/m3)

  double eff_rad_expected = pow( ( 3.0 / 4.0 / 3.14159265359 * volume_density ), 1.0/3.0 );
  double f_jam = volume_density_jam / volume_density_layer_2;
  double f_bread = volume_density_bread / volume_density_layer_3;

  double eff_sa_expected = f_jam * f_bread * 4.0 * 3.14159265359 * pow( eff_rad_expected, 2.0 );
  ret_val += ASSERT_MSG(fabs(eff_sa-eff_sa_expected) < 1.0e-4*eff_sa_expected,
                        "Bad surface area layer");

  ret_val += ASSERT_MSG(partial_deriv[0] = 999.9,
                        "Bad Jacobian (-1)");

  double d_eff_sa_dx = ((volume_density_layer_2 - volume_density_jam) *
              pow(volume_density_layer_2, -2.0) * f_bread * eff_sa_expected) +
              ((volume_density_layer_3 - volume_density_bread) *
              pow(volume_density_layer_3, -2.0) * f_jam * eff_sa_expected) +
              (2.0 * f_jam * f_bread * pow(volume_density * 3.0 / 4.0 / 3.14159265359, -1.0 / 3.0)) ;

  ret_val += ASSERT_MSG(fabs(partial_deriv[1] - d_eff_sa_dx / DENSITY_wheat) <
                        1.0e-10 * partial_deriv[1], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv[2] - d_eff_sa_dx / DENSITY_water) <
                        1.0e-10 * partial_deriv[2], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv[3] - d_eff_sa_dx / DENSITY_salt) <
                        1.0e-10 * partial_deriv[3], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv[4] - d_eff_sa_dx / DENSITY_almonds) <
                        1.0e-10 * partial_deriv[4], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv[5] - d_eff_sa_dx / DENSITY_sugar) <
                        1.0e-10 * partial_deriv[5], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv[6] - d_eff_sa_dx / DENSITY_rasberry) <
                        1.0e-10 * partial_deriv[6], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv[7] - d_eff_sa_dx / DENSITY_honey) <
                        1.0e-10 * partial_deriv[7], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv[8] - d_eff_sa_dx / DENSITY_sugar) <
                        1.0e-10 * partial_deriv[8], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv[9] - d_eff_sa_dx / DENSITY_lemon) <
                        1.0e-10 * partial_deriv[9], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv[10] - d_eff_sa_dx / DENSITY_almonds) <
                        1.0e-10 * partial_deriv[10], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv[11] - d_eff_sa_dx / DENSITY_sugar) <
                        1.0e-10 * partial_deriv[11], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv[12] - d_eff_sa_dx / DENSITY_wheat) <
                        1.0e-10 * partial_deriv[12], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv[13] - d_eff_sa_dx / DENSITY_water) <
                        1.0e-10 * partial_deriv[13], "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv[14] - d_eff_sa_dx / DENSITY_salt) <
                        1.0e-10 * partial_deriv[14], "Bad Jacobian element");
  for( int i = 15; i < N_JAC_ELEM+1; ++i )
    ret_val += ASSERT_MSG(partial_deriv[i] == ZERO,
                          "Bad Jacobian element");
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
                           AERO_PHASE_IDX_1, &num_conc, &(partial_deriv[1]));

  ret_val += ASSERT_MSG(fabs(num_conc-PART_NUM_CONC) < 1.0e-10*PART_NUM_CONC,
                        "Bad number concentration");

  ret_val += ASSERT_MSG(partial_deriv[0] = 999.9,
                        "Bad Jacobian (-1)");
  for( int i = 6; i < N_JAC_ELEM+1; ++i )
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

  aero_rep_get_aero_phase_mass__kg_m3(model_data, AERO_REP_IDX, AERO_PHASE_IDX_2,
                               &phase_mass_2, &(partial_deriv_2[1]));

  double mass_1 = CONC_rasberry + CONC_honey + CONC_sugar + CONC_lemon;
  double mass_2 = CONC_wheat + CONC_water + CONC_salt;

  ret_val += ASSERT_MSG(fabs(phase_mass_1-mass_1) < 1.0e-10*mass_1,
                        "Bad aerosol phase mass");

  ret_val += ASSERT_MSG(partial_deriv_1[0] = 999.9,
                        "Bad Jacobian (-1)");
  for( int i = 1; i < 6; ++i )
    ret_val += ASSERT_MSG(partial_deriv_1[i] == ZERO,
                          "Bad Jacobian element");
  for( int i = 6; i < 10; ++i )
    ret_val += ASSERT_MSG(partial_deriv_1[i] == ONE,
                          "Bad Jacobian element");
  for( int i = 10; i < 12; ++i )
    ret_val += ASSERT_MSG(partial_deriv_1[i] == ZERO,
                          "Bad Jacobian element");
  for( int i = 12; i < N_JAC_ELEM+1; ++i )
    ret_val += ASSERT_MSG(partial_deriv_1[i] == ZERO,
                          "Bad Jacobian element");
  ret_val += ASSERT_MSG(partial_deriv_1[N_JAC_ELEM+1] = 999.9,
                        "Bad Jacobian (end+1)");

  ret_val += ASSERT_MSG(fabs(phase_mass_2-mass_2) < 1.0e-10*mass_2,
                        "Bad aerosol phase mass");
  ret_val += ASSERT_MSG(partial_deriv_2[0] = 999.9,
                        "Bad Jacobian (-1)");
  for( int i = 1; i < 6; ++i )
    ret_val += ASSERT_MSG(partial_deriv_2[i] == ZERO,
                          "Bad Jacobian element");
  for( int i = 6; i < 10; ++i )
    ret_val += ASSERT_MSG(partial_deriv_2[i] == ZERO,
                          "Bad Jacobian element");
  for( int i = 10; i < 17; ++i )
    ret_val += ASSERT_MSG(partial_deriv_2[i] == ZERO,
                          "Bad Jacobian element");
  for( int i = 17; i < N_JAC_ELEM+1; ++i )
    ret_val += ASSERT_MSG(partial_deriv_2[i] == ONE,
                          "Bad Jacobian element");
  ret_val += ASSERT_MSG(partial_deriv_2[N_JAC_ELEM+1] = 999.9,
                        "Bad Jacobian (end+1)");

  return ret_val;
}

/** \brief Test the aerosol phase average molecular weight function
 *
 * \param model_data Pointer to the model data
 * \param state Solver state
 */

int test_aero_phase_avg_MW(ModelData * model_data, N_Vector state) {

  int ret_val = 0;
  double partial_deriv_1[N_JAC_ELEM+2];
  double partial_deriv_2[N_JAC_ELEM+2];
  double avg_mw_1 = -999.9;
  double avg_mw_2 = -999.9;

  for( int i = 0; i < N_JAC_ELEM+2; ++i ) partial_deriv_1[i] = 999.9;
  for( int i = 0; i < N_JAC_ELEM+2; ++i ) partial_deriv_2[i] = 999.9;

  aero_rep_get_aero_phase_avg_MW__kg_mol(model_data, AERO_REP_IDX, AERO_PHASE_IDX_1,
                                 &avg_mw_1, &(partial_deriv_1[1]));
  aero_rep_get_aero_phase_avg_MW__kg_mol(model_data, AERO_REP_IDX, AERO_PHASE_IDX_2,
                                 &avg_mw_2, &(partial_deriv_2[1]));

  double mass_1 = CONC_rasberry + CONC_honey + CONC_sugar + CONC_lemon;
  double moles_1 = CONC_rasberry / MW_rasberry + CONC_honey / MW_honey + CONC_sugar / MW_sugar + CONC_lemon / MW_lemon;
  double avg_mw_real_1 = mass_1 / moles_1;
  double dMW_drasberry = ONE / moles_1 - mass_1 / (moles_1 * moles_1 * MW_rasberry);
  double dMW_dhoney = ONE / moles_1 - mass_1 / (moles_1 * moles_1 * MW_honey);
  double dMW_dsugar = ONE / moles_1 - mass_1 / (moles_1 * moles_1 * MW_sugar);
  double dMW_dlemon = ONE / moles_1 - mass_1 / (moles_1 * moles_1 * MW_lemon);

  double mass_2 = CONC_wheat + CONC_water + CONC_salt;
  double moles_2 = CONC_wheat / MW_wheat + CONC_water / MW_water + CONC_salt / MW_salt; 
  double avg_mw_real_2 = mass_2 / moles_2;
  double dMW_dwheat = ONE / moles_2 - mass_2 / (moles_2 * moles_2 * MW_wheat);
  double dMW_dwater = ONE / moles_2 - mass_2 / (moles_2 * moles_2 * MW_water);
  double dMW_dsalt = ONE / moles_2 - mass_2 / (moles_2 * moles_2 * MW_salt);

  ret_val += ASSERT_MSG(fabs(avg_mw_1-avg_mw_real_1) < 1.0e-10*avg_mw_real_1,
                        "Bad average MW");

  ret_val += ASSERT_MSG(partial_deriv_1[0] = 999.9,
                        "Bad Jacobian (-1)");
  for( int i = 1; i < 6; ++i )
    ret_val += ASSERT_MSG(partial_deriv_1[i] == ZERO,
                          "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[6]-dMW_drasberry) < 1.0e-10*fabs(dMW_drasberry),
                        "Bad Jacobian (-1)");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[7]-dMW_dhoney) < 1.0e-10*fabs(dMW_dhoney),
                        "Bad Jacobian (-1)");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[8]-dMW_dsugar) < 1.0e-10*fabs(dMW_dsugar),
                        "Bad Jacobian (-1)");
  ret_val += ASSERT_MSG(fabs(partial_deriv_1[9]-dMW_dlemon) < 1.0e-10*fabs(dMW_dlemon),
                        "Bad Jacobian (-1)");
  for( int i = 10; i < N_JAC_ELEM+1; ++i )
    ret_val += ASSERT_MSG(partial_deriv_1[i] == ZERO,
                          "Bad Jacobian element");
  ret_val += ASSERT_MSG(partial_deriv_1[N_JAC_ELEM+1] = 999.9,
                        "Bad Jacobian (end+1)");


  ret_val += ASSERT_MSG(fabs(avg_mw_2-avg_mw_real_2) < 1.0e-10*avg_mw_real_2,
                        "Bad average MW");

  ret_val += ASSERT_MSG(partial_deriv_2[0] = 999.9,
                        "Bad Jacobian (-1)");
  for( int i = 1; i < 16; ++i )
    ret_val += ASSERT_MSG(partial_deriv_2[i] == ZERO,
                          "Bad Jacobian element");
  ret_val += ASSERT_MSG(fabs(partial_deriv_2[17]-dMW_dwheat) < 1.0e-10*fabs(dMW_dwheat),
                        "Bad Jacobian (-1)");
  ret_val += ASSERT_MSG(fabs(partial_deriv_2[18]-dMW_dwater) < 1.0e-10*fabs(dMW_dwater),
                        "Bad Jacobian (-1)");
  ret_val += ASSERT_MSG(fabs(partial_deriv_2[19]-dMW_dsalt) < 1.0e-10*fabs(dMW_dsalt),
                        "Bad Jacobian (-1)");
  ret_val += ASSERT_MSG(partial_deriv_2[N_JAC_ELEM+1] = 999.9,
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

  int aero_phase_idx = AERO_PHASE_IDX_2;   // phase 2
  int aero_rep_idx   = AERO_REP_IDX;     // only one aero rep in the test

  int n_jac_elem = aero_rep_get_used_jac_elem(model_data, aero_rep_idx,
                       aero_phase_idx, jac_struct);

  free(jac_struct);

  ret_val += ASSERT_MSG(n_jac_elem==N_JAC_ELEM, "Bad number of Jac elements");


  // set concentrations of test particle species
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*19+0] = state[(TEST_PARTICLE_1-1)*19+0] = CONC_wheat; // layer one, phase one 
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*19+1] = state[(TEST_PARTICLE_1-1)*19+1] = CONC_water; // layer one, phase one
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*19+2] = state[(TEST_PARTICLE_1-1)*19+2] = CONC_salt; // layer one, phase one
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*19+3] = state[(TEST_PARTICLE_1-1)*19+3] = CONC_almonds; // layer one, phase two
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*19+4] = state[(TEST_PARTICLE_1-1)*19+4] = CONC_sugar; // layer one, phase two
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*19+5] = state[(TEST_PARTICLE_1-1)*19+5] = CONC_rasberry; // layer two, phase one
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*19+6] = state[(TEST_PARTICLE_1-1)*19+6] = CONC_honey; // layer two, phase one
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*19+7] = state[(TEST_PARTICLE_1-1)*19+7] = CONC_sugar; // layer two, phase one
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*19+8] = state[(TEST_PARTICLE_1-1)*19+8] = CONC_lemon; // layer two, phase one
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*19+9] = state[(TEST_PARTICLE_1-1)*19+9] = CONC_almonds; // layer two, phase two
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*19+10] = state[(TEST_PARTICLE_1-1)*19+10] = CONC_sugar; // layer two, phase two
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*19+11] = state[(TEST_PARTICLE_1-1)*19+11] = CONC_wheat; // layer two, phase three
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*19+12] = state[(TEST_PARTICLE_1-1)*19+12] = CONC_water; // layer two, phase three
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*19+13] = state[(TEST_PARTICLE_1-1)*19+13] = CONC_salt; // layer two, phase three
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*19+14] = state[(TEST_PARTICLE_1-1)*19+14] = CONC_almonds; // layer three, phase one
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*19+15] = state[(TEST_PARTICLE_1-1)*19+15] = CONC_sugar; // layer three, phase one
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*19+16] = state[(TEST_PARTICLE_1-1)*19+16] = CONC_wheat; // layer three, phase two
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*19+17] = state[(TEST_PARTICLE_1-1)*19+17] = CONC_water; // layer three, phase two
  NV_DATA_S(solver_state)[(TEST_PARTICLE_1-1)*19+18] = state[(TEST_PARTICLE_1-1)*19+18] = CONC_salt; // layer three, phase two

  // Set the environment-dependent parameter pointer to the first grid cell
  model_data->grid_cell_aero_rep_env_data = model_data->aero_rep_env_data;

  // Update the environmental and concentration states
  aero_rep_update_env_state(model_data);
  aero_rep_update_state(model_data);

  aero_rep_print_data(sd);

  // Run the property tests
  ret_val += test_effective_radius(model_data, solver_state);
  //ret_val += test_surface_area_layer(model_data, solver_state);
  ret_val += test_aero_phase_mass(model_data, solver_state);
  ret_val += test_aero_phase_avg_MW(model_data, solver_state);
  ret_val += test_number_concentration(model_data, solver_state);

  N_VDestroy(solver_state);
#endif

  return ret_val;
}

