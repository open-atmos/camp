#!/usr/bin/env bash
set -e
unset I_MPI_PMI_LIBRARY
scriptdir="$(dirname "$0")"
cd "$scriptdir"
cd ../build
sed -i 's/^USE_TESTS:BOOL=OFF$/USE_TESTS:BOOL=ON/' CMakeCache.txt # Enable all tests (MUST BE ENABLED BY DEFAULT)
#sed -i 's/^USE_TESTS:BOOL=ON$/USE_TESTS:BOOL=OFF/' CMakeCache.txt # Disable all, useful to test individual ones (SET ONLY FOR DEBUG)
make -j 8
ctest --output-on-failure
#ctest --verbose
#Run individual tests:
#./test_run/unit_rxn_data/test_HL_phase_transfer.sh MPI
#./test_run/unit_rxn_data/test_SIMPOL_phase_transfer.sh MPI
#cd test_run/chemistry/cb05cl_ae5
#./test_chemistry_cb05cl_ae5.sh MPI
#cd test_run/unit_rxn_data
#./test_run/unit_sub_model_data/test_ZSR_aerosol_water.sh MPI
#./test_run/unit_rxn_data/test_emission.sh MPI
#./test_run/unit_rxn_data/test_HL_phase_transfer.sh
#./test_run/unit_rxn_data/test_CMAQ_H2O2.sh MPI
#./test_run/unit_rxn_data/test_arrhenius.sh MPI
#./test_run/unit_rxn_data/test_photolysis.sh MPI #TODO: CHECK WHY CPU WORKS BUT GPU NOT
#./test_run/unit_rxn_data/test_troe.sh MPI #TODO: CHECK WHY CPU WORKS BUT GPU NOT
#./test_run/unit_rxn_data/test_aqueous_equilibrium.sh
#./test_run/unit_sub_model_data//test_UNIFAC.sh MPI
#mpirun -np 2 ./unit_test_aero_rep_single_particle
#mpirun -np 2 ./unit_test_aero_rep_modal_binned_mass
#cd ../test/monarch
#./checkGPU.sh
#python checkGPU.py
