set -e
cd ../build
make -j 4
#ctest --output-on-failure
./test_run/unit_rxn_data/test_CMAQ_H2O2.sh
#./test_run/unit_rxn_data/test_HL_phase_transfer.sh
#./test_run/unit_rxn_data/test_arrhenius.sh
#./test_run/unit1_rxn_data/test_aqueous_equilibrium.sh
#./unit_test_aero_rep_single_particle
cd ../test/monarch
#./checkGPU.sh
#python checkGPU.py
camp_mpi_size