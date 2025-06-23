include(test_util)

create_standard_test(NAME aero_rep_modal_binned_mass SOURCES test_aero_rep_modal_binned_mass.F90 test_aero_rep_modal_binned_mass.c RESULT "PASS" WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)
create_standard_test(NAME aero_rep_single_particle SOURCES test_aero_rep_single_particle.F90 test_aero_rep_single_particle.c RESULT "PASS" WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)
