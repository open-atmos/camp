include(test_util)

create_standard_test(NAME rxn_aqueous_equilibrium SOURCES test_rxn_aqueous_equilibrium.F90 RESULT "PASS" WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data)
create_standard_test(NAME rxn_CMAQ_H2O2 SOURCES test_rxn_CMAQ_H2O2.F90 RESULT "PASS" WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data)
create_standard_test(NAME rxn_CMAQ_OH_HNO3 SOURCES test_rxn_CMAQ_OH_HNO3.F90 RESULT "PASS" WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data)
create_standard_test(NAME rxn_condensed_phase_arrhenius SOURCES test_rxn_condensed_phase_arrhenius.F90 RESULT "PASS" WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data)
create_standard_test(NAME rxn_condensed_phase_photolysis SOURCES test_rxn_condensed_phase_photolysis.F90 RESULT "PASS" WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data)
create_standard_test(NAME rxn_emission SOURCES test_rxn_emission.F90 RESULT "PASS" WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data)
create_standard_test(NAME rxn_first_order_loss SOURCES test_rxn_first_order_loss.F90 RESULT "PASS" WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data)
create_standard_test(NAME rxn_HL_phase_transfer SOURCES test_rxn_HL_phase_transfer.F90 RESULT "PASS" WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data)
create_standard_test(NAME rxn_photolysis SOURCES test_rxn_photolysis.F90 RESULT "PASS" WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data)
create_standard_test(NAME rxn_SIMPOL_phase_transfer SOURCES test_rxn_SIMPOL_phase_transfer.F90 RESULT "PASS" WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data)
create_standard_test(NAME rxn_surface SOURCES test_rxn_surface.F90 RESULT "PASS" WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data)
create_standard_test(NAME rxn_ternary_chemical_activation SOURCES test_rxn_ternary_chemical_activation.F90 RESULT "PASS" WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data)
create_standard_test(NAME rxn_troe SOURCES test_rxn_troe.F90 RESULT "PASS" WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data)
create_standard_test(NAME rxn_wennberg_no_ro2 SOURCES test_rxn_wennberg_no_ro2.F90 RESULT "PASS" WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data)
create_standard_test(NAME rxn_wennberg_tunneling SOURCES test_rxn_wennberg_tunneling.F90 RESULT "PASS" WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data)
