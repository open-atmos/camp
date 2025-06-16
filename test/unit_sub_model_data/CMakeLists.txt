include(test_util)

create_standard_test(NAME sub_model_PDFiTE SOURCES test_sub_model_PDFiTE.F90 RESULT "PASS" WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test_run/unit_sub_model_data)
create_standard_test(NAME sub_model_UNIFAC SOURCES test_sub_model_UNIFAC.F90 RESULT "PASS" WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test_run/unit_sub_model_data)
create_standard_test(NAME sub_model_ZSR_aerosol_water SOURCES test_sub_model_ZSR_aerosol_water.F90 test_sub_model_ZSR_aerosol_water.c RESULT "PASS" WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test_run/unit_sub_model_data)
