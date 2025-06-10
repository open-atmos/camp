######################################################################
# Unit test macro

include(test_util)

if(NOT ${CMAKE_Fortran_COMPILER_ID} MATCHES "Intel")
  set(UNIT_TEST_SRC
      unit_test_data.F90
      unit_test_driver.F90)
  
  set(UNIT_TEST_RXN_ARRHENIUS_SRC
      rxns/unit_test_rxn_arrhenius.F90 ${UNIT_TEST_SRC})
  if(${CMAKE_Fortran_COMPILER_ID} MATCHES "Intel")
    set_source_files_properties(${UNIT_TEST_RXN_ARRHENIUS_SRC} PROPERTIES COMPILE_DEFINITIONS
      "UNIT_TEST_MODULE_=camp_unit_test_rxn_arrhenius \
      ;UNIT_TEST_TYPE_='unit_test_rxn_arrhenius_t()'")
  elseif(${CMAKE_Fortran_COMPILER_ID} MATCHES "GNU")
    set_source_files_properties(${UNIT_TEST_RXN_ARRHENIUS_SRC} PROPERTIES COMPILE_DEFINITIONS
      "UNIT_TEST_MODULE_=camp_unit_test_rxn_arrhenius \
      ;UNIT_TEST_TYPE_=unit_test_rxn_arrhenius_t()")
  endif()
  create_standard_test(NAME rxn_arrhenius SOURCES ${UNIT_TEST_RXN_ARRHENIUS_SRC} WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test_run/unit_tests)
endif()

