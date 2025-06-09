cmake_minimum_required(VERSION 3.21)

project(camp-distribution VERSION 1.1.0 LANGUAGES Fortran C)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${PROJECT_SOURCE_DIR}/cmake")

set(PROJECT_BUGREPORT "mattldawson@gmail.com")
set(PROJECT_NAME "CAMP")
set(PROJECT_STRING "CAMP 1.1.0")
set(PROJECT_TARNAME "camp")
set(PROJECT_VERSION "1.1.0")

if ( "${CMAKE_SYSTEM_NAME}" MATCHES "Darwin" )
  set ( ENABLE_DYLIBS_USE_RPATH TRUE CACHE BOOL
    "Enable @rpath install name for dylibs" )
  mark_as_advanced ( ENABLE_DYLIBS_USE_RPATH )
endif ()

if ( ENABLE_DYLIBS_USE_RPATH )
  set ( CMAKE_MACOSX_RPATH TRUE )
else ()
  set ( CMAKE_INSTALL_NAME_DIR
    "${CMAKE_INSTALL_FULL_LIBDIR}" )
endif ()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

include(GNUInstallDirs)

######################################################################
# options

option(ENABLE_MPI "Enable MPI parallel support" OFF)
option(ENABLE_DEBUG "Compile debugging functions" OFF)
option(FAILURE_DETAIL "Output conditions before and after solver failures" OFF)
option(ENABLE_CXX "Enable C++" OFF)
option(ENABLE_GPU "Enable use of GPUs in chemistry solving" OFF)

mark_as_advanced(FORCE ENABLE_DEBUG FAILURE_DETAIL)

include(${PROJECT_SOURCE_DIR}/cmake/dependencies.cmake)

################################################################################
# Projet wide setup variables 

set(CAMP_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR})
set(CAMP_MOD_DIR ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR})
set(CAMP_LIB_DIR ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CAMP_PROJECT_SRC_DIR ${PROJECT_SOURCE_DIR})
set(camp_compile_definitions "")

######################################################################
## Compiler settings

if(ENABLE_CXX)
  list(APPEND camp_compile_definitions CAMP_USE_CXX)
  enable_language(CXX)
endif()

######################################################################
# GPU support

if (ENABLE_GPU)
  list(APPEND camp_compile_definitions CAMP_USE_GPU)
  enable_language(CUDA)
endif()

######################################################################
# debugging options

if (ENABLE_DEBUG)
  list(APPEND camp_compile_definitions CAMP_DEBUG)
endif()
if (FAILURE_DETAIL)
  list(APPEND camp_compile_definitions FAILURE_DETAIL)
endif()

######################################################################
# Unit test macro

macro(do_unit_test test_name result)
  if(ENABLE_MPI)
    add_test(unit_test_${test_name} mpirun -v -np 2 ${CMAKE_BINARY_DIR}/unit_test_${test_name})
  else()
    add_test(unit_test_${test_name} ${CMAKE_BINARY_DIR}/unit_test_${test_name})
  endif()
  set_tests_properties(unit_test_${test_name}
	  PROPERTIES PASS_REGULAR_EXPRESSION ${result})
endmacro(do_unit_test)

######################################################################
# copy dirs

add_custom_target(copy_data ALL ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/data ${CMAKE_BINARY_DIR}/data_run)
add_custom_target(copy_test ALL ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/test ${CMAKE_BINARY_DIR}/test_run)
add_custom_target(copy_mechanism ALL ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/mechanisms ${CMAKE_BINARY_DIR}/mechanisms_run)
add_custom_target(copy_boot_camp ALL ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/doc/camp_tutorial/boot_camp ${CMAKE_BINARY_DIR}/boot_camp_run)
set_property(DIRECTORY PROPERTY ADDITIONAL_MAKE_CLEAN_FILES test_run mechanisms_run scenarios_run boot_camp_run)

######################################################################
# tests

enable_testing()
do_unit_test(property "PASS")
do_unit_test(chem_spec_data "PASS")
do_unit_test(aero_phase_data "PASS")
do_unit_test(jacobian "PASS")
do_unit_test(aero_rep_single_particle "PASS")
do_unit_test(aero_rep_modal_binned_mass "PASS")
do_unit_test(camp_core "PASS")

if (ENABLE_MPI)
  set(MPI_TEST_FLAG MPI)
else()
  set(MPI_TEST_FLAG serial)
endif()

# New unit tests (UNDER DEVELOPMENT)
if (NOT ${CMAKE_Fortran_COMPILER_ID} MATCHES "Intel")
  add_test(test_rxn_arrhenius_mech ${CMAKE_BINARY_DIR}/test_run/unit_tests/input_files/run_rxn_arrhenius.sh ${MPI_TEST_FLAG})
endif()

# Old-style unit tests
add_test(test_rxn_aqueous_equilibrium ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data/test_aqueous_equilibrium.sh ${MPI_TEST_FLAG})
add_test(test_rxn_CMAQ_H2O2 ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data/test_CMAQ_H2O2.sh ${MPI_TEST_FLAG})
add_test(test_rxn_CMAQ_OH_HNO3 ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data/test_CMAQ_OH_HNO3.sh ${MPI_TEST_FLAG})
add_test(test_rxn_condensed_phase_arrhenius ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data/test_condensed_phase_arrhenius.sh ${MPI_TEST_FLAG})
add_test(test_rxn_condensed_phase_photolysis ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data/test_condensed_phase_photolysis.sh ${MPI_TEST_FLAG})
add_test(test_rxn_emission ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data/test_emission.sh ${MPI_TEST_FLAG})
add_test(test_rxn_first_order_loss ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data/test_first_order_loss.sh ${MPI_TEST_FLAG})
add_test(test_rxn_HL_phase_transfer ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data/test_HL_phase_transfer.sh ${MPI_TEST_FLAG})
add_test(test_rxn_photolysis ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data/test_photolysis.sh ${MPI_TEST_FLAG})
add_test(test_rxn_SIMPOL_phase_transfer ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data/test_SIMPOL_phase_transfer.sh ${MPI_TEST_FLAG})
add_test(test_rxn_surface ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data/test_surface.sh ${MPI_TEST_FLAG})
add_test(test_rxn_ternary_chemical_activation ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data/test_ternary_chemical_activation.sh ${MPI_TEST_FLAG})
add_test(test_rxn_troe ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data/test_troe.sh ${MPI_TEST_FLAG})
add_test(test_rxn_wennberg_no_ro2 ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data/test_wennberg_no_ro2.sh ${MPI_TEST_FLAG})
add_test(test_rxn_wennberg_tunneling ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data/test_wennberg_tunneling.sh ${MPI_TEST_FLAG})
add_test(test_rxn_wet_deposition ${CMAKE_BINARY_DIR}/test_run/unit_rxn_data/test_wet_deposition.sh ${MPI_TEST_FLAG})
add_test(test_sub_model_PDFiTE ${CMAKE_BINARY_DIR}/test_run/unit_sub_model_data/test_PDFiTE.sh ${MPI_TEST_FLAG})
add_test(test_sub_model_UNIFAC ${CMAKE_BINARY_DIR}/test_run/unit_sub_model_data/test_UNIFAC.sh ${MPI_TEST_FLAG})
add_test(test_sub_model_ZSR_aerosol_water ${CMAKE_BINARY_DIR}/test_run/unit_sub_model_data/test_ZSR_aerosol_water.sh ${MPI_TEST_FLAG})
add_test(test_chem_mech_solver ${CMAKE_BINARY_DIR}/test_run/chemistry/test_chemistry_1.sh ${MPI_TEST_FLAG})
add_test(test_chemistry_cb05cl_ae5 ${CMAKE_BINARY_DIR}/test_run/chemistry/cb05cl_ae5/test_chemistry_cb05cl_ae5.sh ${MPI_TEST_FLAG})
if(NOT ${CMAKE_Fortran_COMPILER_ID} MATCHES "Intel")
  add_test(test_MONARCH_1 ${CMAKE_BINARY_DIR}/test_run/monarch/test_monarch_1.sh ${MPI_TEST_FLAG})
  add_test(test_MONARCH_2 ${CMAKE_BINARY_DIR}/test_run/monarch/test_monarch_2.sh ${MPI_TEST_FLAG})
endif()
add_test(boot_camp_part_1 ${CMAKE_BINARY_DIR}/boot_camp_run/part_2_code/run_part_1.sh ${MPI_TEST_FLAG})
add_test(boot_camp_part_2 ${CMAKE_BINARY_DIR}/boot_camp_run/part_2_code/run_part_2.sh ${MPI_TEST_FLAG})
add_test(boot_camp_part_3 ${CMAKE_BINARY_DIR}/boot_camp_run/part_3_code/run_part_3.sh ${MPI_TEST_FLAG})
add_test(boot_camp_part_4 ${CMAKE_BINARY_DIR}/boot_camp_run/part_4_code/run_part_4.sh ${MPI_TEST_FLAG})
add_test(camp_v1_paper_binned ${CMAKE_BINARY_DIR}/data_run/CAMP_v1_paper/binned/test_monarch_binned.sh ${MPI_TEST_FLAG})
add_test(camp_v1_paper_modal ${CMAKE_BINARY_DIR}/data_run/CAMP_v1_paper/modal/test_monarch_modal.sh ${MPI_TEST_FLAG})

######################################################################
# camp library

add_subdirectory(src)

######################################################################
# camp-chem box model

add_executable(camp_box_model test/camp_box_model_data.F90
                              test/camp_box_model.F90)

target_link_libraries(camp_box_model camp::camp)

######################################################################
# test_chemistry_cb05cl_ae5

set(CB5_EBI_SOLVER
	test/chemistry/cb05cl_ae5/EXT-CONST_mod.F90
	test/chemistry/cb05cl_ae5/ext-hrg2.F90
	test/chemistry/cb05cl_ae5/ext-hrcalcks.F90
	test/chemistry/cb05cl_ae5/ext-hrinit.F90
	test/chemistry/cb05cl_ae5/ext-hrdata_mod.F90
	test/chemistry/cb05cl_ae5/ext-hrg3.F90
	test/chemistry/cb05cl_ae5/ext-hrprodloss.F90
	test/chemistry/cb05cl_ae5/ext-hrg1.F90
	test/chemistry/cb05cl_ae5/ext-hrg4.F90
	test/chemistry/cb05cl_ae5/ext-hrrates.F90
	test/chemistry/cb05cl_ae5/ext-hrsolver.F90
	test/chemistry/cb05cl_ae5/EXT-RXCM_mod.F90
	)

set(CB5_KPP_SOLVER
	test/chemistry/cb05cl_ae5/cb05cl_ae5_Function.f90
	test/chemistry/cb05cl_ae5/cb05cl_ae5_Global.f90
	test/chemistry/cb05cl_ae5/cb05cl_ae5_Initialize.f90
        test/chemistry/cb05cl_ae5/cb05cl_ae5_Integrator.F90
	test/chemistry/cb05cl_ae5/cb05cl_ae5_Jacobian.f90
	test/chemistry/cb05cl_ae5/cb05cl_ae5_JacobianSP.f90
	test/chemistry/cb05cl_ae5/cb05cl_ae5_LinearAlgebra.f90
	test/chemistry/cb05cl_ae5/cb05cl_ae5_Model.f90
	test/chemistry/cb05cl_ae5/cb05cl_ae5_Monitor.f90
	test/chemistry/cb05cl_ae5/cb05cl_ae5_Parameters.f90
	test/chemistry/cb05cl_ae5/cb05cl_ae5_Precision.f90
	test/chemistry/cb05cl_ae5/cb05cl_ae5_Rates.f90
	test/chemistry/cb05cl_ae5/cb05cl_ae5_Util.f90
	)

if(${CMAKE_Fortran_COMPILER_ID} MATCHES "Intel")
	set(CB5_EBI_FLAGS "-fixed -extend-source -warn truncated_source")
elseif(${CMAKE_Fortran_COMPILER_ID} MATCHES "GNU")
	set(CB5_EBI_FLAGS "-ffixed-form -ffixed-line-length-none")
endif()

set_source_files_properties(${CB5_EBI_SOLVER} PROPERTIES COMPILE_FLAGS
	${CB5_EBI_FLAGS})

######################################################################
# test_chemistry_cb05cl_ae5_big

add_executable(test_chemistry_cb05cl_ae5_big
        test/chemistry/cb05cl_ae5/test_cb05cl_ae5_big.F90
        ${CB5_EBI_SOLVER} ${CB5_KPP_SOLVER}
        test/chemistry/cb05cl_ae5/module_BSC_CHEM_DATA.F90)
target_link_libraries(test_chemistry_cb05cl_ae5_big camp::camp)

######################################################################
# MONARCH interface

if(NOT ${CMAKE_Fortran_COMPILER_ID} MATCHES "Intel")
  add_executable(mock_monarch test/monarch/mock_monarch.F90
          test/monarch/camp_monarch_interface.F90)
  target_link_libraries(mock_monarch camp::camp)
endif()

######################################################################
# test_chemistry_cb05cl_ae5

add_executable(test_chemistry_cb05cl_ae5
    test/chemistry/cb05cl_ae5/test_cb05cl_ae5.F90
    ${CB5_EBI_SOLVER} ${CB5_KPP_SOLVER}
    test/chemistry/cb05cl_ae5/module_BSC_CHEM_DATA.F90)
target_link_libraries(test_chemistry_cb05cl_ae5 camp::camp)

######################################################################
# test_property

add_executable(unit_test_property test/unit_property/test_property.F90)

target_link_libraries(unit_test_property camp::camp)

######################################################################
# test_jacobian

add_executable(unit_test_jacobian test/unit_jacobian/test_jacobian.c)

target_link_libraries(unit_test_jacobian camp::camp)

######################################################################
# test_chem_spec_data

add_executable(unit_test_chem_spec_data test/unit_chem_spec_data/test_chem_spec_data.F90)

target_link_libraries(unit_test_chem_spec_data camp::camp)

######################################################################
# test_rxn_data

add_executable(test_rxn_aqueous_equilibrium test/unit_rxn_data/test_rxn_aqueous_equilibrium.F90)
target_link_libraries(test_rxn_aqueous_equilibrium camp::camp)
add_executable(test_rxn_CMAQ_H2O2 test/unit_rxn_data/test_rxn_CMAQ_H2O2.F90)
target_link_libraries(test_rxn_CMAQ_H2O2 camp::camp)
add_executable(test_rxn_CMAQ_OH_HNO3 test/unit_rxn_data/test_rxn_CMAQ_OH_HNO3.F90)
target_link_libraries(test_rxn_CMAQ_OH_HNO3 camp::camp)
add_executable(test_rxn_condensed_phase_arrhenius test/unit_rxn_data/test_rxn_condensed_phase_arrhenius.F90)
target_link_libraries(test_rxn_condensed_phase_arrhenius camp::camp)
add_executable(test_rxn_condensed_phase_photolysis test/unit_rxn_data/test_rxn_condensed_phase_photolysis.F90)
target_link_libraries(test_rxn_condensed_phase_photolysis camp::camp)
add_executable(test_rxn_emission test/unit_rxn_data/test_rxn_emission.F90)
target_link_libraries(test_rxn_emission camp::camp)
add_executable(test_rxn_first_order_loss test/unit_rxn_data/test_rxn_first_order_loss.F90)
target_link_libraries(test_rxn_first_order_loss camp::camp)
add_executable(test_rxn_HL_phase_transfer test/unit_rxn_data/test_rxn_HL_phase_transfer.F90)
target_link_libraries(test_rxn_HL_phase_transfer camp::camp)
add_executable(test_rxn_photolysis test/unit_rxn_data/test_rxn_photolysis.F90)
target_link_libraries(test_rxn_photolysis camp::camp)
add_executable(test_rxn_SIMPOL_phase_transfer test/unit_rxn_data/test_rxn_SIMPOL_phase_transfer.F90)
target_link_libraries(test_rxn_SIMPOL_phase_transfer camp::camp)
add_executable(test_rxn_surface test/unit_rxn_data/test_rxn_surface.F90)
target_link_libraries(test_rxn_surface camp::camp)
add_executable(test_rxn_ternary_chemical_activation test/unit_rxn_data/test_rxn_ternary_chemical_activation.F90)
target_link_libraries(test_rxn_ternary_chemical_activation camp::camp)
add_executable(test_rxn_troe test/unit_rxn_data/test_rxn_troe.F90)
target_link_libraries(test_rxn_troe camp::camp)
add_executable(test_rxn_wennberg_no_ro2 test/unit_rxn_data/test_rxn_wennberg_no_ro2.F90)
target_link_libraries(test_rxn_wennberg_no_ro2 camp::camp)
add_executable(test_rxn_wennberg_tunneling test/unit_rxn_data/test_rxn_wennberg_tunneling.F90)
target_link_libraries(test_rxn_wennberg_tunneling camp::camp)
add_executable(test_rxn_wet_deposition test/unit_rxn_data/test_rxn_wet_deposition.F90)
target_link_libraries(test_rxn_wet_deposition camp::camp)

######################################################################
# test_sub_model_data

add_executable(test_sub_model_PDFiTE test/unit_sub_model_data/test_sub_model_PDFiTE.F90)
target_link_libraries(test_sub_model_PDFiTE camp::camp)
add_executable(test_sub_model_UNIFAC test/unit_sub_model_data/test_sub_model_UNIFAC.F90)
target_link_libraries(test_sub_model_UNIFAC camp::camp)
add_executable(test_sub_model_ZSR_aerosol_water
               test/unit_sub_model_data/test_sub_model_ZSR_aerosol_water.c
               test/unit_sub_model_data/test_sub_model_ZSR_aerosol_water.F90)
target_link_libraries(test_sub_model_ZSR_aerosol_water camp::camp)

######################################################################
# New unit tests (UNDER DEVELOPMENT)

if(NOT ${CMAKE_Fortran_COMPILER_ID} MATCHES "Intel")
  set(UNIT_TEST_SRC
      test/unit_tests/unit_test_data.F90
      test/unit_tests/unit_test_driver.F90)
  
  set(UNIT_TEST_RXN_ARRHENIUS_SRC
      test/unit_tests/rxns/unit_test_rxn_arrhenius.F90 ${UNIT_TEST_SRC})
  if(${CMAKE_Fortran_COMPILER_ID} MATCHES "Intel")
    set_source_files_properties(${UNIT_TEST_RXN_ARRHENIUS_SRC} PROPERTIES COMPILE_DEFINITIONS
      "UNIT_TEST_MODULE_=camp_unit_test_rxn_arrhenius \
      ;UNIT_TEST_TYPE_='unit_test_rxn_arrhenius_t()'")
  elseif(${CMAKE_Fortran_COMPILER_ID} MATCHES "GNU")
    set_source_files_properties(${UNIT_TEST_RXN_ARRHENIUS_SRC} PROPERTIES COMPILE_DEFINITIONS
      "UNIT_TEST_MODULE_=camp_unit_test_rxn_arrhenius \
      ;UNIT_TEST_TYPE_=unit_test_rxn_arrhenius_t()")
  endif()
  add_executable(unit_test_rxn_arrhenius ${UNIT_TEST_RXN_ARRHENIUS_SRC})
  target_link_libraries(unit_test_rxn_arrhenius camp::camp)
endif()

######################################################################
# test_camp_core

add_executable(unit_test_camp_core test/unit_camp_core/test_camp_core.F90)

target_link_libraries(unit_test_camp_core camp::camp)

######################################################################
# test_aero_phase_data

add_executable(unit_test_aero_phase_data test/unit_aero_phase_data/test_aero_phase_data.F90)

target_link_libraries(unit_test_aero_phase_data camp::camp)

######################################################################
# test_aero_rep_single_particle

add_executable(unit_test_aero_rep_single_particle
	test/unit_aero_rep_data/test_aero_rep_single_particle.c
	test/unit_aero_rep_data/test_aero_rep_single_particle.F90)

target_link_libraries(unit_test_aero_rep_single_particle camp::camp)

######################################################################
# test_aero_rep_modal_binned_mass

add_executable(unit_test_aero_rep_modal_binned_mass
        test/unit_aero_rep_data/test_aero_rep_modal_binned_mass.c
	test/unit_aero_rep_data/test_aero_rep_modal_binned_mass.F90)

target_link_libraries(unit_test_aero_rep_modal_binned_mass camp::camp)

######################################################################
# test_chem_mech_solver

add_executable(test_chem_mech_solver test/chemistry/test_chem_mech_solver.F90)

target_link_libraries(test_chem_mech_solver camp::camp)

######################################################################
# BootCAMP Tutorial Exercises
######################################################################

######################################################################
# Part 1

add_executable(boot_camp_part_1 doc/camp_tutorial/boot_camp/part_1_code/box_model.F90)

target_link_libraries(boot_camp_part_1 camp::camp)

######################################################################
# Part 3

add_executable(boot_camp_part_3 doc/camp_tutorial/boot_camp/part_3_code/box_model.F90)

target_link_libraries(boot_camp_part_3 camp::camp)

######################################################################
# Part 4

add_executable(boot_camp_part_4 doc/camp_tutorial/boot_camp/part_4_code/box_model.F90)

target_link_libraries(boot_camp_part_4 camp::camp)

######################################################################
# CAMP v1.0.0 paper configurations
######################################################################

add_executable(camp_v1_paper_binned data/CAMP_v1_paper/binned/camp_monarch_interface.F90
                                    data/CAMP_v1_paper/binned/mock_monarch.F90)

target_link_libraries(camp_v1_paper_binned camp::camp)

######################################################################

add_executable(camp_v1_paper_modal data/CAMP_v1_paper/modal/camp_monarch_interface.F90
                                   data/CAMP_v1_paper/modal/mock_monarch.F90)

target_link_libraries(camp_v1_paper_modal camp::camp)

######################################################################
