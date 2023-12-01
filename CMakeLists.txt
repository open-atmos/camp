cmake_minimum_required(VERSION 3.0)
# Require 2.6.4 due to -I/usr/include behavior:
# http://www.cmake.org/Bug/view.php?id=8598
# http://gcc.gnu.org/bugzilla/show_bug.cgi?id=35707
# http://www.cmake.org/Bug/view.php?id=8408
project(CAMP Fortran C)

set(PACKAGE_BUGREPORT "mattldawson@gmail.com")
set(PACKAGE_NAME "CAMP")
set(PACKAGE_STRING "CAMP 1.1.0")
set(PACKAGE_TARNAME "camp")
set(PACKAGE_VERSION "1.1.0")

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

######################################################################
# options

option(ENABLE_GSL "Enable GSL library for Jacobian evaluation" OFF)
option(ENABLE_MPI "Enable MPI parallel support" OFF)
option(ENABLE_DEBUG "Compile debugging functions" OFF)
option(FAILURE_DETAIL "Output conditions before and after solver failures" OFF)
option(ENABLE_CXX "Enable C++" OFF)
option(ENABLE_GPU "Enable use of GPUs in chemistry solving" OFF)

mark_as_advanced(FORCE ENABLE_DEBUG FAILURE_DETAIL)

######################################################################
# CPack

set(CPACK_SOURCE_GENERATOR "TGZ")
SET(CPACK_SOURCE_PACKAGE_FILE_NAME "${PACKAGE_TARNAME}-${PACKAGE_VERSION}")
set(CPACK_SOURCE_IGNORE_FILES "${CPACK_SOURCE_IGNORE_FILES};/.*~$;/[.].*/;/build/;/figures/;/scenarios/[^1234].*/;/old/;/tool/;/TODO")

include(CPack)

######################################################################
# GSL

if(ENABLE_GSL)
  find_path(GSL_INCLUDE_DIR gsl/gsl_math.h
    DOC "GSL include directory (must have gsl/ subdir)"
    PATHS $ENV{GSL_HOME}/include /opt/local/include)
  find_library(GSL_LIB gsl
    DOC "GSL library"
    PATHS $ENV{GSL_HOME}/lib /opt/local/lib)
  find_library(GSL_CBLAS_LIB gslcblas
    DOC "GSL CBLAS library"
    PATHS $ENV{GSL_HOME}/lib /opt/local/lib)
  find_library(M_LIB m
    DOC "standard C math library")
  set(GSL_SRC src/rand_gsl.c)
  set(GSL_LIBS ${GSL_LIB} ${GSL_CBLAS_LIB} ${M_LIB})
  include_directories(${GSL_INCLUDE_DIR})
  add_definitions(-DCAMP_USE_GSL)
endif()

######################################################################
# MPI

if(ENABLE_MPI)
  add_definitions(-DCAMP_USE_MPI)
endif()

######################################################################
# SUNDIALS

find_path(SUITE_SPARSE_INCLUDE_DIR klu.h
  DOC "SuiteSparse include directory (must have klu.h)"
  PATHS $ENV{SUITE_SPARSE_HOME}/include $ENV{SUNDIALS_HOME}/include
        /opt/local/include /usr/local/include)
find_library(SUITE_SPARSE_KLU_LIB klu
  DOC "SuiteSparse klu library"
  PATHS $ENV{SUITE_SPARSE_HOME}/lib $ENV{SUNDIALS_HOME}/lib
        /opt/local/lib /usr/local/lib)
find_library(SUITE_SPARSE_AMD_LIB amd
  DOC "SuiteSparse amd library"
  PATHS $ENV{SUITE_SPARSE_HOME}/lib $ENV{SUNDIALS_HOME}/lib
        /opt/local/lib /usr/local/lib)
find_library(SUITE_SPARSE_BTF_LIB btf
  DOC "SuiteSparse btf library"
  PATHS $ENV{SUITE_SPARSE_HOME}/lib $ENV{SUNDIALS_HOME}/lib
        /opt/local/lib /usr/local/lib)
find_library(SUITE_SPARSE_COLAMD_LIB colamd
  DOC "SuiteSparse colamd library"
  PATHS $ENV{SUITE_SPARSE_HOME}/lib $ENV{SUNDIALS_HOME}/lib
        /opt/local/lib /usr/local/lib)
find_library(SUITE_SPARSE_CONFIG_LIB suitesparseconfig
  DOC "SuiteSparse config library"
  PATHS $ENV{SUITE_SPARSE_HOME}/lib $ENV{SUNDIALS_HOME}/lib
        /opt/local/lib /usr/local/lib)
find_path(SUNDIALS_INCLUDE_DIR cvode/cvode.h
  DOC "SUNDIALS include directory (must have cvode/, sundials/, nvector/ subdirs)"
  PATHS $ENV{SUNDIALS_HOME}/include /opt/local/include /usr/local/include)
find_library(SUNDIALS_NVECSERIAL_LIB sundials_nvecserial
  DOC "SUNDIALS serial vector library"
  PATHS $ENV{SUNDIALS_HOME}/lib /opt/local/lib /usr/local/lib)
find_library(SUNDIALS_CVODE_LIB sundials_cvode
  DOC "SUNDIALS CVODE library"
  PATHS $ENV{SUNDIALS_HOME}/lib /opt/local/lib /usr/local/lib)
find_library(SUNDIALS_KLU_LIB sundials_sunlinsolklu
  DOC "SUNDIALS KLU library"
  PATHS $ENV{SUITE_SPARSE_HOME}/lib $ENV{SUNDIALS_HOME}/lib
        /opt/local/lib /usr/local/lib)
find_library(SUNDIALS_SUNMATRIX_SPARSE_LIB sundials_sunmatrixsparse
  DOC "SUNDIALS SUNMatrixSparse library"
  PATHS $ENV{SUITE_SPARSE_HOME}/lib $ENV{SUNDIALS_HOME}/lib
        /opt/local/lib /usr/local/lib)
set(SUNDIALS_LIBS ${SUNDIALS_NVECSERIAL_LIB} ${SUNDIALS_CVODE_LIB}
  ${SUNDIALS_KLU_LIB} ${SUNDIALS_SUNMATRIX_SPARSE_LIB} ${SUITE_SPARSE_KLU_LIB}
  ${SUITE_SPARSE_COLAMD_LIB} ${SUITE_SPARSE_AMD_LIB} ${SUITE_SPARSE_BTF_LIB}
  ${SUITE_SPARSE_CONFIG_LIB})
include_directories(${SUNDIALS_INCLUDE_DIR} ${SUITE_SPARSE_INCLUDE_DIR})
add_definitions(-DCAMP_USE_SUNDIALS)
add_definitions(-DCAMP_CUSTOM_CVODE)

######################################################################
# json-fortran

find_path(JSON_INCLUDE_DIR json_module.mod
  DOC "json-fortran include directory (must include json_*.mod files)"
  PATHS $ENV{JSON_FORTRAN_HOME}/lib /opt/local/lib /usr/local/lib)
find_library(JSON_LIB jsonfortran
  DOC "json-fortran library"
  PATHS $ENV{JSON_FORTRAN_HOME}/lib /opt/local/lib /usr/local/lib)
include_directories(${JSON_INCLUDE_DIR})
add_definitions(-DCAMP_USE_JSON)

######################################################################
##CXX

if(ENABLE_CXX)
  add_definitions(-DCAMP_USE_CXX)
  enable_language(CXX)
endif()

######################################################################
# GPU support

if (ENABLE_GPU)
  add_definitions(-DCAMP_USE_GPU)
  enable_language(CUDA)
endif()

######################################################################
# debugging options

if (ENABLE_DEBUG)
  add_definitions(-DCAMP_DEBUG)
endif()
if (FAILURE_DETAIL)
  add_definitions(-DFAILURE_DETAIL)
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
add_test(test_rxn_arrhenius_mech ${CMAKE_BINARY_DIR}/test_run/unit_tests/input_files/run_rxn_arrhenius.sh ${MPI_TEST_FLAG})

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
add_test(test_MONARCH_1 ${CMAKE_BINARY_DIR}/test_run/monarch/test_monarch_1.sh ${MPI_TEST_FLAG})
add_test(test_MONARCH_2 ${CMAKE_BINARY_DIR}/test_run/monarch/test_monarch_2.sh ${MPI_TEST_FLAG})
add_test(boot_camp_part_1 ${CMAKE_BINARY_DIR}/boot_camp_run/part_2_code/run_part_1.sh ${MPI_TEST_FLAG})
add_test(boot_camp_part_2 ${CMAKE_BINARY_DIR}/boot_camp_run/part_2_code/run_part_2.sh ${MPI_TEST_FLAG})
add_test(boot_camp_part_3 ${CMAKE_BINARY_DIR}/boot_camp_run/part_3_code/run_part_3.sh ${MPI_TEST_FLAG})
add_test(boot_camp_part_4 ${CMAKE_BINARY_DIR}/boot_camp_run/part_4_code/run_part_4.sh ${MPI_TEST_FLAG})
add_test(camp_v1_paper_binned ${CMAKE_BINARY_DIR}/data_run/CAMP_v1_paper/binned/test_monarch_binned.sh ${MPI_TEST_FLAG})
add_test(camp_v1_paper_modal ${CMAKE_BINARY_DIR}/data_run/CAMP_v1_paper/modal/test_monarch_modal.sh ${MPI_TEST_FLAG})

######################################################################
# camp library

set(STD_C_FLAGS "-std=c99 -Werror=format")
set(STD_CUDA_FLAGS "-dc -arch=compute_70 -code=sm_70")

if(${CMAKE_Fortran_COMPILER_ID} MATCHES "Intel")
  set(STD_F_FLAGS "-extend_source -warn truncated_source")
elseif(${CMAKE_Fortran_COMPILER_ID} MATCHES "GNU")
  set(STD_F_FLAGS "-ffree-line-length-none")
endif()

set(REACTIONS_F_SRC
	src/rxns/rxn_aqueous_equilibrium.F90
	src/rxns/rxn_arrhenius.F90
	src/rxns/rxn_CMAQ_H2O2.F90
	src/rxns/rxn_CMAQ_OH_HNO3.F90
	src/rxns/rxn_condensed_phase_arrhenius.F90
	src/rxns/rxn_condensed_phase_photolysis.F90
	src/rxns/rxn_emission.F90
	src/rxns/rxn_first_order_loss.F90
	src/rxns/rxn_HL_phase_transfer.F90
	src/rxns/rxn_photolysis.F90
  src/rxns/rxn_SIMPOL_phase_transfer.F90
  src/rxns/rxn_surface.F90
  src/rxns/rxn_ternary_chemical_activation.F90
  src/rxns/rxn_troe.F90
  src/rxns/rxn_wennberg_no_ro2.F90
  src/rxns/rxn_wennberg_tunneling.F90
  src/rxns/rxn_wet_deposition.F90)

set(REACTIONS_C_SRC
	src/rxns/rxn_aqueous_equilibrium.c
	src/rxns/rxn_arrhenius.c
	src/rxns/rxn_CMAQ_H2O2.c
	src/rxns/rxn_CMAQ_OH_HNO3.c
	src/rxns/rxn_condensed_phase_arrhenius.c
	src/rxns/rxn_condensed_phase_photolysis.c
	src/rxns/rxn_emission.c
	src/rxns/rxn_first_order_loss.c
	src/rxns/rxn_HL_phase_transfer.c
	src/rxns/rxn_photolysis.c
  src/rxns/rxn_SIMPOL_phase_transfer.c
  src/rxns/rxn_surface.c
  src/rxns/rxn_ternary_chemical_activation.c
	src/rxns/rxn_troe.c
  src/rxns/rxn_wennberg_no_ro2.c
  src/rxns/rxn_wennberg_tunneling.c
  src/rxns/rxn_wet_deposition.c)

set_source_files_properties(${REACTIONS_F_SRC} PROPERTIES COMPILE_FLAGS
        ${STD_F_FLAGS})

set_source_files_properties(${REACTIONS_C_SRC} PROPERTIES COMPILE_FLAGS
        ${STD_C_FLAGS})

set(REACTIONS_SRC ${REACTIONS_F_SRC} ${REACTIONS_C_SRC})

set(AEROSOL_REPS_F_SRC
        src/aero_reps/aero_rep_modal_binned_mass.F90
        src/aero_reps/aero_rep_single_particle.F90)

set(AEROSOL_REPS_C_SRC
        src/aero_reps/aero_rep_modal_binned_mass.c
        src/aero_reps/aero_rep_single_particle.c)

set_source_files_properties(${AEROSOL_REPS_F_SRC} PROPERTIES COMPILE_FLAGS
        ${STD_F_FLAGS})

set_source_files_properties(${AEROSOL_REPS_C_SRC} PROPERTIES COMPILE_FLAGS
        ${STD_C_FLAGS})

set(AEROSOL_REPS_SRC ${AEROSOL_REPS_F_SRC} ${AEROSOL_REPS_C_SRC})

set(SUB_MODELS_F_SRC
  src/sub_models/sub_model_PDFiTE.F90
        src/sub_models/sub_model_UNIFAC.F90
        src/sub_models/sub_model_ZSR_aerosol_water.F90)

set(SUB_MODELS_C_SRC
  src/sub_models/sub_model_PDFiTE.c
        src/sub_models/sub_model_UNIFAC.c
        src/sub_models/sub_model_ZSR_aerosol_water.c)

set_source_files_properties(${SUB_MODELS_F_SRC} PROPERTIES COMPILE_FLAGS
        ${STD_F_FLAGS})

set_source_files_properties(${SUB_MODELS_C_SRC} PROPERTIES COMPILE_FLAGS
        ${STD_C_FLAGS})

set(SUB_MODELS_SRC ${SUB_MODELS_F_SRC} ${SUB_MODELS_C_SRC})

set(CAMP_C_SRC
        src/camp_solver.c src/rxn_solver.c src/aero_phase_solver.c
        src/aero_rep_solver.c src/sub_model_solver.c
        src/time_derivative.c src/Jacobian.c src/debug_diff_check.c)

set_source_files_properties(${CAMP_C_SRC} PROPERTIES COMPILE_FLAGS
        ${STD_C_FLAGS})

set(CAMP_CXX_SRC "")

if(ENABLE_GPU)
  set(CAMP_CUDA_SRC
          src/cuda/camp_gpu_solver.cu
          src/cuda/rxns_gpu/rxn_aqueous_equilibrium.cu
          src/cuda/rxns_gpu/rxn_arrhenius.cu
          src/cuda/rxns_gpu/rxn_CMAQ_H2O2.cu
          src/cuda/rxns_gpu/rxn_CMAQ_OH_HNO3.cu
          src/cuda/rxns_gpu/rxn_condensed_phase_arrhenius.cu
          src/cuda/rxns_gpu/rxn_emission.cu
          src/cuda/rxns_gpu/rxn_first_order_loss.cu
          src/cuda/rxns_gpu/rxn_HL_phase_transfer.cu
          src/cuda/rxns_gpu/rxn_photolysis.cu
          src/cuda/rxns_gpu/rxn_SIMPOL_phase_transfer.cu
          src/cuda/rxns_gpu/rxn_troe.cu
          src/cuda/rxns_gpu/rxn_wet_deposition.cu
          src/cuda/aeros_gpu/aero_phase_solver.cu
          src/cuda/aeros_gpu/aero_rep_solver.cu
          src/cuda/aeros_gpu/sub_model_solver.cu
          src/cuda/aero_reps_gpu/aero_rep_modal_binned_mass.cu
          src/cuda/aero_reps_gpu/aero_rep_single_particle.cu)

  set_source_files_properties(${CAMP_CUDA_SRC} PROPERTIES COMPILE_FLAGS
        ${STD_CUDA_FLAGS})
  set_source_files_properties(${CAMP_CUDA_SRC} PROPERTIES LANGUAGE CUDA)
else()
  set(CAMP_CUDA_SRC "")
endif()

set(CAMP_LIB_SRC
  src/constants.F90
  src/util.F90
  src/mpi.F90
  src/env_state.F90
  src/rand.F90
  src/property.F90 src/chem_spec_data.F90
  src/rxn_data.F90 src/camp_state.F90 src/mechanism_data.F90
  src/camp_core.F90 src/camp_solver_data.F90 src/aero_rep_data.F90
  src/aero_phase_data.F90 src/aero_rep_factory.F90
  src/rxn_factory.F90 src/sub_model_data.F90 src/sub_model_factory.F90
  src/solver_stats.F90
  src/debug_diff_check.F90
  ${CAMP_C_SRC} ${AEROSOL_REPS_SRC} ${SUB_MODELS_SRC} ${REACTIONS_SRC}
  ${CAMP_CUDA_SRC} ${GSL_SRC} ${CAMP_CXX_SRC} )

add_library(camplib SHARED ${CAMP_LIB_SRC})
add_library(camplib-static STATIC ${CAMP_LIB_SRC})

target_link_libraries(camplib ${SUNDIALS_LIBS} ${GSL_LIBS} ${JSON_LIB})
target_link_libraries(camplib-static ${SUNDIALS_LIBS} ${GSL_LIBS} ${JSON_LIB})

set(MODULE_DIR "${CMAKE_BINARY_DIR}/include")

set_target_properties(camplib PROPERTIES
                         OUTPUT_NAME camp
                         PREFIX lib
                         VERSION ${PACKAGE_VERSION}
                         SOVERSION ${PACKAGE_VERSION}
                         LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
                         Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR})
set_target_properties(camplib-static PROPERTIES
                         OUTPUT_NAME camp
                         PREFIX lib
                         VERSION ${PACKAGE_VERSION}
                         ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
                         Fortran_MODULE_DIRECTORY ${MODULE_DIR})

######################################################################
# Install options

include(GNUInstallDirs)
install(TARGETS camplib camplib-static
        EXPORT camplib-targets
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
       )

set(MOD_DIR_TO_INSTALL "${MODULE_DIR}")
set(MOD_DESTINATION_DIR "${CMAKE_INSTALL_INCLUDEDIR}/camp")
install(
  CODE "file(GLOB_RECURSE MODULE_FILES \"${MOD_DIR_TO_INSTALL}/*.mod\")"
  CODE "file(INSTALL \${MODULE_FILES} DESTINATION \"\${CMAKE_INSTALL_PREFIX}/${MOD_DESTINATION_DIR}\")"
  )

######################################################################
# camp-chem box model

add_executable(camp_box_model test/camp_box_model_data.F90
                              test/camp_box_model.F90)

target_link_libraries(camp_box_model camplib)

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
	set(CB5_EBI_FLAGS "-fixed -extend_source -warn truncated_source")
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
target_link_libraries(test_chemistry_cb05cl_ae5_big camplib)

######################################################################
# MONARCH interface

add_executable(mock_monarch test/monarch/mock_monarch.F90
        test/monarch/camp_monarch_interface.F90)

target_link_libraries(mock_monarch camplib)

######################################################################
# test_chemistry_cb05cl_ae5

add_executable(test_chemistry_cb05cl_ae5
    test/chemistry/cb05cl_ae5/test_cb05cl_ae5.F90
    ${CB5_EBI_SOLVER} ${CB5_KPP_SOLVER}
    test/chemistry/cb05cl_ae5/module_BSC_CHEM_DATA.F90)
target_link_libraries(test_chemistry_cb05cl_ae5 camplib)

######################################################################
# test_property

add_executable(unit_test_property test/unit_property/test_property.F90)

target_link_libraries(unit_test_property camplib)

######################################################################
# test_jacobian

add_executable(unit_test_jacobian test/unit_jacobian/test_jacobian.c)

target_link_libraries(unit_test_jacobian camplib)

######################################################################
# test_chem_spec_data

add_executable(unit_test_chem_spec_data test/unit_chem_spec_data/test_chem_spec_data.F90)

target_link_libraries(unit_test_chem_spec_data camplib)

######################################################################
# test_rxn_data

add_executable(test_rxn_aqueous_equilibrium test/unit_rxn_data/test_rxn_aqueous_equilibrium.F90)
target_link_libraries(test_rxn_aqueous_equilibrium camplib)
add_executable(test_rxn_CMAQ_H2O2 test/unit_rxn_data/test_rxn_CMAQ_H2O2.F90)
target_link_libraries(test_rxn_CMAQ_H2O2 camplib)
add_executable(test_rxn_CMAQ_OH_HNO3 test/unit_rxn_data/test_rxn_CMAQ_OH_HNO3.F90)
target_link_libraries(test_rxn_CMAQ_OH_HNO3 camplib)
add_executable(test_rxn_condensed_phase_arrhenius test/unit_rxn_data/test_rxn_condensed_phase_arrhenius.F90)
target_link_libraries(test_rxn_condensed_phase_arrhenius camplib)
add_executable(test_rxn_condensed_phase_photolysis test/unit_rxn_data/test_rxn_condensed_phase_photolysis.F90)
target_link_libraries(test_rxn_condensed_phase_photolysis camplib)
add_executable(test_rxn_emission test/unit_rxn_data/test_rxn_emission.F90)
target_link_libraries(test_rxn_emission camplib)
add_executable(test_rxn_first_order_loss test/unit_rxn_data/test_rxn_first_order_loss.F90)
target_link_libraries(test_rxn_first_order_loss camplib)
add_executable(test_rxn_HL_phase_transfer test/unit_rxn_data/test_rxn_HL_phase_transfer.F90)
target_link_libraries(test_rxn_HL_phase_transfer camplib)
add_executable(test_rxn_photolysis test/unit_rxn_data/test_rxn_photolysis.F90)
target_link_libraries(test_rxn_photolysis camplib)
add_executable(test_rxn_SIMPOL_phase_transfer test/unit_rxn_data/test_rxn_SIMPOL_phase_transfer.F90)
target_link_libraries(test_rxn_SIMPOL_phase_transfer camplib)
add_executable(test_rxn_surface test/unit_rxn_data/test_rxn_surface.F90)
target_link_libraries(test_rxn_surface camplib)
add_executable(test_rxn_ternary_chemical_activation test/unit_rxn_data/test_rxn_ternary_chemical_activation.F90)
target_link_libraries(test_rxn_ternary_chemical_activation camplib)
add_executable(test_rxn_troe test/unit_rxn_data/test_rxn_troe.F90)
target_link_libraries(test_rxn_troe camplib)
add_executable(test_rxn_wennberg_no_ro2 test/unit_rxn_data/test_rxn_wennberg_no_ro2.F90)
target_link_libraries(test_rxn_wennberg_no_ro2 camplib)
add_executable(test_rxn_wennberg_tunneling test/unit_rxn_data/test_rxn_wennberg_tunneling.F90)
target_link_libraries(test_rxn_wennberg_tunneling camplib)
add_executable(test_rxn_wet_deposition test/unit_rxn_data/test_rxn_wet_deposition.F90)
target_link_libraries(test_rxn_wet_deposition camplib)

######################################################################
# test_sub_model_data

add_executable(test_sub_model_PDFiTE test/unit_sub_model_data/test_sub_model_PDFiTE.F90)
target_link_libraries(test_sub_model_PDFiTE camplib)
add_executable(test_sub_model_UNIFAC test/unit_sub_model_data/test_sub_model_UNIFAC.F90)
target_link_libraries(test_sub_model_UNIFAC camplib)
add_executable(test_sub_model_ZSR_aerosol_water
               test/unit_sub_model_data/test_sub_model_ZSR_aerosol_water.c
               test/unit_sub_model_data/test_sub_model_ZSR_aerosol_water.F90)
target_link_libraries(test_sub_model_ZSR_aerosol_water camplib)

######################################################################
# New unit tests (UNDER DEVELOPMENT)

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
set_source_files_properties(${UNIT_TEST_RXN_ARRHENIUS_SRC} PROPERTIES COMPILE_FLAGS
                            ${STD_F_FLAGS})
add_executable(unit_test_rxn_arrhenius ${UNIT_TEST_RXN_ARRHENIUS_SRC})
target_link_libraries(unit_test_rxn_arrhenius camplib)

######################################################################
# test_camp_core

add_executable(unit_test_camp_core test/unit_camp_core/test_camp_core.F90)

target_link_libraries(unit_test_camp_core camplib)

######################################################################
# test_aero_phase_data

add_executable(unit_test_aero_phase_data test/unit_aero_phase_data/test_aero_phase_data.F90)

target_link_libraries(unit_test_aero_phase_data camplib)

######################################################################
# test_aero_rep_single_particle

add_executable(unit_test_aero_rep_single_particle
	test/unit_aero_rep_data/test_aero_rep_single_particle.c
	test/unit_aero_rep_data/test_aero_rep_single_particle.F90)

target_link_libraries(unit_test_aero_rep_single_particle camplib)

######################################################################
# test_aero_rep_modal_binned_mass

add_executable(unit_test_aero_rep_modal_binned_mass
        test/unit_aero_rep_data/test_aero_rep_modal_binned_mass.c
	test/unit_aero_rep_data/test_aero_rep_modal_binned_mass.F90)

target_link_libraries(unit_test_aero_rep_modal_binned_mass camplib)

######################################################################
# test_chem_mech_solver

add_executable(test_chem_mech_solver test/chemistry/test_chem_mech_solver.F90)

target_link_libraries(test_chem_mech_solver camplib)

######################################################################
# BootCAMP Tutorial Exercises
######################################################################

######################################################################
# Part 1

add_executable(boot_camp_part_1 doc/camp_tutorial/boot_camp/part_1_code/box_model.F90)

target_link_libraries(boot_camp_part_1 camplib)

######################################################################
# Part 3

add_executable(boot_camp_part_3 doc/camp_tutorial/boot_camp/part_3_code/box_model.F90)

target_link_libraries(boot_camp_part_3 camplib)

######################################################################
# Part 4

add_executable(boot_camp_part_4 doc/camp_tutorial/boot_camp/part_4_code/box_model.F90)

target_link_libraries(boot_camp_part_4 camplib)

######################################################################
# CAMP v1.0.0 paper configurations
######################################################################

add_executable(camp_v1_paper_binned data/CAMP_v1_paper/binned/camp_monarch_interface.F90
                                    data/CAMP_v1_paper/binned/mock_monarch.F90)

target_link_libraries(camp_v1_paper_binned camplib)

######################################################################

add_executable(camp_v1_paper_modal data/CAMP_v1_paper/modal/camp_monarch_interface.F90
                                   data/CAMP_v1_paper/modal/mock_monarch.F90)

target_link_libraries(camp_v1_paper_modal camplib)

######################################################################
