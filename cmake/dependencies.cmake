######################################################################
# CPack

set(CPACK_SOURCE_GENERATOR "TGZ")
SET(CPACK_SOURCE_PACKAGE_FILE_NAME "${PACKAGE_TARNAME}-${PACKAGE_VERSION}")
set(CPACK_SOURCE_IGNORE_FILES "${CPACK_SOURCE_IGNORE_FILES};/.*~$;/[.].*/;/build/;/figures/;/scenarios/[^1234].*/;/old/;/tool/;/TODO")

include(CPack)

######################################################################
# MPI

if(ENABLE_MPI)
  list(APPEND camp_compile_definitions CAMP_USE_MPI)
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
list(APPEND camp_compile_definitions CAMP_USE_SUNDIALS)
list(APPEND camp_compile_definitions CAMP_CUSTOM_CVODE)

######################################################################
# json-fortran

find_path(JSON_INCLUDE_DIR json_module.mod
  DOC "json-fortran include directory (must include json_*.mod files)"
  PATHS $ENV{JSON_FORTRAN_HOME}/lib /opt/local/lib /usr/local/lib)
find_library(JSON_LIB jsonfortran
  DOC "json-fortran library"
  PATHS $ENV{JSON_FORTRAN_HOME}/lib /opt/local/lib /usr/local/lib)
include_directories(${JSON_INCLUDE_DIR})
list(APPEND camp_compile_definitions CAMP_USE_JSON)

######################################################################