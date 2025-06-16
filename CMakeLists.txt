cmake_minimum_required(VERSION 3.21)

project(camp-distribution VERSION 1.1.0 LANGUAGES Fortran C)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_USER_MAKE_RULES_OVERRIDE ${PROJECT_SOURCE_DIR}/cmake/set_defaults.cmake)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

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

# Set the Valgrind suppressions file for tests
if(CAMP_ENABLE_MEMCHECK)
  find_program(MEMORYCHECK_COMMAND "valgrind")
  set(MEMCHECK_SUPPRESS "--suppressions=${PROJECT_SOURCE_DIR}/valgrind.supp")
endif()

include(GNUInstallDirs)

######################################################################
# options

option(ENABLE_MPI "Enable MPI parallel support" OFF)
option(ENABLE_DEBUG "Compile debugging functions" OFF)
option(FAILURE_DETAIL "Output conditions before and after solver failures" OFF)
option(ENABLE_CXX "Enable C++" OFF)
option(ENABLE_GPU "Enable use of GPUs in chemistry solving" OFF)
option(CAMP_ENABLE_MEMCHECK "Enable memory checking with Valgrind" OFF)

mark_as_advanced(FORCE ENABLE_DEBUG FAILURE_DETAIL)

################################################################################
# Projet wide setup variables 

set(CAMP_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR})
set(CAMP_MOD_DIR ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR})
set(CAMP_LIB_DIR ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CAMP_PROJECT_SRC_DIR ${PROJECT_SOURCE_DIR})
set(camp_compile_definitions "")

include(${PROJECT_SOURCE_DIR}/cmake/dependencies.cmake)

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
# camp library

add_subdirectory(src)

######################################################################
# copy mechanism data

add_custom_target(copy_mechanism ALL ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/mechanisms ${CMAKE_BINARY_DIR}/mechanisms_run)
set_property(DIRECTORY PROPERTY ADDITIONAL_MAKE_CLEAN_FILES mechanisms)

######################################################################
# test

if (PROJECT_IS_TOP_LEVEL)
  enable_testing()
  add_subdirectory(test)
endif()

######################################################################
