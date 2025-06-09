######################################################################
# Utility functions for testing

# Set the Valgrind suppressions file for tests
if(CAMP_ENABLE_MEMCHECK)
  find_program(MEMORYCHECK_COMMAND "valgrind")
  set(MEMCHECK_SUPPRESS "--suppressions=${PROJECT_SOURCE_DIR}/valgrind.supp")
endif()

######################################################################
# build and add a standard test linked to the camp library
function(create_standard_test)
  set(prefix TEST)
  set(singleValues NAME WORKING_DIRECTORY RESULT)
  set(multiValues SOURCES LIBRARIES)
  include(CMakeParseArguments)
  cmake_parse_arguments("${prefix}" "" "${singleValues}" "${multiValues}" ${ARGN})

  add_executable(test_${TEST_NAME} ${TEST_SOURCES})
  target_link_libraries(test_${TEST_NAME} PUBLIC camp::camp)

  if (${CMAKE_Fortran_COMPILER_ID} MATCHES "Intel" OR ${CMAKE_Fortran_COMPILER_ID} MATCHES "NVHPC")
    set_target_properties(test_${TEST_NAME} PROPERTIES LINKER_LANGUAGE Fortran)
  endif()

  # link additional libraries
  foreach(library ${TEST_LIBRARIES})
    target_link_libraries(test_${TEST_NAME} PUBLIC ${library})
  endforeach()

  if(NOT DEFINED TEST_WORKING_DIRECTORY)
    set(TEST_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
  endif()

  add_camp_test(${TEST_NAME} test_${TEST_NAME} "" ${TEST_WORKING_DIRECTORY} ${TEST_RESULT})
endfunction(create_standard_test)

######################################################################
# Add a test with the given name and executable

function(add_camp_test test_name test_binary test_args working_dir result_string)
  if(ENABLE_MPI)
    add_test(NAME ${test_name}
      COMMAND mpirun -v -np 2 ${CMAKE_BINARY_DIR}/${test_binary} ${test_args}
             WORKING_DIRECTORY ${working_dir})
  else()
    add_test(NAME ${test_name}
             COMMAND ${test_binary} ${test_args}
             WORKING_DIRECTORY ${working_dir})
    set_tests_properties(${test_name} PROPERTIES TIMEOUT 20)
  endif()
  set(MEMORYCHECK_COMMAND_OPTIONS "--error-exitcode=1 --trace-children=yes --leak-check=full --gen-suppressions=all ${MEMCHECK_SUPPRESS}")
  set(memcheck "${MEMORYCHECK_COMMAND} ${MEMORYCHECK_COMMAND_OPTIONS}")
  separate_arguments(memcheck)
  if(ENABLE_MPI AND MEMORYCHECK_COMMAND AND CAMP_ENABLE_MEMCHECK)
    add_test(NAME memcheck_${test_name}
      COMMAND mpirun -v -np 2 ${memcheck} ${CMAKE_BINARY_DIR}/${test_binary} ${test_args}
             WORKING_DIRECTORY ${working_dir})
    
    # add dependency between memcheck and previous test
    # https://stackoverflow.com/a/66931930/5217293
    set_tests_properties(${test_name} PROPERTIES FIXTURES_SETUP f_${test_name})
    set_tests_properties(memcheck_${test_name} PROPERTIES FIXTURES_REQUIRED f_${test_name})
  elseif(MEMORYCHECK_COMMAND AND CAMP_ENABLE_MEMCHECK)
    add_test(NAME memcheck_${test_name}
             COMMAND ${memcheck} ${CMAKE_BINARY_DIR}/${test_binary} ${test_args}
             WORKING_DIRECTORY ${working_dir})
  endif()
  if(NOT "${result_string}" STREQUAL "")
    set_tests_properties(${test_name}
      PROPERTIES PASS_REGULAR_EXPRESSION ${result_string})
  endif()
endfunction(add_camp_test)

######################################################################
