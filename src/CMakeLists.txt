################################################################################
# Preamble

project(
  camp
  VERSION ${PROJECT_VERSION}
  LANGUAGES C CXX
)

message (STATUS "CMake build configuration for ${PROJECT_NAME} (${CMAKE_BUILD_TYPE}) ${PROJECT_VERSION}")

##############################################################################
# CAMP targets

add_library(camp)
add_library(camp::camp ALIAS camp)

include(silence_warnings)
silence_warnings(camp)

target_sources(camp
  PRIVATE
    aero_phase_data.F90
    aero_phase_solver.c
    aero_rep_data.F90
    aero_rep_factory.F90
    aero_rep_solver.c
    camp_core.F90
    camp_solver_data.F90
    camp_solver.c
    camp_state.F90
    chem_spec_data.F90
    constants.F90
    debug_diff_check.c
    debug_diff_check.F90
    env_state.F90
    Jacobian.c
    mechanism_data.F90
    mpi.F90
    property.F90
    rand.F90
    rxn_data.F90
    rxn_factory.F90
    rxn_solver.c
    solver_stats.F90
    sub_model_data.F90
    sub_model_factory.F90
    sub_model_solver.c
    time_derivative.c
    util.F90
)

target_link_libraries(camp
  PUBLIC
    ${SUNDIALS_LIBS}
    ${JSON_LIB}
)

target_compile_definitions(camp PUBLIC ${camp_compile_definitions})

if(${CMAKE_Fortran_COMPILER_ID} MATCHES "Intel")
  target_compile_options(camp PUBLIC "-extend-source -warn truncated_source")
elseif(${CMAKE_Fortran_COMPILER_ID} MATCHES "GNU")
  target_compile_options(camp PUBLIC -ffree-line-length-none)
endif()

set_target_properties(camp
  PROPERTIES
    OUTPUT_NAME camp
    PREFIX lib
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION}
    LIBRARY_OUTPUT_DIRECTORY ${CAMP_LIB_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${CAMP_LIB_DIR}
    Fortran_MODULE_DIRECTORY ${CAMP_MOD_DIR}
)

target_include_directories(camp
  PUBLIC
    $<BUILD_INTERFACE:${CAMP_PROJECT_SRC_DIR}/include>
    $<BUILD_INTERFACE:${CAMP_MOD_DIR}>
    $<INSTALL_INTERFACE:${CAMP_INSTALL_INCLUDE_DIR}>
)

##############################################################################
# CAMP box model

add_executable(camp_box_model camp_box_model_data.F90
                              camp_box_model.F90)

target_link_libraries(camp_box_model camp::camp)

##############################################################################
# Subdirectories

add_subdirectory(aero_reps)
add_subdirectory(rxns)
add_subdirectory(sub_models)
if (ENABLE_CUDA)
  add_subdirectory(cuda)
endif ()

##############################################################################
# packaging

add_subdirectory(packaging)
