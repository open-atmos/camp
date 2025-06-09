################################################################################
# Copy test data

add_custom_target(copy_test ALL ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/test_run)
set_property(DIRECTORY PROPERTY ADDITIONAL_MAKE_CLEAN_FILES mechanisms_run scenarios_run boot_camp_run)

################################################################################
# Tests

add_subdirectory(chemistry)
add_subdirectory(unit_aero_phase_data)

################################################################################
