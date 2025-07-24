################################################################################
# Copy test data

add_custom_target(copy_test ALL ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/test_run)
set_property(DIRECTORY PROPERTY ADDITIONAL_MAKE_CLEAN_FILES mechanisms_run scenarios_run boot_camp_run)

################################################################################
# Tests

add_subdirectory(chemistry)
add_subdirectory(monarch)
add_subdirectory(unit_aero_phase_data)
add_subdirectory(unit_camp_core)
add_subdirectory(unit_chem_spec_data)
add_subdirectory(unit_aero_rep_data)
add_subdirectory(unit_jacobian)
add_subdirectory(unit_property)
add_subdirectory(unit_rxn_data)
add_subdirectory(unit_sub_model_data)
add_subdirectory(unit_tests)

######################################################################
# BootCAMP Tutorial Exercises
######################################################################
add_custom_target(copy_boot_camp ALL ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/doc/camp_tutorial/boot_camp ${CMAKE_BINARY_DIR}/boot_camp_run)
set_property(DIRECTORY PROPERTY ADDITIONAL_MAKE_CLEAN_FILES boot_camp_run)

######################################################################
# Part 1

add_executable(boot_camp_part_1
  ${PROJECT_SOURCE_DIR}/doc/camp_tutorial/boot_camp/part_1_code/box_model.F90
)
target_link_libraries(boot_camp_part_1 camp::camp)
add_test(NAME boot_camp_part_1
  COMMAND boot_camp_part_1
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/boot_camp_run/part_1_code
)

######################################################################
# Part 2

add_executable(boot_camp_part_2
  ${PROJECT_SOURCE_DIR}/doc/camp_tutorial/boot_camp/part_2_code/box_model.F90
)
target_link_libraries(boot_camp_part_2 camp::camp)
add_test(NAME boot_camp_part_2
  COMMAND boot_camp_part_2
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/boot_camp_run/part_2_code
)

######################################################################
# Part 3

add_executable(boot_camp_part_3
  ${PROJECT_SOURCE_DIR}/doc/camp_tutorial/boot_camp/part_3_code/box_model.F90
)
target_link_libraries(boot_camp_part_3 camp::camp)
add_test(NAME boot_camp_part_3
  COMMAND boot_camp_part_3
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/boot_camp_run/part_3_code
)

######################################################################
# Part 4

add_executable(boot_camp_part_4
  ${PROJECT_SOURCE_DIR}/doc/camp_tutorial/boot_camp/part_4_code/box_model.F90
)
target_link_libraries(boot_camp_part_4 camp::camp)
add_test(NAME boot_camp_part_4
  COMMAND boot_camp_part_4
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/boot_camp_run/part_4_code
)

######################################################################
# CAMP v1.0.0 paper configurations
######################################################################

add_custom_target(copy_data ALL ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/data ${CMAKE_BINARY_DIR}/data_run)
set_property(DIRECTORY PROPERTY ADDITIONAL_MAKE_CLEAN_FILES data)

add_executable(camp_v1_paper_binned ${PROJECT_SOURCE_DIR}/data/CAMP_v1_paper/binned/camp_monarch_interface.F90
                                    ${PROJECT_SOURCE_DIR}/data/CAMP_v1_paper/binned/mock_monarch.F90)

target_link_libraries(camp_v1_paper_binned camp::camp)
if (ENABLE_MPI)
  add_test(camp_v1_paper_binned ${CMAKE_BINARY_DIR}/data_run/CAMP_v1_paper/binned/test_monarch_binned.sh MPI)
else()
  add_test(camp_v1_paper_binned ${CMAKE_BINARY_DIR}/data_run/CAMP_v1_paper/binned/test_monarch_binned.sh serial)
endif()

######################################################################

add_executable(camp_v1_paper_modal ${PROJECT_SOURCE_DIR}/data/CAMP_v1_paper/modal/camp_monarch_interface.F90
                                   ${PROJECT_SOURCE_DIR}/data/CAMP_v1_paper/modal/mock_monarch.F90)

target_link_libraries(camp_v1_paper_modal camp::camp)
if (ENABLE_MPI)
  add_test(camp_v1_paper_modal ${CMAKE_BINARY_DIR}/data_run/CAMP_v1_paper/modal/test_monarch_modal.sh MPI)
else()
  add_test(camp_v1_paper_modal ${CMAKE_BINARY_DIR}/data_run/CAMP_v1_paper/modal/test_monarch_modal.sh serial)
endif()

######################################################################
