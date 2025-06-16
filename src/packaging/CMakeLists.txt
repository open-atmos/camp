include(CMakePackageConfigHelpers)

install(
  TARGETS
    camp
  EXPORT
    camp_Exports
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
  DIRECTORY
    ${CAMP_PROJECT_SRC_DIR}/include/
  DESTINATION
    ${CAMP_INSTALL_INCLUDE_DIR}
)

install(
  DIRECTORY
    ${CAMP_MOD_DIR}/../include/
  DESTINATION
    ${CAMP_INSTALL_INCLUDE_DIR}/camp/fortran
  FILES_MATCHING PATTERN "*.mod"
)

set(cmake_config_install_location "${CMMAKE_INSTALL_LIBDIR}/cmake/camp")

install(
  EXPORT
    camp_Exports
  NAMESPACE
    camp::
  DESTINATION
    ${cmake_config_install_location}
)

configure_package_config_file(
  "${CAMP_PROJECT_SRC_DIR}/cmake/campConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/campConfig.cmake"
  INSTALL_DESTINATION
    ${cmake_config_install_location}
)

write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/campConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(
  FILES
    "${PROJECT_BINARY_DIR}/campConfig.cmake"
    "${PROJECT_BINARY_DIR}/campConfigVersion.cmake"
  DESTINATION
    ${cmake_config_install_location}
)

################
# uninstall target

if(NOT TARGET uninstall)
  configure_file(
    "${CAMP_PROJECT_SRC_DIR}/cmake/cmake_uninstall.cmake.in"
    "${PROJECT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY
  )
  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CAMP_PROJECT_BINARY_DIR}/cmake_uninstall.cmake
    COMMENT "Uninstalling camp"
  )
endif()