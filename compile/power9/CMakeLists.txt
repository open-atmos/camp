cmake_minimum_required(VERSION 2.6.4)
project(CAMP Fortran C)

message("HI")
find_package(CUDA)
enable_language(CUDA)

SET(OBJS
        ${CMAKE_CURRENT_SOURCE_DIR}/../../build/libcamp.a
        )

add_executable(mock_monarch ../../test/monarch/mock_monarch.F90
        ../../test/monarch/camp_monarch_interface.F90
        ${CB5_EBI_SOLVER} ${CB5_KPP_SOLVER}
        ../../test/chemistry/cb05cl_ae5/module_BSC_CHEM_DATA.F90)
target_link_libraries(mock_monarch camplib)


SET_SOURCE_FILES_PROPERTIES(
        ${OBJS}
        PROPERTIES
        EXTERNAL_OBJECT true
        GENERATED true
)