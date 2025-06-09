set(CB5_EBI_SOLVER
	cb05cl_ae5/EXT-CONST_mod.F90
	cb05cl_ae5/ext-hrg2.F90
	cb05cl_ae5/ext-hrcalcks.F90
	cb05cl_ae5/ext-hrinit.F90
	cb05cl_ae5/ext-hrdata_mod.F90
	cb05cl_ae5/ext-hrg3.F90
	cb05cl_ae5/ext-hrprodloss.F90
	cb05cl_ae5/ext-hrg1.F90
	cb05cl_ae5/ext-hrg4.F90
	cb05cl_ae5/ext-hrrates.F90
	cb05cl_ae5/ext-hrsolver.F90
	cb05cl_ae5/EXT-RXCM_mod.F90
	)

set(CB5_KPP_SOLVER
	cb05cl_ae5/cb05cl_ae5_Function.f90
	cb05cl_ae5/cb05cl_ae5_Global.f90
	cb05cl_ae5/cb05cl_ae5_Initialize.f90
        cb05cl_ae5/cb05cl_ae5_Integrator.F90
	cb05cl_ae5/cb05cl_ae5_Jacobian.f90
	cb05cl_ae5/cb05cl_ae5_JacobianSP.f90
	cb05cl_ae5/cb05cl_ae5_LinearAlgebra.f90
	cb05cl_ae5/cb05cl_ae5_Model.f90
	cb05cl_ae5/cb05cl_ae5_Monitor.f90
	cb05cl_ae5/cb05cl_ae5_Parameters.f90
	cb05cl_ae5/cb05cl_ae5_Precision.f90
	cb05cl_ae5/cb05cl_ae5_Rates.f90
	cb05cl_ae5/cb05cl_ae5_Util.f90
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
        cb05cl_ae5/test_cb05cl_ae5_big.F90
        ${CB5_EBI_SOLVER} ${CB5_KPP_SOLVER}
        cb05cl_ae5/module_BSC_CHEM_DATA.F90)
target_link_libraries(test_chemistry_cb05cl_ae5_big camp::camp)

######################################################################
# test_chemistry_cb05cl_ae5

add_executable(test_chemistry_cb05cl_ae5
    cb05cl_ae5/test_cb05cl_ae5.F90
    ${CB5_EBI_SOLVER} ${CB5_KPP_SOLVER}
    cb05cl_ae5/module_BSC_CHEM_DATA.F90)
target_link_libraries(test_chemistry_cb05cl_ae5 camp::camp)

######################################################################