! Copyright (C) 2017 Matt Dawson
! Licensed under the GNU General Public License version 2 or (at your
! option) any later version. See the file COPYING for details.

!> \file
!> The mock_monarch program

!> Mock version of the MONARCH model for testing integration with PartMC
program mock_monarch

  use pmc_constants,                    only: const
  use pmc_util,                          only : assert_msg, almost_equal, &
                                                to_string
  use camp_monarch_interface
  use pmc_mpi
#ifdef PMC_USE_JSON
  use json_module
#endif

#ifdef SOLVE_EBI_IMPORT_CAMP_INPUT

  ! EBI Solver
  use module_bsc_chem_data
  use EXT_HRDATA
  use EXT_RXCM,                               only : NRXNS, RXLABEL

  ! KPP Solver
  use cb05cl_ae5_Initialize,                  only : KPP_Initialize => Initialize
  use cb05cl_ae5_Model,                       only : KPP_NSPEC => NSPEC, &
          KPP_STEPMIN => STEPMIN, &
          KPP_STEPMAX => STEPMAX, &
          KPP_RTOL => RTOL, &
          KPP_ATOL => ATOL, &
          KPP_TIME => TIME, &
          KPP_C => C, &
          KPP_RCONST => RCONST, &
          KPP_Update_RCONST => Update_RCONST, &
          KPP_INTEGRATE => INTEGRATE, &
          KPP_SPC_NAMES => SPC_NAMES, &
          KPP_PHOTO_RATES => PHOTO_RATES, &
          KPP_TEMP => TEMP, &
          KPP_PRESS => PRESS, &
          KPP_SUN => SUN, &
          KPP_M => M, &
          KPP_N2 => N2, &
          KPP_O2 => O2, &
          KPP_H2 => H2, &
          KPP_H2O => H2O, &
          KPP_N2O => N2O, &
          KPP_CH4 => CH4, &
          KPP_NVAR => NVAR, &
          KPP_NREACT => NREACT, &
          KPP_DT => DT
  use cb05cl_ae5_Parameters,                  only : KPP_IND_O2 => IND_O2
  use cb05cl_ae5_Initialize, ONLY: Initialize

#endif

  implicit none

  !> File unit for model run-time messages
  integer, parameter :: OUTPUT_FILE_UNIT = 6
  !> File unit for model results
  integer, parameter :: RESULTS_FILE_UNIT = 7
  !> File unit for script generation
  integer, parameter :: SCRIPTS_FILE_UNIT = 8
  !> File unit for results comparison
  integer, parameter :: COMPARE_FILE_UNIT = 9
  integer, parameter :: RESULTS_FILE_UNIT_TABLE = 10
  integer, parameter :: RESULTS_FILE_UNIT_PY = 11
  integer, parameter :: IMPORT_FILE_UNIT = 12

  integer(kind=i_kind), parameter :: NUM_CAMP_SPEC = 79
  integer(kind=i_kind), parameter :: NUM_EBI_SPEC = 72
  integer(kind=i_kind), parameter :: NUM_EBI_PHOTO_RXN = 23

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Parameters for mock MONARCH model !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Number of total species in mock MONARCH
  integer, parameter :: NUM_MONARCH_SPEC = 250 !800
  !> Number of vertical cells in mock MONARCH
  integer :: NUM_VERT_CELLS
  !> Starting W-E cell for camp-chem call
  integer :: I_W
  !> Ending W-E cell for camp-chem call
  integer :: I_E
  !> Starting S-N cell for camp-chem call
  integer :: I_S
  !> Ending S-N cell for camp-chem call
  integer :: I_N
  !> Number of W-E cells in mock MONARCH
  integer :: NUM_WE_CELLS
  !> Number of S-N cells in mock MONARCH
  integer :: NUM_SN_CELLS
  !> Starting index for camp-chem species in tracer array
  integer, parameter :: START_CAMP_ID = 1!100
  !> Ending index for camp-chem species in tracer array
  integer, parameter :: END_CAMP_ID = 210!350
  !> Time step (min)
  real, parameter :: TIME_STEP = 2.!1.6
  !> Number of time steps to integrate over
  integer, parameter :: NUM_TIME_STEP = 720!1!720!180
  !> Index for water vapor in water_conc()
  integer, parameter :: WATER_VAPOR_ID = 5
  !> Start time
  real, parameter :: START_TIME = 0.0
  !> Number of cells to compute simultaneously
  integer :: n_cells
  !> Check multiple cells results are correct?
  logical :: check_multiple_cells = .false.
  character(len=:), allocatable :: ADD_EMISIONS

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! State variables for mock MONARCH model !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> NMMB style arrays (W->E, S->N, top->bottom, ...)
  !> Temperature (K)
  real,allocatable :: temperature(:, :, :)
  !> Species conc (various units)
  real, allocatable  :: species_conc(:, :, :, :)
  !> Water concentrations (kg_H2O/kg_air)
  real, allocatable  :: water_conc(:, :, :, :)
  !> Air density (kg_air/m^3)
  real, allocatable  :: air_density(:, :, :)
  !> Air pressure (Pa)
  real, allocatable  :: pressure(:, :, :)
  !> Cell height (m)
  real :: height

  !> Emissions parameters
  !> Emission conversion parameter (mol s-1 m-2 to ppmv)
  real :: conv
  !> Emissions hour counter
  integer :: i_hour = 0

  !> Comparison values
  real :: comp_species_conc(0:NUM_TIME_STEP, NUM_MONARCH_SPEC)
  real, allocatable :: species_conc_copy(:, :, :, :)

  !> Starting time for mock model run (min since midnight)
  !! is tracked in MONARCH
  real :: curr_time = START_TIME

  !> Set starting time for gnuplot scripts (includes initial conditions as first
  !! data point)
  real :: plot_start_time = START_TIME

  !> !!! Add to MONARCH variables !!!
  type(monarch_interface_t), pointer :: pmc_interface

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Mock model setup and evaluation variables !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> CAMP-chem input file file
  character(len=:), allocatable :: camp_input_file
  !> PartMC-camp <-> MONARCH interface configuration file
  character(len=:), allocatable :: interface_input_file
  !> Results file prefix
  character(len=:), allocatable :: output_file_prefix, output_file_title, str_to_int_aux
  !> CAMP-chem input file file
  type(string_t), allocatable :: name_gas_species_to_print(:), name_aerosol_species_to_print(:)
  integer(kind=i_kind), allocatable :: id_gas_species_to_print(:), id_aerosol_species_to_print(:)
  integer(kind=i_kind) :: size_gas_species_to_print, size_aerosol_species_to_print

  ! MPI
#ifdef PMC_USE_MPI
  character, allocatable :: buffer(:)
  integer(kind=i_kind) :: pos, pack_size
#endif

  character(len=500) :: arg
  integer :: status_code, i_time, i_spec, i_case, i, j, k, z,n_cells_plot,cell_to_print
  !> Partmc nº of cases to test
  integer :: pmc_cases = 1
  integer :: plot_case, new_v_cells

  ! initialize mpi (to take the place of a similar MONARCH call)
  call pmc_mpi_init()

  if(check_multiple_cells) then
    pmc_cases=2
  end if

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! **** Add to MONARCH during initialization **** !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  ! Initialize PartMC-camp
  call get_command_argument(1, arg, status=status_code)
  call assert_msg(678165802, status_code.eq.0, "Error getting PartMC-camp "//&
          "configuration file name")
  camp_input_file = trim(arg)
  call get_command_argument(2, arg, status=status_code)
  call assert_msg(664104564, status_code.eq.0, "Error getting PartMC-camp "//&
          "<-> MONARCH interface configuration file name")
  interface_input_file = trim(arg)

  ! Initialize the mock model
  call get_command_argument(3, arg, status=status_code)
  call assert_msg(234156729, status_code.eq.0, "Error getting output file prefix")
  output_file_title = "Mock_"//trim(arg)
  output_file_prefix = "out/"//trim(arg)

  call get_command_argument(4, arg, status=status_code)
  if(status_code.eq.0) then
    ADD_EMISIONS = trim(arg)
  else
    !One-cell case as default
    print*, "WARNING: not ADD_EMISIONS parameter received, value set to ON"
    ADD_EMISIONS = "ON"
  end if

  call get_command_argument(5, arg, status=status_code)
  if(status_code.eq.0) then
    str_to_int_aux = trim(arg)
    read(str_to_int_aux, *) NUM_VERT_CELLS
  else
    !One-cell case as default
    print*, "WARNING: not n_cells parameter received, value set to 1"
    NUM_VERT_CELLS = 1
  end if

  call get_command_argument(6, arg, status=status_code)
  if(status_code.eq.0) then
    if(arg.eq."Multi-cells") then
      n_cells = (I_E - I_W+1)*(I_N - I_S+1)*NUM_VERT_CELLS
    else
      n_cells = 1
    end if
  else
    !One-cell case as default
    print*, "WARNING: not Multi-cells flag parameter received, value set to One-cell"
    n_cells = 1
  end if

  I_W=1
  I_E=1
  I_S=1
  I_N=1
  NUM_WE_CELLS = I_E-I_W+1
  NUM_SN_CELLS = I_N-I_S+1

  !One-cell:
  !n_cells = 1
  !Multi:
  !n_cells = (I_E - I_W+1)*(I_N - I_S+1)*NUM_VERT_CELLS

  allocate(temperature(NUM_WE_CELLS,NUM_SN_CELLS,NUM_VERT_CELLS))
  allocate(species_conc(NUM_WE_CELLS,NUM_SN_CELLS,NUM_VERT_CELLS,NUM_MONARCH_SPEC))
  allocate(water_conc(NUM_WE_CELLS,NUM_SN_CELLS,NUM_VERT_CELLS,WATER_VAPOR_ID))
  allocate(air_density(NUM_WE_CELLS,NUM_SN_CELLS,NUM_VERT_CELLS))
  allocate(pressure(NUM_WE_CELLS,NUM_SN_CELLS,NUM_VERT_CELLS))
  allocate(species_conc_copy(NUM_WE_CELLS,NUM_SN_CELLS,NUM_VERT_CELLS,NUM_MONARCH_SPEC))

  n_cells_plot = 1
  cell_to_print = 1
  !cell_to_print = n_cells

  if(interface_input_file.eq."interface_simple.json") then

    if (pmc_mpi_rank().eq.0) then
      write(*,*) "Config simple (test 1)"
    end if

    size_gas_species_to_print=3
    size_aerosol_species_to_print=0

    allocate(name_gas_species_to_print(size_gas_species_to_print))
    allocate(name_aerosol_species_to_print(size_aerosol_species_to_print))
    allocate(id_gas_species_to_print(size_gas_species_to_print))
    allocate(id_aerosol_species_to_print(size_aerosol_species_to_print))

    !name_gas_species_to_print(1)%string=("A")
    !name_gas_species_to_print(2)%string=("C")
    name_gas_species_to_print(1)%string=("A")
    name_gas_species_to_print(2)%string=("B")
    name_gas_species_to_print(3)%string=("C")

  else

    if (pmc_mpi_rank().eq.0) then
      write(*,*) "Config complex (test 2)"
    end if

    plot_case=4
    if(plot_case == 0)then
      size_gas_species_to_print=4
      !size_aerosol_species_to_print=1
      size_aerosol_species_to_print=0
    elseif(plot_case == 1)then
      size_gas_species_to_print=1
      size_aerosol_species_to_print=1
    elseif(plot_case == 2 .or. plot_case == 3)then
      size_gas_species_to_print=3
      size_aerosol_species_to_print=2
    elseif(plot_case == 4)then
      size_gas_species_to_print=5
      size_aerosol_species_to_print=24
    endif

    allocate(name_gas_species_to_print(size_gas_species_to_print))
    allocate(name_aerosol_species_to_print(size_aerosol_species_to_print))
    allocate(id_gas_species_to_print(size_gas_species_to_print))
    allocate(id_aerosol_species_to_print(size_aerosol_species_to_print))

    if(plot_case == 0)then
      name_gas_species_to_print(1)%string=("O3")
      name_gas_species_to_print(2)%string=("NO2")
      name_gas_species_to_print(3)%string=("NO")
      name_gas_species_to_print(4)%string=("ISOP")
      !name_aerosol_species_to_print(1)%string=("organic_matter.1.organic_matter.POA")
    elseif(plot_case == 1)then
      name_gas_species_to_print(1)%string=("OH")
      name_aerosol_species_to_print(1)%string=("organic_matter.1.organic_matter.POA")
    elseif(plot_case == 2)then
      name_gas_species_to_print(1)%string=("ISOP")
      name_gas_species_to_print(2)%string=("ISOP-P1")
      name_gas_species_to_print(3)%string=("ISOP-P2")
      name_aerosol_species_to_print(1)%string=("organic_matter.1.organic_matter.ISOP-P1_aero")
      name_aerosol_species_to_print(2)%string=("organic_matter.2.organic_matter.ISOP-P1_aero")
    elseif(plot_case == 3)then
      name_gas_species_to_print(1)%string=("TERP")
      name_gas_species_to_print(2)%string=("TERP-P1")
      name_gas_species_to_print(3)%string=("TERP-P2")
      name_aerosol_species_to_print(1)%string=("organic_matter.1.organic_matter.TERP-P1_aero")
      name_aerosol_species_to_print(2)%string=("organic_matter.1.organic_matter.TERP-P2_aero")
    elseif(plot_case == 4)then
      name_gas_species_to_print(1)%string=("O3")
      name_gas_species_to_print(2)%string=("NO2")
      name_gas_species_to_print(3)%string=("ISOP")
      name_gas_species_to_print(4)%string=("ISOP-P1")
      name_gas_species_to_print(5)%string=("ISOP-P2")
      name_aerosol_species_to_print(1)%string=("organic_matter.1.organic_matter.POA")
      name_aerosol_species_to_print(2)%string=("organic_matter.2.organic_matter.POA")
      name_aerosol_species_to_print(3)%string=("organic_matter.3.organic_matter.POA")
      name_aerosol_species_to_print(4)%string=("organic_matter.4.organic_matter.POA")
      name_aerosol_species_to_print(5)%string=("organic_matter.5.organic_matter.POA")
      name_aerosol_species_to_print(6)%string=("organic_matter.6.organic_matter.POA")
      name_aerosol_species_to_print(7)%string=("organic_matter.7.organic_matter.POA")
      name_aerosol_species_to_print(8)%string=("organic_matter.8.organic_matter.POA")

      name_aerosol_species_to_print(9)%string=("organic_matter.1.organic_matter.ISOP-P1_aero")
      name_aerosol_species_to_print(10)%string=("organic_matter.2.organic_matter.ISOP-P1_aero")
      name_aerosol_species_to_print(11)%string=("organic_matter.3.organic_matter.ISOP-P1_aero")
      name_aerosol_species_to_print(12)%string=("organic_matter.4.organic_matter.ISOP-P1_aero")
      name_aerosol_species_to_print(13)%string=("organic_matter.5.organic_matter.ISOP-P1_aero")
      name_aerosol_species_to_print(14)%string=("organic_matter.6.organic_matter.ISOP-P1_aero")
      name_aerosol_species_to_print(15)%string=("organic_matter.7.organic_matter.ISOP-P1_aero")
      name_aerosol_species_to_print(16)%string=("organic_matter.8.organic_matter.ISOP-P1_aero")

      name_aerosol_species_to_print(17)%string=("organic_matter.1.organic_matter.ISOP-P2_aero")
      name_aerosol_species_to_print(18)%string=("organic_matter.2.organic_matter.ISOP-P2_aero")
      name_aerosol_species_to_print(19)%string=("organic_matter.3.organic_matter.ISOP-P2_aero")
      name_aerosol_species_to_print(20)%string=("organic_matter.4.organic_matter.ISOP-P2_aero")
      name_aerosol_species_to_print(21)%string=("organic_matter.5.organic_matter.ISOP-P2_aero")
      name_aerosol_species_to_print(22)%string=("organic_matter.6.organic_matter.ISOP-P2_aero")
      name_aerosol_species_to_print(23)%string=("organic_matter.7.organic_matter.ISOP-P2_aero")
      name_aerosol_species_to_print(24)%string=("organic_matter.8.organic_matter.ISOP-P2_aero")

    endif
  end if

  if (pmc_mpi_rank().eq.0) then
    write(*,*) "Num time-steps:", NUM_TIME_STEP, "Num cells:",&
            NUM_WE_CELLS*NUM_SN_CELLS*NUM_VERT_CELLS, trim(arg)
  end if

  !Check if repeat program to compare one cell and multicell
  if(check_multiple_cells) then
    pmc_cases=2
  end if

  call model_initialize(output_file_prefix)

  !Repeat in case we want a checking
  do i_case=1, pmc_cases

    pmc_interface => monarch_interface_t(camp_input_file, interface_input_file, &
            START_CAMP_ID, END_CAMP_ID, n_cells, ADD_EMISIONS)!, n_cells

    if (pmc_mpi_rank().eq.0) then
      do j=1, size(name_gas_species_to_print)
        do z=1, size(pmc_interface%monarch_species_names)
          if(pmc_interface%monarch_species_names(z)%string.eq.name_gas_species_to_print(j)%string) then
            id_gas_species_to_print(j)=pmc_interface%map_monarch_id(z)
          end if
        end do
      end do

      do j=1, size(name_aerosol_species_to_print)
        do z=1, size(pmc_interface%monarch_species_names)
          if(pmc_interface%monarch_species_names(z)%string.eq.name_aerosol_species_to_print(j)%string) then
            id_aerosol_species_to_print(j)=pmc_interface%map_monarch_id(z)
          end if
        end do
      end do
    end if

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! **** end initialization modification **** !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ! Set conc from mock_model
    call pmc_interface%get_init_conc(species_conc, water_conc, WATER_VAPOR_ID, &
            air_density,i_W,I_E,I_S,I_N)

#ifndef IMPORT_CAMP_INPUT
    call import_camp_input(pmc_interface)
    !call import_camp_input_json(pmc_interface)
#endif

#ifdef SOLVE_EBI_IMPORT_CAMP_INPUT
    if (pmc_mpi_rank().eq.0) then
      !Not working in other ranks than 0 (memory allocation error)
      call solve_ebi(pmc_interface)
    end if
#endif

    ! Run the model
    do i_time=1, NUM_TIME_STEP

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! **** Add to MONARCH during runtime for each time step **** !
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      if (pmc_mpi_rank().eq.0) then
        !todo fix print_state_gnuplot to use advantages from output_results
        !call print_state_gnuplot(curr_time,pmc_interface, name_gas_species_to_print,id_gas_species_to_print&
        !       ,name_aerosol_species_to_print,id_aerosol_species_to_print,RESULTS_FILE_UNIT)
        call output_results(curr_time,pmc_interface,species_conc)
      end if



      call pmc_interface%integrate(curr_time,         & ! Starting time (min)
                                   TIME_STEP,         & ! Time step (min)
                                   I_W,               & ! Starting W->E grid cell
                                   I_E,               & ! Ending W->E grid cell
                                   I_S,               & ! Starting S->N grid cell
                                   I_N,               & ! Ending S->N grid cell
                                   temperature,       & ! Temperature (K)
                                   species_conc,      & ! Tracer array
                                   water_conc,        & ! Water concentrations (kg_H2O/kg_air)
                                   WATER_VAPOR_ID,    & ! Index in water_conc() corresponding to water vapor
                                   air_density,       & ! Air density (kg_air/m^3)
                                   pressure,          & ! Air pressure (Pa)
                                   conv,              &
                                   i_hour)
      curr_time = curr_time + TIME_STEP



      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! **** end runtime modification **** !
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    end do

#ifdef PMC_USE_MPI
    if (pmc_mpi_rank().eq.0) then
      write(*,*) "Model run time: ", comp_time, " s"
    end if
#else
    write(*,*) "Model run time: ", comp_time, " s"
#endif

    !Save results
    if(i_case.eq.1) then
      species_conc_copy(:,:,:,:) = species_conc(:,:,:,:)
    end if

    ! Set 1 cell to get a checksum case
    n_cells = 1

  end do

  if (pmc_mpi_rank().eq.0) then
  !if (pmc_mpi_rank().ne.999) then
    do i = I_W, I_E
      do j = I_S, I_N
        do k = 1, NUM_VERT_CELLS
          do z=1, size(name_gas_species_to_print)
            !print*,id_gas_species_to_print(z),name_gas_species_to_print(z)%string,&
            !species_conc(i,j,k,id_gas_species_to_print(z))
          end do
        end do
      end do
    end do
  end if
  !print*,"hola"
  !print*,"Rank",pmc_mpi_rank(), "conc",&
  !        species_conc(1,1,1,pmc_interface%map_monarch_id(:))

  !If something to compare
  if(pmc_cases.gt.1) then
    !Compare results
    do i = I_W, I_E
      do j = I_S, I_N
        do k = 1, NUM_VERT_CELLS
          do i_spec = START_CAMP_ID, END_CAMP_ID
            call assert_msg( 394742768, &
              almost_equal( real( species_conc(i,j,k,i_spec), kind=dp ), &
                  real( species_conc_copy(i,j,k,i_spec), kind=dp ), &
                  1.d-5, 1d-4 ), &
              "Concentration species mismatch for species "// &
                  trim( to_string( i_spec ) )//". Expected: "// &
                  trim( to_string( species_conc(i,j,k,i_spec) ) )//", got: "// &
                  trim( to_string( species_conc_copy(i,j,k,i_spec) ) ) )
          end do
        end do
      end do
    end do
  end if

#ifdef SOLVE_EBI_IMPORT_CAMP_INPUT
    if (pmc_mpi_rank().eq.0) then
      !Not working in other ranks than 0 (memory allocation error)
      call compare_ebi_camp_json(pmc_interface)
    end if
#endif

  ! Output results and scripts
  if (pmc_mpi_rank().eq.0) then
    !write(*,*) "MONARCH interface tests - PASS"
    !call print_state_gnuplot(curr_time,pmc_interface,species_conc)
    !call print_state_gnuplot(curr_time,pmc_interface, name_gas_species_to_print,id_gas_species_to_print&
    !        ,name_aerosol_species_to_print,id_aerosol_species_to_print,RESULTS_FILE_UNIT)
    call output_results(curr_time,pmc_interface,species_conc)
    call create_gnuplot_script(pmc_interface, output_file_prefix, &
            plot_start_time, curr_time)
    !call create_gnuplot_persist(pmc_interface, output_file_prefix, &
    !        output_file_title, plot_start_time, curr_time, n_cells_plot, cell_to_print)
    call create_gnuplot_persist(pmc_interface, output_file_prefix, &
            plot_start_time, curr_time)
  end if

  close(RESULTS_FILE_UNIT)
  close(RESULTS_FILE_UNIT_TABLE)
  close(RESULTS_FILE_UNIT_PY)

  ! Deallocation
  deallocate(camp_input_file)
  deallocate(interface_input_file)
  deallocate(output_file_prefix)
  deallocate(output_file_title)

  !print*,"MPI_FINALIZE RANK",pmc_mpi_rank()

!#ifdef PMC_USE_MPI

  deallocate(pmc_interface)

  call pmc_mpi_finalize()

!#endif

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Initialize the mock model
  subroutine model_initialize(file_prefix)

    !> File prefix for model results
    character(len=:), allocatable, intent(in) :: file_prefix

    character(len=:), allocatable :: file_name
    character(len=:), allocatable :: aux_str, aux_str_py
    character(len=128) :: i_str !if len !=128 crashes (e.g len=100)
    integer :: z,i,j,k,r,i_cell,i_spec
    integer :: n_cells_print

    ! Open the output file
    file_name = file_prefix//"_results.txt"
    open(RESULTS_FILE_UNIT, file=file_name, status="replace", action="write")
    file_name = file_prefix//"_results_table.txt"
    open(RESULTS_FILE_UNIT_TABLE, file=file_name, status="replace", action="write")
    file_name = file_prefix//"_urban_plume_0001.txt"
    open(RESULTS_FILE_UNIT_PY, file=file_name, status="replace", action="write")

    n_cells_print=(I_E - I_W+1)*(I_N - I_S+1)*NUM_VERT_CELLS

    !print Titles
    aux_str = "Time"
    aux_str_py = "Time Cell"

    do i_cell=1,n_cells_print
      write(i_str,*) i_cell
      i_str=adjustl(i_str)
      do i_spec=1, size(name_gas_species_to_print)
        aux_str = aux_str//" "//name_gas_species_to_print(i_spec)%string//"_"//trim(i_str)
      end do
    end do

    aux_str = aux_str//" "//"Time"

    do i_cell=1,n_cells_print
      write(i_str,*) i_cell
      i_str=adjustl(i_str)
      do i_spec=1, size(name_aerosol_species_to_print)
        aux_str = aux_str//" "//name_aerosol_species_to_print(i_spec)%string//"_"//trim(i_str)
      end do
    end do

    do i_spec=1, size(name_gas_species_to_print)
      aux_str_py = aux_str_py//" "//name_gas_species_to_print(i_spec)%string
    end do

    do i_spec=1, size(name_aerosol_species_to_print)
      aux_str_py = aux_str_py//" "//name_aerosol_species_to_print(i_spec)%string//"_"//trim(i_str)
    end do

    write(RESULTS_FILE_UNIT, "(A)", advance="no") aux_str
    write(RESULTS_FILE_UNIT, *) ""

    write(RESULTS_FILE_UNIT_PY, "(A)", advance="no") aux_str_py
    write(RESULTS_FILE_UNIT_PY, '(a)') ''

    ! TODO refine initial model conditions
    species_conc(:,:,:,:) = 0.0
    height=1. !(m)

    if(ADD_EMISIONS.eq."ON") then

      water_conc(:,:,:,WATER_VAPOR_ID) = 0. !0.01165447 !this is equal to 95% RH !1e-14 !0.01 !kg_h2o/kg-1_air
      temperature(:,:,:) = 290.016!300.614166259766
      pressure(:,:,:) = 100000.!94165.7187500000
  !    air_density(:,:,:) = pressure(:,:,:)/(287.04*temperature(:,:,:))!1.225
      air_density(:,:,:) = pressure(:,:,:)/(287.04*temperature(:,:,:)* &
           (1.+0.60813824*water_conc(:,:,:,WATER_VAPOR_ID))) !kg m-3
      conv=0.02897/air_density(1,1,1)*(TIME_STEP*60.)*1e6/height !units of time_step to seconds
  !    conv=0.02897/air_density(1,1,1)*(TIME_STEP)*1e6/height !units of time_step to seconds
    else

      temperature(:,:,:) = 300.614166259766
      pressure(:,:,:) = 94165.7187500000
      air_density(:,:,:) = 1.225
      conv=0.02897/air_density(1,1,1)*(TIME_STEP*60.)*1e6/height !units of time_step to seconds
      water_conc(:,:,:,WATER_VAPOR_ID) = 0.03!0.01

      !Initialize different axis values
      !Species_conc is modified in monarch_interface%get_init_conc
      !do i=I_W, I_E
      !  temperature(i,:,:) = temperature(i,:,:) !+ 0.1*i
      !  pressure(i,:,:) = pressure(i,:,:) !- 1*i
      !end do

      !do j=I_S, I_N
      !  temperature(:,j,:) = temperature(:,j,:)! + 0.3*j
      !  pressure(:,:,j) = pressure(:,:,j) !- 3*j
      !end do

      !do k=1, NUM_VERT_CELLS
      !  temperature(:,:,k) = temperature(:,:,k)! + 0.6*k
      !  pressure(:,k,:) = pressure(:,k,:) !- 6*k
      !end do

    end if

    deallocate(file_name)
    deallocate(aux_str)
    deallocate(aux_str_py)

    ! Read the compare file
    ! TODO Implement once results are stable

  end subroutine model_initialize

  !> Read the comparison file (must have same dimensions as current config)
  subroutine read_comp_file()

    integer :: i_time
    real :: time, water

    do i_time = 0, NUM_TIME_STEP + 1
      read(COMPARE_FILE_UNIT, *) time, &
             comp_species_conc(i_time, START_CAMP_ID:END_CAMP_ID), &
             water
    end do

  end subroutine read_comp_file

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine import_camp_input(pmc_interface)

    type(monarch_interface_t), intent(inout) :: pmc_interface
    integer :: z,i,j,k,r,o,i_cell,i_spec,i_photo_rxn
    integer :: state_size_per_cell

#ifdef USE_MAPPING_EBI
    type(string_t), dimension(NUM_EBI_SPEC) :: ebi_spec_names
    type(string_t), dimension(NUM_EBI_SPEC) :: monarch_spec_names
    type(string_t), allocatable :: camp_spec_names(:)
    integer, dimension(NUM_CAMP_SPEC) :: map_camp_monarch
    integer, dimension(NUM_CAMP_SPEC) :: map_monarch_ebi
    !integer, dimension(NUM_CAMP_SPEC) :: map_ebi_monarch
    real(kind=dp), dimension(NUM_CAMP_SPEC) :: aux_state_var
    real(kind=dp), dimension(NUM_CAMP_SPEC) :: aux_state_var2
#endif

    state_size_per_cell = pmc_interface%camp_core%state_size_per_cell()

    !open(IMPORT_FILE_UNIT, file="exports/camp_input.txt", status="old")!default test monarch input
    open(IMPORT_FILE_UNIT, file="exports/camp_input_18.txt", status="old") !monarch
    !open(IMPORT_FILE_UNIT, file="exports/camp_input_322.txt", status="old") !monarch

    if (pmc_mpi_rank().eq.0) then
      write(*,*) "Importing camp input"
    end if

    !if(n_cells.gt.1) then
    !  print*, "ERROR: Import can only handle data from 1 cell, set n_cells to 1"
    !end if

    read(IMPORT_FILE_UNIT,*) (pmc_interface%camp_state%state_var(&
            i),i=1,size(pmc_interface%camp_state%state_var)/n_cells)

    !write(*,*) "Importing temperatures and pressures"

    read(IMPORT_FILE_UNIT,*) ( ( (temperature(i,j,k), k=1,1 ), j=1,1),&
            i=1,1 )
    read(IMPORT_FILE_UNIT,*) ( ( (pressure(i,j,k), k=1,1 ), j=1,1),&
            i=1,1 )

    !write(*,*) "Importing photolysis rates"

    read(IMPORT_FILE_UNIT,*) (pmc_interface%base_rates(&
            i),i=1,pmc_interface%n_photo_rxn)

    do i=I_W,I_E
      do j=I_S,I_N
        do k=1,NUM_VERT_CELLS

          o = (j-1)*(I_E) + (i-1) !Index to 3D
          z = (k-1)*(I_E*I_N) + o !Index for 2D

          do r=1,size(pmc_interface%camp_state%state_var)/n_cells
            pmc_interface%camp_state%state_var(r+z*state_size_per_cell) = &
            pmc_interface%camp_state%state_var(r)
          end do

          pmc_interface%camp_state%state_var(pmc_interface%map_camp_id(:)+(z*state_size_per_cell))=&
                  pmc_interface%camp_state%state_var(pmc_interface%map_camp_id(:))

          species_conc(i,j,k,pmc_interface%map_monarch_id(:)) = &
                  pmc_interface%camp_state%state_var(pmc_interface%map_camp_id(:))

          temperature(i,j,k) = temperature(1,1,1)!+z*0.1
          pressure(i,j,k) = pressure(1,1,1)

          do i_photo_rxn = 1, pmc_interface%n_photo_rxn

          !if (pmc_mpi_rank().eq.1) then
            !pmc_interface%base_rates(i_photo_rxn) = pmc_interface%base_rates(i_photo_rxn)!+0.01
            !pmc_interface%base_rates(i_photo_rxn) = 0.01
            !write(*,*), "rates",i_photo_rxn, pmc_interface%base_rates(i_photo_rxn)
          !end if
          !write(*,*), "rates",i_photo_rxn, pmc_interface%base_rates(i_photo_rxn)


          call pmc_interface%photo_rxns(i_photo_rxn)%set_rate(real(pmc_interface%base_rates(i_photo_rxn), kind=dp))
          !call pmc_interface%photo_rxns(i_photo_rxn)%set_rate(real(0.0, kind=dp)) !works

          call pmc_interface%camp_core%update_data(pmc_interface%photo_rxns(i_photo_rxn),z)

        !print*,"id photo_rate", pmc_interface%base_rates(i_photo_rxn)
          end do
        end do
      end do
    end do


    close(IMPORT_FILE_UNIT)

  end subroutine import_camp_input

  subroutine import_camp_input_json(pmc_interface)

    type(monarch_interface_t), intent(inout) :: pmc_interface
    integer :: z,i,j,k,r,o,i_cell,i_spec,i_photo_rxn
    integer :: state_size_per_cell

    type(json_file) :: jfile
    type(json_core) :: json
    character(len=:), allocatable :: export_path, spec_name_json
    character(len=128) :: mpi_rank_str, i_str
    integer :: mpi_rank, id
    real(kind=dp) :: dt, temp, press, real_val
    type(string_t), allocatable :: camp_spec_names(:)
    real, dimension(NUM_EBI_PHOTO_RXN) :: ebi_photo_rates
    integer, dimension(NUM_EBI_PHOTO_RXN) :: photo_id_camp

    state_size_per_cell = pmc_interface%camp_core%state_size_per_cell()


    !mpi_rank = 18
    mpi_rank = 0
    write(mpi_rank_str,*) mpi_rank
    mpi_rank_str=adjustl(mpi_rank_str)

    export_path = "/gpfs/scratch/bsc32/bsc32815/a2s8/nmmb-monarch/MODEL/"&
            //"SRC_LIBS/partmc/test/monarch/exports/camp_in_out_"&
            //trim(mpi_rank_str)//".json"

    call jfile%initialize()

    call jfile%load_file(export_path); if (jfile%failed()) print*,&
            "JSON not found at compare_ebi_camp_json"

    if (pmc_mpi_rank().eq.0) then
      write(*,*) "Importing camp input json"
    end if

    if(n_cells.gt.1) then
      print*, "ERROR: Import can only handle data from 1 cell, set n_cells to 1"
    end if

    camp_spec_names=pmc_interface%camp_core%unique_names()

    do i=1, size(camp_spec_names)
      call jfile%get('input.species.'//camp_spec_names(i)%string,&
              pmc_interface%camp_state%state_var(i))
      !print*, camp_spec_names(i)%string, pmc_interface%camp_state%state_var(i)
    end do

    do i=I_W,I_E
      do j=I_S,I_N
        do k=1,NUM_VERT_CELLS
          o = (j-1)*(I_E) + (i-1) !Index to 3D
          z = (k-1)*(I_E*I_N) + o !Index for 2D

          species_conc(i,j,k,pmc_interface%map_monarch_id(:)) = &
                  pmc_interface%camp_state%state_var(pmc_interface%map_camp_id(:)+(z*state_size_per_cell))

          call jfile%get('input.temperature',temp)
          temperature(i,j,k)=temp
          call jfile%get('input.pressure',press)
          pressure(i,j,k)=press
          !print*,"PRESSURE READ CAMP",pressure(i,j,k)
        end do
      end do
    end do

    do i=1, pmc_interface%n_photo_rxn
      write(i_str,*) i
      i_str=adjustl(i_str)
      call jfile%get('input.photo_rates.'//trim(i_str),&
              pmc_interface%base_rates(i))
      !print*, trim(i_str), pmc_interface%base_rates(i)
    end do

    call jfile%destroy()

    do i_photo_rxn = 1, pmc_interface%n_photo_rxn

      call pmc_interface%photo_rxns(i_photo_rxn)%set_rate(real(pmc_interface%base_rates(i_photo_rxn), kind=dp))
      !call pmc_interface%photo_rxns(i_photo_rxn)%set_rate(real(0.0, kind=dp)) !works

      call pmc_interface%camp_core%update_data(pmc_interface%photo_rxns(i_photo_rxn))

      !print*,"id photo_rate", pmc_interface%base_rates(i_photo_rxn)
    end do

    close(IMPORT_FILE_UNIT)

  end subroutine import_camp_input_json

#ifdef SOLVE_EBI_IMPORT_CAMP_INPUT

  subroutine solve_ebi(pmc_interface)

    type(monarch_interface_t), intent(inout) :: pmc_interface
    integer :: z,i,j,k,r,o,i_cell,i_spec,i_photo_rxn,i_time

    real(kind=dp) :: dt, temp, press_json, auxr
    real :: press
    type(json_core) :: json
    type(json_value),pointer :: p, species_in, species_out, input, output&
            , photo_rates
    character(len=:), allocatable :: export_path, spec_name
    integer :: mpi_rank
    character(len=128) :: mpi_rank_str, i_str

    type(string_t), dimension(NUM_EBI_SPEC) :: ebi_spec_names
    type(string_t), dimension(NUM_EBI_SPEC) :: monarch_spec_names
    type(string_t), allocatable :: camp_spec_names(:)
    integer, dimension(NUM_EBI_SPEC) :: ebi_spec_id_to_camp
    logical :: is_sunny
    real, dimension(NUM_EBI_PHOTO_RXN) :: ebi_photo_rates
    integer, dimension(NUM_EBI_PHOTO_RXN) :: photo_id_camp
    real(kind=dp) :: rel_error_in_out
    real(kind=dp), allocatable :: ebi_init(:)

    ! Set the BSC chem parameters
    call init_bsc_chem_data()
    ! Set the output unit
    LOGDEV = 6
    ! Set the aerosol flag
    L_AE_VRSN = .false.
    ! Set the aq. chem flag
    L_AQ_VRSN = .false.
    ! Initialize the solver
    call EXT_HRINIT
    RKI(:) = 0.0
    RXRAT(:) = 0.0
    YC(:) = 0.0
    YC0(:) = 0.0
    YCP(:) = 0.0
    PROD(:) = 0.0
    LOSS(:) = 0.0
    PNEG(:) = 0.0
    EBI_TMSTEP = TIME_STEP
    N_EBI_STEPS = 1
    N_INR_STEPS = 1

    is_sunny = .true.

    allocate(ebi_init(size(YC)))

    call set_ebi_species(ebi_spec_names)
    call set_monarch_species(monarch_spec_names)
    camp_spec_names=pmc_interface%camp_core%unique_names()

    call assert_msg(122432506, size(pmc_interface%camp_state%state_var).eq.NUM_CAMP_SPEC, &
            "NUM_CAMP_SPEC not equal size(state_var)")

    call set_ebi_photo_ids_with_camp(photo_id_camp)
    do i=1, NUM_EBI_PHOTO_RXN
      ebi_photo_rates(i)=&
              pmc_interface%base_rates(photo_id_camp(i))*60
      !print*,i,ebi_photo_rates(i)
    end do

    ebi_spec_id_to_camp(:) = 1
    do i = 1, NUM_EBI_SPEC
      do j = 1, NUM_CAMP_SPEC
        if (trim(ebi_spec_names(i)%string).eq.trim(camp_spec_names(j)%string)) then


          ebi_spec_id_to_camp(j) = i
          YC(i) = pmc_interface%camp_state%state_var(j)
          ebi_init(i) = YC(i)

          !print*,ebi_spec_names(i)%string, pmc_interface%camp_state%state_var(j)
        end if
      end do
    end do

    do i = 1, NUM_CAMP_SPEC
      if (trim(camp_spec_names(i)%string).eq."H2O") then
        water_conc(1,1,1,WATER_VAPOR_ID) = pmc_interface%camp_state%state_var(i)
        !print*,"EBI H2O",water_conc(1,1,1,WATER_VAPOR_ID)
      end if
    end do

    press=pressure(1,1,1)/const%air_std_press
    !print*,"EBI press, pressure(1,1,1), const%air_std_press"&
    !,press,pressure(1,1,1),const%air_std_press

    do i_time=1, NUM_TIME_STEP

      call EXT_HRCALCKS( NUM_EBI_PHOTO_RXN,       & ! Number of EBI solver photolysis reactions
              is_sunny,                & ! Flag for sunlight
              ebi_photo_rates,             & ! Photolysis rates
              temperature(1,1,1),             & ! Temperature (K)
              press,                & ! Air pressure (atm)
              water_conc(1,1,1,WATER_VAPOR_ID),              & ! Water vapor concentration (ppmV)
              RKI)                       ! Rate constants
      call EXT_HRSOLVER( 2018012, 070000, 1, 1, 1) ! These dummy variables are just for output

    end do

    if (pmc_mpi_rank().eq.0) then

      call json%initialize()
      call json%create_object(p,'')
      call json%create_object(input,'input')
      !call json%add(p, "id", 1) !test
      call json%add(p, input)

      z=1
      r=1
      o=1
      i_cell=(o-1)*(I_E*I_N) + (r-1)*(I_E) + z
      call json%add(input, "cell", i_cell)
      dt=TIME_STEP*60
      call json%add(input, "dt",dt)
      temp=temperature(z,r,o)
      call json%add(input, "temperature", temp)
      press_json=pressure(z,r,o)/const%air_std_press
      call json%add(input, "pressure", press_json)

      call json%create_object(species_in,'species')
      call json%add(input, species_in)
      call json%create_object(output,'output')
      call json%add(p, output)
      call json%create_object(species_out,'species')
      call json%add(output, species_out)

      do j = 1, NUM_EBI_SPEC
        !i = ebi_spec_id_to_camp(j) !CAMP order
        i = j !EBI order
        auxr=ebi_init(i)
        call json%add(species_in, ebi_spec_names(i)%string, auxr)
        auxr=YC(i)
        call json%add(species_out, ebi_spec_names(i)%string, auxr)
        end do

      call json%create_object(photo_rates,'photo_rates')
      call json%add(input, photo_rates)
      do i=1, size(ebi_photo_rates)
        write(i_str,*) i
        i_str=adjustl(i_str)
        auxr=ebi_photo_rates(i)
        call json%add(photo_rates, trim(i_str), auxr)
      end do

#ifdef PMC_USE_MPI
      mpi_rank = pmc_mpi_rank()

      write(mpi_rank_str,*) mpi_rank
      mpi_rank_str=adjustl(mpi_rank_str)

      export_path = "/gpfs/scratch/bsc32/bsc32815/a2s8/nmmb-monarch/MODEL/"&
              //"SRC_LIBS/partmc/test/monarch/exports/ebi_in_out_"&
              //trim(mpi_rank_str)//".json"
#else
      export_path = "/gpfs/scratch/bsc32/bsc32815/a2s8/nmmb-monarch/MODEL/"&
              //"SRC_LIBS/partmc/test/monarch/exports/ebi_in_out.json"
#endif
      call json%print(p,export_path)

      !cleanup:
      call json%destroy(p)
      if (json%failed()) stop 1

    end if

#ifndef PRINT_EBI_INPUT
    print*,"EBI species"
    print*, "TIME_STEP", TIME_STEP
    print*, "Temp", temperature(1,1,1)
    print*, "Press", pressure(1,1,1)
    print*,"water_conc",water_conc(1,1,1,WATER_VAPOR_ID)
    print*,"ebi_photo_rates:", ebi_photo_rates(:)
#endif

#ifdef PRINT_EBI_SPEC_BY_CAMP_ORDER
    print*,"CAMP order:"
    print*,"Name, init, out, rel. error [(init-out)/(init+out)]"
    do j = 1, NUM_EBI_SPEC
      i = ebi_spec_id_to_camp(j)
      rel_error_in_out=abs((ebi_init(i)-YC(i))/&
              (ebi_init(i)+YC(i)+1.0d-30))
      print*,ebi_spec_names(i)%string, ebi_init(i)&
      ,YC(i), rel_error_in_out
    end do
#endif

#ifdef PRINT_EBI_SPEC_BY_EBI_ORDER
    print*,"EBI order:"
    print*,"Name, id, init, out, rel. error [(init-out)/(init+out)]"
    do j = 1, NUM_EBI_SPEC
      i = j
      rel_error_in_out=abs((ebi_init(i)-YC(i))/&
              (ebi_init(i)+YC(i)+1.0d-30))
      print*,ebi_spec_names(i)%string, i, ebi_init(i)&
              ,YC(i)!, rel_error_in_out
    end do
#endif

  end subroutine solve_ebi

  subroutine solve_kpp(pmc_interface)

    type(monarch_interface_t), intent(inout) :: pmc_interface
    integer :: z,i,j,k,r,o,i_cell,i_spec,i_photo_rxn,i_time

    ! Set the step limits
    KPP_STEPMIN = 0.0d0
    KPP_STEPMAX = 0.0d0
    KPP_SUN = 1.0
    ! Set the tolerances
    do i_spec = 1, KPP_NVAR
      KPP_RTOL(i_spec) = 1.0d-4
      KPP_ATOL(i_spec) = 1.0d-3
    end do
    CALL KPP_Initialize()

    KPP_PHOTO_RATES(:) = pmc_interface%base_rates(:)

  end subroutine solve_kpp

  subroutine compare_ebi_camp_json(pmc_interface)

    type(monarch_interface_t), intent(inout) :: pmc_interface
    integer :: z,i,j,k,r,o,i_cell,i_spec,i_photo_rxn,i_time

    type(json_file) :: jfile
    type(json_core) :: json
    type(json_value), pointer :: j_obj, child, next
    integer :: n_childs
    character(kind=json_ck, len=:), allocatable :: key, unicode_str_val
    integer(kind=json_ik) :: var_type

    character(len=:), allocatable :: export_path, spec_name_json
    character(len=128) :: mpi_rank_str, i_str
    integer :: mpi_rank, id

    real(kind=dp) :: dt, temp, press, real_val
    real(kind=dp), dimension(NUM_EBI_SPEC) :: ebi_spec_out, ebi_spec_in
    real(kind=dp), dimension(NUM_CAMP_SPEC) :: camp_spec_out
    type(string_t), dimension(NUM_EBI_SPEC) :: ebi_spec_names
    type(string_t), dimension(NUM_EBI_SPEC) :: monarch_spec_names
    type(string_t), allocatable :: camp_spec_names(:)
    real, dimension(NUM_EBI_PHOTO_RXN) :: ebi_photo_rates
    integer, dimension(NUM_EBI_PHOTO_RXN) :: photo_id_camp

    real(kind=dp) :: rel_error_in_out
    real(kind=dp) :: MAX_REL_ERROR_TOL = 0.8

    call set_ebi_species(ebi_spec_names)
    call set_monarch_species(monarch_spec_names)
    camp_spec_names=pmc_interface%camp_core%unique_names()

    !mpi_rank = 18
    mpi_rank = 0
    write(mpi_rank_str,*) mpi_rank
    mpi_rank_str=adjustl(mpi_rank_str)

    export_path = "/gpfs/scratch/bsc32/bsc32815/a2s8/nmmb-monarch/MODEL/"&
            //"SRC_LIBS/partmc/test/monarch/exports/ebi_in_out_"&
            //trim(mpi_rank_str)//".json"

    call jfile%initialize()

    call jfile%load_file(export_path); if (jfile%failed()) print*,&
            "JSON not found at compare_ebi_camp_json"

    do i=1, size(ebi_spec_out)
      call jfile%get('output.species.'//ebi_spec_names(i)%string,&
              ebi_spec_out(i))
      call jfile%get('input.species.'//ebi_spec_names(i)%string,&
              ebi_spec_in(i))
      !print*, ebi_spec_names(i)%string, ebi_spec_out(i)
    end do

    do i=1, size(ebi_spec_out)
      call jfile%get('output.species.'//ebi_spec_names(i)%string,&
              ebi_spec_out(i))
      !print*, ebi_spec_names(i)%string, ebi_spec_out(i)
    end do

    call jfile%destroy()

    export_path = "/gpfs/scratch/bsc32/bsc32815/a2s8/nmmb-monarch/MODEL/"&
            //"SRC_LIBS/partmc/test/monarch/exports/camp_in_out_"&
            //trim(mpi_rank_str)//".json"

    call jfile%initialize()

    call jfile%load_file(export_path); if (json%failed()) print*,&
            "JSON not found at compare_ebi_camp_json"

    do i=1, size(camp_spec_names)
      call jfile%get('output.species.'//camp_spec_names(i)%string,&
              camp_spec_out(i))
      !print*, camp_spec_names(i)%string, camp_spec_out(i)
    end do

    call jfile%destroy()

    print*, "Specs relative error[(ebi-camp)/(ebi+camp)]&
            greater than MAX_REL_ERROR_TOL",MAX_REL_ERROR_TOL
    print*, "Name, input, ebi_out, camp_out, camp_id"! &
            !,rel. error [(ebi-camp)/(ebi+camp)]"

    do i=1, size(ebi_spec_names)
      do j=1, size(camp_spec_names)
        if (ebi_spec_names(i)%string.eq.camp_spec_names(j)%string) then
          rel_error_in_out=abs((ebi_spec_out(i)-camp_spec_out(j))/&
                (ebi_spec_out(i)+camp_spec_out(j)+1.0d-30))
          if(rel_error_in_out.gt.MAX_REL_ERROR_TOL) then
            print*, ebi_spec_names(i)%string, ebi_spec_in(i), ebi_spec_out(i)&
                    ,camp_spec_out(j), j!,rel_error_in_out
          end if
        end if
      end do
    end do

#ifdef DEBUG_INPUT_OUTPUT
    print*, "Specs with error greater than MAX_REL_ERROR_TOL",MAX_REL_ERROR_TOL
    print*, "Name, init_state_var, out_state_var, &
            ,rel. error [(init-out)/(init+out)]"
#endif

#ifdef DEBUG_INPUT_OUTPUT
      rel_error_in_out=abs((this%init_state_var(i)-camp_state%state_var(i))/&
            (this%init_state_var(i)+camp_state%state_var(i)+1.0d-30))
    if(rel_error_in_out.gt.MAX_REL_ERROR_TOL) then
      print*, this%spec_names(i)%string, this%init_state_var(i)&
              ,camp_state%state_var(i),rel_error_in_out
    end if
    print*, "All specs"
    print*, "Name, init_state_var, out_state_var, &
            ,rel. error [(init-out)/(init+out)]"
    do i=1, size(this%spec_names)
      rel_error_in_out=abs((this%init_state_var(i)-camp_state%state_var(i))/&
              (this%init_state_var(i)+camp_state%state_var(i)+1.0d-30))
      print*, this%spec_names(i)%string, this%init_state_var(i)&
              ,camp_state%state_var(i),rel_error_in_out
    end do
#endif

  end subroutine compare_ebi_camp_json

#endif

  !> Output the model results
  !subroutine print_state_gnuplot(curr_time,pmc_interface,species_conc)
  subroutine print_state_gnuplot(curr_time_in, pmc_interface, name_gas_species_to_print,id_gas_species_to_print&
          ,name_aerosol_species_to_print,id_aerosol_species_to_print, file_unit, n_cells_to_print)

    !> Current model time (min since midnight)
    real, intent(in) :: curr_time_in
    type(monarch_interface_t), intent(in) :: pmc_interface
    type(string_t), allocatable, intent(inout) :: name_gas_species_to_print(:), name_aerosol_species_to_print(:)
    integer(kind=i_kind), allocatable, intent(inout) :: id_gas_species_to_print(:), id_aerosol_species_to_print(:)
    integer, intent(inout), optional :: n_cells_to_print
    integer, intent(in) :: file_unit

    integer :: z,i,j,k,r,i_cell
    character(len=:), allocatable :: aux_str, aux_str_py
    character(len=128) :: i_cell_str, time_str
    integer :: n_cells
    real :: curr_time

    !curr_time_min=curr_time_in/60.0
    curr_time=curr_time_in

    write(RESULTS_FILE_UNIT_TABLE, *) "Time_step:", curr_time

    do i=I_W,I_E
      do j=I_S,I_N
        do k=1,NUM_VERT_CELLS
          write(RESULTS_FILE_UNIT_TABLE, *) "i:",i,"j:",j,"k:",k
          write(RESULTS_FILE_UNIT_TABLE, *) "Spec_name, Concentrations, Map_monarch_id"
          do z=1, size(pmc_interface%monarch_species_names)
            write(RESULTS_FILE_UNIT_TABLE, *) pmc_interface%monarch_species_names(z)%string&
            , species_conc(i,j,k,pmc_interface%map_monarch_id(z))&
            , pmc_interface%map_monarch_id(z)
            !write(*,*) "species_conc out",species_conc(i,j,k,pmc_interface%map_monarch_id(z))
          end do
        end do
      end do
    end do

    write(file_unit, "(F12.4)", advance="no") curr_time

    do i=I_W,I_E
      do j=I_S,I_N
        do k=1,NUM_VERT_CELLS

          write(time_str,*) curr_time
          time_str=adjustl(time_str)
          write(RESULTS_FILE_UNIT_PY, "(A)", advance="no") trim(time_str)

          !write(RESULTS_FILE_UNIT_PY, "(F12.4)", advance="no") curr_time
          write(RESULTS_FILE_UNIT_PY, "(A)", advance="no") " "

          i_cell = (k-1)*(I_E*I_N) + (j-1)*(I_E) + i
          write(i_cell_str,*) i_cell
          i_cell_str=adjustl(i_cell_str)
          write(RESULTS_FILE_UNIT_PY, "(A)", advance="no") trim(i_cell_str)

          do z=1, size(name_gas_species_to_print)
            write(file_unit, "(ES13.6)", advance="no") &
                    species_conc(i,j,k,id_gas_species_to_print(z))
            write(RESULTS_FILE_UNIT_PY, "(ES13.6)", advance="no") &
                    species_conc(i,j,k,id_gas_species_to_print(z))
            !print*,id_gas_species_to_print(z),name_gas_species_to_print(z)%string,species_conc(i,j,k,id_gas_species_to_print(z))
          end do

          do z=1, size(name_aerosol_species_to_print)
            write(RESULTS_FILE_UNIT_PY, "(ES13.6)", advance="no") &
                    species_conc(i,j,k,id_aerosol_species_to_print(z))
          end do

          write(RESULTS_FILE_UNIT_PY, '(a)') ''
        end do
      end do
    end do

    write(file_unit, "(F12.4)", advance="no") curr_time

    do i=I_W,I_E
      do j=I_S,I_N
        do k=1,NUM_VERT_CELLS
          do z=1, size(name_aerosol_species_to_print)
            write(file_unit, "(ES13.6)", advance="no") &
                    species_conc(i,j,k,id_aerosol_species_to_print(z))
          end do
        end do
      end do
    end do

    write(file_unit, *) ""

    !todo include water_conc with species_conc to easy access
    !write(RESULTS_FILE_UNIT, *) curr_time, &
    !        species_conc(1,1,1,START_CAMP_ID:END_CAMP_ID), &
    !        water_conc(1,1,1,WATER_VAPOR_ID)


  end subroutine print_state_gnuplot

  !> Output the model results
  subroutine output_results(curr_time,pmc_interface,species_conc)

    !> Current model time (min since midnight)
    real, intent(in) :: curr_time
    type(monarch_interface_t), intent(in) :: pmc_interface
    integer :: z,i,j,k
    real, intent(inout) :: species_conc(:,:,:,:)
    !type(string_t), allocatable :: species_names(:)
    !integer(kind=i_kind), allocatable :: tracer_ids(:)
    !call pmc_interface%get_MONARCH_species(species_names, tracer_ids)

    character(len=:), allocatable :: aux_str
    real, allocatable :: aux_real
    logical, save :: first_time=.true.

    write(RESULTS_FILE_UNIT_TABLE, *) "Time_step:", curr_time

    do i=I_W,I_E
      do j=I_W,I_E
        do k=I_W,I_E
          write(RESULTS_FILE_UNIT_TABLE, *) "i:",i,"j:",j,"k:",k
          write(RESULTS_FILE_UNIT_TABLE, *) "Spec_name, Concentrations, Map_monarch_id"
          do z=1, size(pmc_interface%monarch_species_names)
            write(RESULTS_FILE_UNIT_TABLE, *) pmc_interface%monarch_species_names(z)%string&
            , species_conc(i,j,k,pmc_interface%map_monarch_id(z))&
            , pmc_interface%map_monarch_id(z)
            !write(*,*) "species_conc out",species_conc(i,j,k,pmc_interface%map_monarch_id(z))
          end do
        end do
      end do
    end do

    !write(RESULTS_FILE_UNIT, *) curr_time,species_conc(1,1,1,START_CAMP_ID:END_CAMP_ID)
    !write(RESULTS_FILE_UNIT, *) curr_time,species_conc(1,1,1,START_CAMP_ID:3)

    !Specific names
    !do z=1, size(pmc_interface%monarch_species_names)
    !  if(pmc_interface%monarch_species_names(z)%string.eq.name_specie_to_print) then
    !    aux_str = pmc_interface%monarch_species_names(z)%string//" "//pmc_interface%monarch_species_names(z+1)%string
    !    write(RESULTS_FILE_UNIT, *) "Time ",aux_str
    !    write(RESULTS_FILE_UNIT, *) curr_time,species_conc(1,1,1,pmc_interface%map_monarch_id(z):pmc_interface%map_monarch_id(z)+1)
        !write(RESULTS_FILE_UNIT, *) curr_time,species_conc(1,1,1,pmc_interface%map_monarch_id(z))
    !  end if
    !end do

    !print Titles
    if(first_time)then
    aux_str = "Time(min)"
    do z=1, size(name_gas_species_to_print)
      aux_str = aux_str//" "//name_gas_species_to_print(z)%string
      !aux_str = aux_str//' "'//name_gas_species_to_print(z)%string//'"'
    end do
!    aux_str = aux_str//" "//"Time"

    do z=1, size(name_aerosol_species_to_print)
      !if (name_aerosol_species_to_print(z)%string)

      !end if
      aux_str = aux_str//" "//name_aerosol_species_to_print(z)%string
      !aux_str = aux_str//' "'//name_aerosol_species_to_print(z)%string//'"'
    end do

    write(RESULTS_FILE_UNIT, "(A)", advance="no") aux_str
    write(RESULTS_FILE_UNIT, "(A)", advance="yes") " "
    first_time=.false.
    endif

    !write(RESULTS_FILE_UNIT, "(F12.4)", advance="no") curr_time
    write(RESULTS_FILE_UNIT, "(i4.4)", advance="no") int(curr_time)
    do z=1, size(name_gas_species_to_print)
      write(RESULTS_FILE_UNIT, "(ES13.6)", advance="no") &
              species_conc(1,1,1,id_gas_species_to_print(z))
    end do
 !   write(RESULTS_FILE_UNIT, "(F12.4)", advance="no") curr_time
    do z=1, size(name_aerosol_species_to_print)
      write(RESULTS_FILE_UNIT, "(ES13.6)", advance="no") &
      species_conc(1,1,1,id_aerosol_species_to_print(z))
    end do

    write(RESULTS_FILE_UNIT, "(A)", advance="yes") " "

    !todo include water_conc with species_conc
    !write(RESULTS_FILE_UNIT, *) curr_time, &
    !        species_conc(1,1,1,START_CAMP_ID:END_CAMP_ID), &
    !        water_conc(1,1,1,WATER_VAPOR_ID)

  end subroutine output_results

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Create a gnuplot script for viewing species concentrations
  subroutine create_gnuplot_script(pmc_interface, file_path, start_time, &
            end_time)

    !> PartMC-camp <-> MONARCH interface
    type(monarch_interface_t), intent(in) :: pmc_interface
    !> File prefix for gnuplot script
    character(len=:), allocatable :: file_path
    !> Plot start time
    real :: start_time
    !> Plot end time
    real :: end_time

    type(string_t), allocatable :: species_names(:)
    integer(kind=i_kind), allocatable :: tracer_ids(:)
    character(len=:), allocatable :: file_name, spec_name
    integer(kind=i_kind) :: i_char, i_spec, tracer_id

    ! Get the species names and ids
    call pmc_interface%get_MONARCH_species(species_names, tracer_ids)

    ! Adjust the tracer ids to match the results file
    tracer_ids(:) = tracer_ids(:) - START_CAMP_ID + 2

    ! Create the gnuplot script
    file_name = file_path//".conf"
    open(unit=SCRIPTS_FILE_UNIT, file=file_name, status="replace", action="write")
    write(SCRIPTS_FILE_UNIT,*) "# "//file_name
    write(SCRIPTS_FILE_UNIT,*) "# Run as: gnuplot "//file_name
    write(SCRIPTS_FILE_UNIT,*) "set terminal png truecolor"
    write(SCRIPTS_FILE_UNIT,*) "set autoscale"
    write(SCRIPTS_FILE_UNIT,*) "set xrange [", start_time, ":", end_time, "]"
    do i_spec = 1, size(species_names)
      spec_name = species_names(i_spec)%string
      forall (i_char = 1:len(spec_name), spec_name(i_char:i_char).eq.'/') &
                spec_name(i_char:i_char) = '_'
      write(SCRIPTS_FILE_UNIT,*) "set output '"//file_path//"_"// &
              spec_name//".png'"
      write(SCRIPTS_FILE_UNIT,*) "plot\"
      write(SCRIPTS_FILE_UNIT,*) " '"//file_path//"_results.txt'\"
      write(SCRIPTS_FILE_UNIT,*) " using 1:"// &
              trim(to_string(tracer_ids(i_spec)))//" title '"// &
              species_names(i_spec)%string//" (MONARCH)'"
    end do
    tracer_id = END_CAMP_ID - START_CAMP_ID + 3
    write(SCRIPTS_FILE_UNIT,*) "set output '"//file_path//"_H2O.png'"
    write(SCRIPTS_FILE_UNIT,*) "plot\"
    write(SCRIPTS_FILE_UNIT,*) " '"//file_path//"_results.txt'\"
    write(SCRIPTS_FILE_UNIT,*) " using 1:"// &
            trim(to_string(tracer_id))//" title 'H2O (MONARCH)'"
    close(SCRIPTS_FILE_UNIT)

    deallocate(species_names)
    deallocate(tracer_ids)
    deallocate(file_name)
    deallocate(spec_name)

  end subroutine create_gnuplot_script

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Create a gnuplot script for viewing species concentrations
  subroutine create_gnuplot_persist_gpu(pmc_interface, file_path, plot_title, &
          start_time, end_time, n_cells_plot, i_cell)

    !> PartMC-camp <-> MONARCH interface
    type(monarch_interface_t), intent(in) :: pmc_interface
    !> File prefix for gnuplot script
    character(len=:), allocatable :: plot_title, file_path
    !> Plot start time
    real :: start_time
    !> Plot end time
    real :: end_time
    integer, intent(in) :: n_cells_plot, i_cell

    type(string_t), allocatable :: species_names(:)
    integer(kind=i_kind), allocatable :: tracer_ids(:)
    character(len=:), allocatable :: gnuplot_path, spec_name
    integer(kind=i_kind) :: i_char, i_spec, tracer_id
    integer(kind=i_kind) :: gas_species_start_plot,n_gas_species_plot, n_aerosol_species_plot, aerosol_species_start_plot&
    ,aerosol_species_time_plot
    character(len=100) :: n_gas_species_plot_str
    character(len=100) :: gas_species_start_plot_str
    character(len=100) :: n_aerosol_species_plot_str
    character(len=100) :: aerosol_species_start_plot_str
    character(len=100) :: aerosol_species_time_plot_str
    integer :: n_cells

    n_cells=(I_E-I_W+1)*(I_N-I_S+1)*NUM_VERT_CELLS

    call assert_msg(207035921, n_cells_plot.le.n_cells, &
            "More cells to plot than cells available")
    call assert_msg(207035921, i_cell.le.n_cells, &
            "Cell to plot more than cells available")

    ! Get the species names and ids
    call pmc_interface%get_MONARCH_species(species_names, tracer_ids)

    ! Adjust the tracer ids to match the results file
    tracer_ids(:) = tracer_ids(:) - START_CAMP_ID + 2

    if(pmc_interface%n_cells.eq.1) then
      plot_title=plot_title//" - One_cell"
    else
      plot_title=plot_title//" - Multi_cells"
    end if

    !if n_cells_plot<n_cells
    gas_species_start_plot=size(name_gas_species_to_print)*(i_cell-1)+2
    write(gas_species_start_plot_str,*) gas_species_start_plot
    gas_species_start_plot_str=adjustl(gas_species_start_plot_str)

    n_gas_species_plot = size(name_gas_species_to_print)*n_cells_plot+gas_species_start_plot-1
    write(n_gas_species_plot_str,*) n_gas_species_plot
    n_gas_species_plot_str=adjustl(n_gas_species_plot_str)

    aerosol_species_time_plot=size(name_gas_species_to_print)*n_cells+2
    write(aerosol_species_time_plot_str,*) aerosol_species_time_plot
    aerosol_species_time_plot_str=adjustl(aerosol_species_time_plot_str)

    aerosol_species_start_plot=aerosol_species_time_plot+size(name_aerosol_species_to_print)*(i_cell-1)+1
    write(aerosol_species_start_plot_str,*) aerosol_species_start_plot
    aerosol_species_start_plot_str=adjustl(aerosol_species_start_plot_str)

    n_aerosol_species_plot = size(name_aerosol_species_to_print)*n_cells_plot+aerosol_species_start_plot-1
    write(n_aerosol_species_plot_str,*) n_aerosol_species_plot
    n_aerosol_species_plot_str=adjustl(n_aerosol_species_plot_str)

    ! Create the gnuplot script
    gnuplot_path = file_path//".gnuplot"
    open(unit=SCRIPTS_FILE_UNIT, file=gnuplot_path, status="replace", action="write")
    write(SCRIPTS_FILE_UNIT,*) "# Run as: gnuplot -persist "//gnuplot_path
    !write(SCRIPTS_FILE_UNIT,*) "set key top left"
    write(SCRIPTS_FILE_UNIT,*) "set title '"//plot_title//"'"
    write(SCRIPTS_FILE_UNIT,*) "set xlabel 'Time (min)'"
    write(SCRIPTS_FILE_UNIT,*) "set ylabel 'Gas concentration [ppmv]'"
    write(SCRIPTS_FILE_UNIT,*) "set y2label 'Aerosol concentration [kg/m^3]'"
    write(SCRIPTS_FILE_UNIT,*) "set ytics nomirror"
    write(SCRIPTS_FILE_UNIT,*) "set y2tics nomirror"

    !write(SCRIPTS_FILE_UNIT,*) "set autoscale"
    write(SCRIPTS_FILE_UNIT,*) "set logscale y"
    write(SCRIPTS_FILE_UNIT,*) "set logscale y2"
    write(SCRIPTS_FILE_UNIT,*) "set xrange [", start_time, ":", end_time, "]"
    !write(SCRIPTS_FILE_UNIT,*) "set xrange [", start_time/60.0, ":", end_time/60.0, "]"

    i_spec=1
    spec_name = species_names(i_spec)%string
    forall (i_char = 1:len(spec_name), spec_name(i_char:i_char).eq.'/') &
          spec_name(i_char:i_char) = '_'

    !write(SCRIPTS_FILE_UNIT,*) "set key outside"
    write(SCRIPTS_FILE_UNIT,*) "set key top left"

    if(size(name_aerosol_species_to_print).gt.0) then

      write(SCRIPTS_FILE_UNIT,"(A)",advance="no") "plot for [col="&
      //trim(gas_species_start_plot_str)//":" &
      //trim(n_gas_species_plot_str)//"] &
      '"//file_path//"_results.txt' &
      using 1:col axis x1y1 title columnheader, for [col2=" &
      //trim(aerosol_species_start_plot_str)//":" &
      //trim(n_aerosol_species_plot_str)//"] &
      '"//file_path//"_results.txt' &
      using " &
      //trim(aerosol_species_time_plot_str)// &
      ":col2 axis x1y2 title columnheader"

    else

      write(SCRIPTS_FILE_UNIT,"(A)",advance="no") "plot for [col="&
      //trim(gas_species_start_plot_str)//":" &
      //trim(n_gas_species_plot_str)//"] &
      '"//file_path//"_results.txt' &
      using 1:col axis x1y1 title columnheader"

    end if

    !tracer_id = END_CAMP_ID - START_CAMP_ID + 3
    !write(SCRIPTS_FILE_UNIT,*) "set output '"//file_name//"_H2O.png'"
    !write(SCRIPTS_FILE_UNIT,*) "plot\"
    !write(SCRIPTS_FILE_UNIT,*) " '"//file_name//"_results.txt'\"
    !write(SCRIPTS_FILE_UNIT,*) " using 1:"// &
    !        trim(to_string(tracer_id))//" title 'H2O (MONARCH)'"


    close(SCRIPTS_FILE_UNIT)

    deallocate(species_names)
    deallocate(tracer_ids)
    deallocate(spec_name)

  end subroutine

  !> Create a gnuplot script for viewing species concentrations
  subroutine create_gnuplot_persist(pmc_interface, file_prefix, start_time, &
          end_time)

    !> PartMC-camp <-> MONARCH interface
    type(monarch_interface_t), intent(in) :: pmc_interface
    !> File prefix for gnuplot script
    character(len=:), allocatable :: file_prefix
    !> Plot start time
    real :: start_time
    !> Plot end time
    real :: end_time

    type(string_t), allocatable :: species_names(:)
    integer(kind=i_kind), allocatable :: tracer_ids(:)
    character(len=:), allocatable :: file_name, spec_name
    integer(kind=i_kind) :: i_char, i_spec, tracer_id
    integer(kind=i_kind) :: n_gas_species_plot, n_aerosol_species_plot, n_aerosol_species_start_plot&
    ,n_aerosol_species_time_plot
    character(len=100) :: n_gas_species_plot_str
    character(len=100) :: n_aerosol_species_plot_str
    character(len=100) :: n_aerosol_species_start_plot_str
    character(len=100) :: n_aerosol_species_time_plot_str


    ! Get the species names and ids
    call pmc_interface%get_MONARCH_species(species_names, tracer_ids)

    ! Adjust the tracer ids to match the results file
    tracer_ids(:) = tracer_ids(:) - START_CAMP_ID + 2

    n_gas_species_plot = size(name_gas_species_to_print)
    n_gas_species_plot = n_gas_species_plot+1
    write(n_gas_species_plot_str,*) n_gas_species_plot
    n_gas_species_plot_str=adjustl(n_gas_species_plot_str)

    n_aerosol_species_plot = size(name_aerosol_species_to_print)
    n_aerosol_species_plot = n_aerosol_species_plot+n_gas_species_plot+1
    write(n_aerosol_species_plot_str,*) n_aerosol_species_plot
    n_aerosol_species_plot_str=adjustl(n_aerosol_species_plot_str)

    n_aerosol_species_start_plot=n_gas_species_plot+2
    write(n_aerosol_species_start_plot_str,*) n_aerosol_species_start_plot
    n_aerosol_species_start_plot_str=adjustl(n_aerosol_species_start_plot_str)

    n_aerosol_species_time_plot=n_gas_species_plot+1
    write(n_aerosol_species_time_plot_str,*) n_aerosol_species_time_plot
    n_aerosol_species_time_plot_str=adjustl(n_aerosol_species_time_plot_str)

    ! Create the gnuplot script
    !file_name = "partmc/build/test_run/monarch/"//file_prefix//".gnuplot"
    file_name = file_prefix//".gnuplot"
    open(unit=SCRIPTS_FILE_UNIT, file=file_name, status="replace", action="write")
    write(SCRIPTS_FILE_UNIT,*) "# "//file_name
    write(SCRIPTS_FILE_UNIT,*) "# Run as: gnuplot -persist "//file_name
    write(SCRIPTS_FILE_UNIT,*) "set terminal jpeg medium size 640,480 truecolor"
    !write(SCRIPTS_FILE_UNIT,*) "set key top left"
    write(SCRIPTS_FILE_UNIT,*) "set title 'Mock_monarch_cb05_soa'"
    write(SCRIPTS_FILE_UNIT,*) "set xlabel 'Time (min)'"
    write(SCRIPTS_FILE_UNIT,*) "set ylabel 'Gas concentration (ppmv)'"
    write(SCRIPTS_FILE_UNIT,*) "set y2label 'Aerosol concentration (kg/m^3)'"
    write(SCRIPTS_FILE_UNIT,*) "set ytics nomirror"
    write(SCRIPTS_FILE_UNIT,*) "set y2tics nomirror"

    !write(SCRIPTS_FILE_UNIT,*) "set autoscale"
    write(SCRIPTS_FILE_UNIT,*) "set logscale y"
    write(SCRIPTS_FILE_UNIT,*) "set logscale y2"
    write(SCRIPTS_FILE_UNIT,*) "set xrange [", start_time, ":", end_time, "]"

    i_spec=1
    spec_name = species_names(i_spec)%string
    forall (i_char = 1:len(spec_name), spec_name(i_char:i_char).eq.'/') &
          spec_name(i_char:i_char) = '_'

    !todo improve path detecting
    !write(SCRIPTS_FILE_UNIT,*) "set key outside"
    write(SCRIPTS_FILE_UNIT,*) "set key top left"
    !write(SCRIPTS_FILE_UNIT,*) "set output '"//file_prefix//"_plot.png'"
    write(SCRIPTS_FILE_UNIT,"(A)",advance="no") "set output '&
            /gpfs/scratch/bsc32/bsc32815/paperpartmc/partmc/build/&
        test_run/monarch/out/monarch_plot.jpg'"
    write(SCRIPTS_FILE_UNIT,*)
    write(SCRIPTS_FILE_UNIT,"(A)",advance="no") "plot for [col=2:"&
    //trim(n_gas_species_plot_str)//"] &
    'partmc/build/test_run/monarch/"//file_prefix//"_results.txt' &
    using 1:col axis x1y1 title columnheader, for [col2=" &
    //trim(n_aerosol_species_start_plot_str)//":" &
    //trim(n_aerosol_species_plot_str)//"] &
    'partmc/build/test_run/monarch/"//file_prefix//"_results.txt' &
    using " &
    //trim(n_aerosol_species_time_plot_str)// &
    ":col2 axis x1y2 title columnheader"

    !tracer_id = END_CAMP_ID - START_CAMP_ID + 3
    !write(SCRIPTS_FILE_UNIT,*) "set output '"//file_prefix//"_H2O.png'"
    !write(SCRIPTS_FILE_UNIT,*) "plot\"
    !write(SCRIPTS_FILE_UNIT,*) " '"//file_prefix//"_results.txt'\"
    !write(SCRIPTS_FILE_UNIT,*) " using 1:"// &
    !        trim(to_string(tracer_id))//" title 'H2O (MONARCH)'"

    close(SCRIPTS_FILE_UNIT)

    deallocate(species_names)
    deallocate(tracer_ids)
    deallocate(file_name)
    deallocate(spec_name)

  end subroutine

  subroutine set_ebi_species(spec_names)

    !> EBI solver species names
    type(string_t), dimension(NUM_EBI_SPEC) :: spec_names

    spec_names(1)%string = "NO2"
    spec_names(2)%string = "NO"
    spec_names(3)%string = "O"
    spec_names(4)%string = "O3"
    spec_names(5)%string = "NO3"
    spec_names(6)%string = "O1D"
    spec_names(7)%string = "OH"
    spec_names(8)%string = "HO2"
    spec_names(9)%string = "N2O5"
    spec_names(10)%string = "HNO3"
    spec_names(11)%string = "HONO"
    spec_names(12)%string = "PNA"
    spec_names(13)%string = "H2O2"
    spec_names(14)%string = "XO2"
    spec_names(15)%string = "XO2N"
    spec_names(16)%string = "NTR"
    spec_names(17)%string = "ROOH"
    spec_names(18)%string = "FORM"
    spec_names(19)%string = "ALD2"
    spec_names(20)%string = "ALDX"
    spec_names(21)%string = "PAR"
    spec_names(22)%string = "CO"
    spec_names(23)%string = "MEO2"
    spec_names(24)%string = "MEPX"
    spec_names(25)%string = "MEOH"
    spec_names(26)%string = "HCO3"
    spec_names(27)%string = "FACD"
    spec_names(28)%string = "C2O3"
    spec_names(29)%string = "PAN"
    spec_names(30)%string = "PACD"
    spec_names(31)%string = "AACD"
    spec_names(32)%string = "CXO3"
    spec_names(33)%string = "PANX"
    spec_names(34)%string = "ROR"
    spec_names(35)%string = "OLE"
    spec_names(36)%string = "ETH"
    spec_names(37)%string = "IOLE"
    spec_names(38)%string = "TOL"
    spec_names(39)%string = "CRES"
    spec_names(40)%string = "TO2"
    spec_names(41)%string = "TOLRO2"
    spec_names(42)%string = "OPEN"
    spec_names(43)%string = "CRO"
    spec_names(44)%string = "MGLY"
    spec_names(45)%string = "XYL"
    spec_names(46)%string = "XYLRO2"
    spec_names(47)%string = "ISOP"
    spec_names(48)%string = "ISPD"
    spec_names(49)%string = "ISOPRXN"
    spec_names(50)%string = "TERP"
    spec_names(51)%string = "TRPRXN"
    spec_names(52)%string = "SO2"
    spec_names(53)%string = "SULF"
    spec_names(54)%string = "SULRXN"
    spec_names(55)%string = "ETOH"
    spec_names(56)%string = "ETHA"
    spec_names(57)%string = "CL2"
    spec_names(58)%string = "CL"
    spec_names(59)%string = "HOCL"
    spec_names(60)%string = "CLO"
    spec_names(61)%string = "FMCL"
    spec_names(62)%string = "HCL"
    spec_names(63)%string = "TOLNRXN"
    spec_names(64)%string = "TOLHRXN"
    spec_names(65)%string = "XYLNRXN"
    spec_names(66)%string = "XYLHRXN"
    spec_names(67)%string = "BENZENE"
    spec_names(68)%string = "BENZRO2"
    spec_names(69)%string = "BNZNRXN"
    spec_names(70)%string = "BNZHRXN"
    spec_names(71)%string = "SESQ"
    spec_names(72)%string = "SESQRXN"

  end subroutine set_ebi_species

  subroutine set_monarch_species(spec_names)

    !> EBI solver species names
    type(string_t), dimension(NUM_EBI_SPEC) :: spec_names

    !Monarch order
    spec_names(1)%string = "NO2"
    spec_names(2)%string = "NO"
    spec_names(3)%string = "O3"
    spec_names(4)%string = "NO3"
    spec_names(5)%string = "N2O5"
    spec_names(6)%string = "HNO3"
    spec_names(7)%string = "HONO"
    spec_names(8)%string = "PNA"
    spec_names(9)%string = "H2O2"
    spec_names(10)%string = "NTR"
    spec_names(11)%string = "ROOH"
    spec_names(12)%string = "FORM"
    spec_names(13)%string = "ALD2"
    spec_names(14)%string = "ALDX"
    spec_names(15)%string = "PAR"
    spec_names(16)%string = "CO"
    spec_names(17)%string = "MEPX"
    spec_names(18)%string = "MEOH"
    spec_names(19)%string = "FACD"
    spec_names(20)%string = "PAN"
    spec_names(21)%string = "PACD"
    spec_names(22)%string = "AACD"
    spec_names(23)%string = "PANX"
    spec_names(24)%string = "OLE"
    spec_names(25)%string = "ETH"
    spec_names(26)%string = "IOLE"
    spec_names(27)%string = "TOL"
    spec_names(28)%string = "CRES"
    spec_names(29)%string = "OPEN"
    spec_names(30)%string = "MGLY"
    spec_names(31)%string = "XYL"
    spec_names(32)%string = "ISOP"
    spec_names(33)%string = "ISPD"
    spec_names(34)%string = "TERP"
    spec_names(35)%string = "SO2"
    spec_names(36)%string = "SULF"
    spec_names(37)%string = "ETOH"
    spec_names(38)%string = "ETHA"
    spec_names(39)%string = "CL2"
    spec_names(40)%string = "HOCL"
    spec_names(41)%string = "FMCL"
    spec_names(42)%string = "HCL"
    spec_names(43)%string = "BENZENE"
    spec_names(44)%string = "SESQ"
    spec_names(45)%string = "O"
    spec_names(46)%string = "O1D"
    spec_names(47)%string = "OH"
    spec_names(48)%string = "HO2"
    spec_names(49)%string = "XO2"
    spec_names(50)%string = "XO2N"
    spec_names(51)%string = "MEO2"
    spec_names(52)%string = "HCO3"
    spec_names(53)%string = "C2O3"
    spec_names(54)%string = "CXO3"
    spec_names(55)%string = "ROR"
    spec_names(56)%string = "TO2"
    spec_names(57)%string = "TOLRO2"
    spec_names(58)%string = "CRO"
    spec_names(59)%string = "XYLRO2"
    spec_names(60)%string = "ISOPRXN"
    spec_names(61)%string = "TRPRXN"
    spec_names(62)%string = "SULRXN"
    spec_names(63)%string = "CL"
    spec_names(64)%string = "CLO"
    spec_names(65)%string = "TOLNRXN"
    spec_names(66)%string = "TOLHRXN"
    spec_names(67)%string = "XYLNRXN"
    spec_names(68)%string = "XYLHRXN"
    spec_names(69)%string = "BENZRO2"
    spec_names(70)%string = "BNZNRXN"
    spec_names(71)%string = "BNZHRXN"
    spec_names(72)%string = "SESQRXN"

  end subroutine set_monarch_species

  subroutine set_ebi_photo_ids_with_camp(photo_id_camp)

    !> EBI solver species names
    integer, dimension(NUM_EBI_PHOTO_RXN) :: photo_id_camp

    !Monarch order
    photo_id_camp(1) = 1
    photo_id_camp(2) = 2
    photo_id_camp(3) = 3
    photo_id_camp(4) = 4
    photo_id_camp(5) = 5
    photo_id_camp(6) = 6
    photo_id_camp(7) = 7
    photo_id_camp(8) = 8
    photo_id_camp(9) = 9
    photo_id_camp(10) = 10
    photo_id_camp(11) = 11
    photo_id_camp(12) = 12
    photo_id_camp(13) = 14
    photo_id_camp(14) = 15
    photo_id_camp(15) = 16
    photo_id_camp(16) = 17
    photo_id_camp(17) = 18
    photo_id_camp(18) = 19
    photo_id_camp(19) = 22
    photo_id_camp(20) = 23
    photo_id_camp(21) = 25 !0.0
    photo_id_camp(22) = 25 !0.0
    photo_id_camp(23) = 25 !0.0s

  end subroutine set_ebi_photo_ids_with_camp

end program mock_monarch
