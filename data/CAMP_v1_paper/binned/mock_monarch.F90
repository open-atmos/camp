! Copyright (C) 2021 Barcelona Supercomputing Center and University of
! Illinois at Urbana-Champaign
! SPDX-License-Identifier: MIT

!> \file
!> The mock_monarch program

!> Mock version of the MONARCH model for testing integration with PartMC
program mock_monarch

  use camp_util,                          only : assert_msg, almost_equal, &
                                                to_string
  use camp_monarch_interface
  use camp_mpi

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

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Parameters for mock MONARCH model !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Number of total species in mock MONARCH
  integer, parameter :: NUM_MONARCH_SPEC = 800
  !> Number of vertical cells in mock MONARCH
  integer, parameter :: NUM_VERT_CELLS = 1
  !> Starting W-E cell for camp-chem call
  integer, parameter :: I_W = 1
  !> Ending W-E cell for camp-chem call
  integer, parameter :: I_E = 1
  !> Starting S-N cell for camp-chem call
  integer, parameter :: I_S = 1
  !> Ending S-N cell for camp-chem call
  integer, parameter :: I_N = 1
  !> Number of W-E cells in mock MONARCH
  integer, parameter :: NUM_WE_CELLS = I_E-I_W+1
  !> Number of S-N cells in mock MONARCH
  integer, parameter :: NUM_SN_CELLS = I_N-I_S+1
  !> Starting index for camp-chem species in tracer array
  integer, parameter :: START_CAMP_ID = 1!100
  !> Ending index for camp-chem species in tracer array
  integer, parameter :: END_CAMP_ID = 210!350
  !> Time step (min)
  real, parameter :: TIME_STEP = 2.!1.6
  !> Number of time steps to integrate over
  integer, parameter :: NUM_TIME_STEP = 720 !180!720!30
  !> Index for water vapor in water_conc()
  integer, parameter :: WATER_VAPOR_ID = 5
  !> Start time
  real, parameter :: START_TIME = 0.0
  !> Number of cells to compute simultaneously
  integer :: n_cells = 1
  !integer :: n_cells = (I_E - I_W+1)*(I_N - I_S+1)*NUM_VERT_CELLS
  !> Check multiple cells results are correct?
  logical :: check_multiple_cells = .false.

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! State variables for mock MONARCH model !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> NMMB style arrays (W->E, S->N, top->bottom, ...)
  !> Temperature (K)
  real :: temperature(NUM_WE_CELLS, NUM_SN_CELLS, NUM_VERT_CELLS)
  !> Species conc (various units)
  real :: species_conc(NUM_WE_CELLS, NUM_SN_CELLS, NUM_VERT_CELLS, NUM_MONARCH_SPEC)
  !> Water concentrations (kg_H2O/kg_air)
  real :: water_conc(NUM_WE_CELLS, NUM_SN_CELLS, NUM_VERT_CELLS, WATER_VAPOR_ID)

  !> WRF-style arrays (W->E, bottom->top, N->S)
  !> Air density (kg_air/m^3)
  real :: air_density(NUM_WE_CELLS, NUM_VERT_CELLS, NUM_SN_CELLS)
  !> Air pressure (Pa)
  real :: pressure(NUM_WE_CELLS, NUM_VERT_CELLS, NUM_SN_CELLS)
  !> Cell height (m)
  real :: height

  !> Emissions parameters
  !> Emission conversion parameter (mol s-1 m-2 to ppmv)
  real :: conv
  !> Emissions hour counter
  integer :: i_hour = 0

  !> Comparison values
  real :: comp_species_conc(0:NUM_TIME_STEP, NUM_MONARCH_SPEC)
  real :: species_conc_copy(NUM_WE_CELLS, NUM_SN_CELLS, NUM_VERT_CELLS, NUM_MONARCH_SPEC)

  !> Starting time for mock model run (min since midnight)
  !! is tracked in MONARCH
  real :: curr_time = START_TIME

  !> Set starting time for gnuplot scripts (includes initial conditions as first
  !! data point)
  real :: plot_start_time = START_TIME

  !> !!! Add to MONARCH variables !!!
  type(monarch_interface_t), pointer :: camp_interface

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Mock model setup and evaluation variables !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> CAMP-chem input file file
  character(len=:), allocatable :: camp_input_file
  !> PartMC-camp <-> MONARCH interface configuration file
  character(len=:), allocatable :: interface_input_file
  !> Results file prefix
  character(len=:), allocatable :: output_file_prefix
  !> CAMP-chem input file file
  type(string_t), allocatable :: name_gas_species_to_print(:), name_aerosol_species_to_print(:)
  integer(kind=i_kind), allocatable :: id_gas_species_to_print(:), id_aerosol_species_to_print(:)
  integer(kind=i_kind) :: size_gas_species_to_print, size_aerosol_species_to_print

  ! MPI
#ifdef CAMP_USE_MPI
  character, allocatable :: buffer(:)
  integer(kind=i_kind) :: pos, pack_size
#endif

  character(len=500) :: arg
  integer :: status_code, i_time, i_spec, i, j, k, z
  !> Partmc nº of cases to test
  integer :: camp_cases = 1
  integer :: plot_case

  ! Check the command line arguments
  call assert_msg(129432506, command_argument_count().eq.3, "Usage: "// &
          "./mock_monarch camp_input_file_list.json "// &
          "interface_input_file.json output_file_prefix")

  ! initialize mpi (to take the place of a similar MONARCH call)
  call camp_mpi_init()

  plot_case=4
  if(plot_case == 0)then
    size_gas_species_to_print=4
    size_aerosol_species_to_print=1
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
    name_aerosol_species_to_print(1)%string=("organic_matter.1.organic_matter.POA")
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
  elseif(plot_case ==4)then
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

  !Check if repeat program to compare n_cells=1 with n_cells=N
  if(check_multiple_cells) then
    camp_cases=2
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
  output_file_prefix = trim(arg)

  call model_initialize(output_file_prefix)

  !Repeat in case we want create a checksum
  do i=1, camp_cases

    camp_interface => monarch_interface_t(camp_input_file, interface_input_file, &
            START_CAMP_ID, END_CAMP_ID, n_cells)!, n_cells

    do j=1, size(name_gas_species_to_print)
      do z=1, size(camp_interface%monarch_species_names)
        if(camp_interface%monarch_species_names(z)%string.eq.name_gas_species_to_print(j)%string) then
          id_gas_species_to_print(j)=camp_interface%map_monarch_id(z)
        end if
      end do
    end do

    do j=1, size(name_aerosol_species_to_print)
      do z=1, size(camp_interface%monarch_species_names)
        if(camp_interface%monarch_species_names(z)%string.eq.name_aerosol_species_to_print(j)%string) then
          id_aerosol_species_to_print(j)=camp_interface%map_monarch_id(z)
        end if
      end do
    end do


    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! **** end initialization modification **** !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ! Set conc from mock_model
    call camp_interface%get_init_conc(species_conc, water_conc, WATER_VAPOR_ID, &
            air_density)

    ! Run the model
    do i_time=1, NUM_TIME_STEP

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! **** Add to MONARCH during runtime for each time step **** !
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      call output_results(curr_time,camp_interface,species_conc)
      call camp_interface%integrate(curr_time,         & ! Starting time (min)
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

#ifdef CAMP_USE_MPI
    if (camp_mpi_rank().eq.0) then
      write(*,*) "Model run time: ", comp_time, " s"
    end if
#else
    write(*,*) "Model run time: ", comp_time, " s"
#endif

    !Save results
    if(i.eq.1) then
      species_conc_copy(:,:,:,:) = species_conc(:,:,:,:)
    end if

    ! Set 1 cell to get a checksum case
    n_cells = 1

  end do

  !If something to compare
  if(camp_cases.gt.1) then
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


  ! Output results and scripts
  if (camp_mpi_rank().eq.0) then
    write(*,*) "MONARCH interface tests - PASS"
    call output_results(curr_time,camp_interface,species_conc)
    call create_gnuplot_script(camp_interface, output_file_prefix, &
            plot_start_time, curr_time)
    call create_gnuplot_persist(camp_interface, output_file_prefix, &
            plot_start_time, curr_time)
  end if

  close(RESULTS_FILE_UNIT)
  close(RESULTS_FILE_UNIT_TABLE)

  ! Deallocation
  deallocate(camp_input_file)
  deallocate(interface_input_file)
  deallocate(output_file_prefix)

  ! finalize mpi
  call camp_mpi_finalize()

  ! Free the interface and the solver
  deallocate(camp_interface)

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Initialize the mock model
  subroutine model_initialize(file_prefix)

    !> File prefix for model results
    character(len=:), allocatable, intent(in) :: file_prefix

    integer :: i_spec
    character(len=:), allocatable :: file_name

    ! Open the output file
    file_name = file_prefix//"_results.txt"
    open(RESULTS_FILE_UNIT, file=file_name, status="replace", action="write")
    file_name = file_prefix//"_results_table.txt"
    open(RESULTS_FILE_UNIT_TABLE, file=file_name, status="replace", action="write")

    ! TODO refine initial model conditions
    species_conc(:,:,:,:) = 0.0
    water_conc(:,:,:,:) = 0.0
    water_conc(:,:,:,WATER_VAPOR_ID) = 0. !0.01165447 !this is equal to 95% RH !1e-14 !0.01 !kg_h2o/kg-1_air
    height=1. !(m)
#ifndef ENABLE_CB05_SOA
    temperature(:,:,:) = 290.016!300.614166259766
    pressure(:,:,:) = 100000.!94165.7187500000
    air_density(:,:,:) = pressure(:,:,:)/(287.04*temperature(:,:,:)* &
         (1.+0.60813824*water_conc(:,:,:,WATER_VAPOR_ID))) !kg m-3
    conv=0.02897/air_density(1,1,1)*(TIME_STEP*60.)*1e6/height !units of time_step to seconds
#else
    temperature(:,:,:) = 300.614166259766
    pressure(:,:,:) = 94165.7187500000
    air_density(:,:,:) = 1.225
    conv=0.02897/air_density(1,1,1)*(TIME_STEP*60.)*1e6/height !units of time_step to seconds

    !Initialize different axis values
    !Species_conc is modified in monarch_interface%get_init_conc
    do i=I_W, I_E
      temperature(i,:,:) = temperature(i,:,:) + 0.1*i
      pressure(i,:,:) = pressure(i,:,:) - 1*i
    end do

    do j=I_S, I_N
      temperature(:,j,:) = temperature(:,j,:) + 0.3*j
      pressure(:,:,j) = pressure(:,:,j) - 3*j
    end do

    do k=1, NUM_VERT_CELLS
      temperature(:,:,k) = temperature(:,:,k) + 0.6*k
      pressure(:,k,:) = pressure(:,k,:) - 6*k
    end do

#endif

    deallocate(file_name)

  end subroutine model_initialize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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

  !> Output the model results
  subroutine output_results(curr_time,camp_interface,species_conc)

    !> Current model time (min since midnight)
    real, intent(in) :: curr_time
    type(monarch_interface_t), intent(in) :: camp_interface
    integer :: z,i,j,k
    real, intent(inout) :: species_conc(:,:,:,:)

    character(len=:), allocatable :: aux_str
    real, allocatable :: aux_real
    logical, save :: first_time=.true.

    write(RESULTS_FILE_UNIT_TABLE, *) "Time_step:", curr_time

    do i=I_W,I_E
      do j=I_W,I_E
        do k=I_W,I_E
          write(RESULTS_FILE_UNIT_TABLE, *) "i:",i,"j:",j,"k:",k
          write(RESULTS_FILE_UNIT_TABLE, *) "Spec_name, Concentrations, Map_monarch_id"
          do z=1, size(camp_interface%monarch_species_names)
            write(RESULTS_FILE_UNIT_TABLE, *) camp_interface%monarch_species_names(z)%string&
            , species_conc(i,j,k,camp_interface%map_monarch_id(z))&
            , camp_interface%map_monarch_id(z)
          end do
        end do
      end do
    end do

    !print Titles
    if(first_time)then
    aux_str = "Time(min)"
    do z=1, size(name_gas_species_to_print)
      aux_str = aux_str//" "//name_gas_species_to_print(z)%string
    end do

    do z=1, size(name_aerosol_species_to_print)
      aux_str = aux_str//" "//name_aerosol_species_to_print(z)%string
    end do

    write(RESULTS_FILE_UNIT, "(A)", advance="no") aux_str
    write(RESULTS_FILE_UNIT, "(A)", advance="yes") " "
    first_time=.false.
    endif

    write(RESULTS_FILE_UNIT, "(i4.4)", advance="no") int(curr_time)
    do z=1, size(name_gas_species_to_print)
      write(RESULTS_FILE_UNIT, "(ES13.6)", advance="no") &
              species_conc(1,1,1,id_gas_species_to_print(z))
    end do
    do z=1, size(name_aerosol_species_to_print)
      write(RESULTS_FILE_UNIT, "(ES13.6)", advance="no") &
      species_conc(1,1,1,id_aerosol_species_to_print(z))
    end do

    write(RESULTS_FILE_UNIT, "(A)", advance="yes") " "

  end subroutine output_results

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Create a gnuplot script for viewing species concentrations
  subroutine create_gnuplot_script(camp_interface, file_prefix, start_time, &
            end_time)

    !> PartMC-camp <-> MONARCH interface
    type(monarch_interface_t), intent(in) :: camp_interface
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

    ! Get the species names and ids
    call camp_interface%get_MONARCH_species(species_names, tracer_ids)

    ! Adjust the tracer ids to match the results file
    tracer_ids(:) = tracer_ids(:) - START_CAMP_ID + 2

    ! Create the gnuplot script
    file_name = file_prefix//".conf"
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
      write(SCRIPTS_FILE_UNIT,*) "set output '"//file_prefix//"_"// &
              spec_name//".png'"
      write(SCRIPTS_FILE_UNIT,*) "plot\"
      write(SCRIPTS_FILE_UNIT,*) " '"//file_prefix//"_results.txt'\"
      write(SCRIPTS_FILE_UNIT,*) " using 1:"// &
              trim(to_string(tracer_ids(i_spec)))//" title '"// &
              species_names(i_spec)%string//" (MONARCH)'"
    end do
    tracer_id = END_CAMP_ID - START_CAMP_ID + 3
    write(SCRIPTS_FILE_UNIT,*) "set output '"//file_prefix//"_H2O.png'"
    write(SCRIPTS_FILE_UNIT,*) "plot\"
    write(SCRIPTS_FILE_UNIT,*) " '"//file_prefix//"_results.txt'\"
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
  subroutine create_gnuplot_persist(camp_interface, file_prefix, start_time, &
          end_time)

    !> PartMC-camp <-> MONARCH interface
    type(monarch_interface_t), intent(in) :: camp_interface
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
    call camp_interface%get_MONARCH_species(species_names, tracer_ids)

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
    file_name = file_prefix//".gnuplot"
    open(unit=SCRIPTS_FILE_UNIT, file=file_name, status="replace", action="write")
    write(SCRIPTS_FILE_UNIT,*) "# "//file_name
    write(SCRIPTS_FILE_UNIT,*) "# Run as: gnuplot -persist "//file_name
    write(SCRIPTS_FILE_UNIT,*) "set terminal jpeg medium size 640,480 truecolor"
    write(SCRIPTS_FILE_UNIT,*) "set title 'Mock_monarch_cb05_soa'"
    write(SCRIPTS_FILE_UNIT,*) "set xlabel 'Time (min)'"
    write(SCRIPTS_FILE_UNIT,*) "set ylabel 'Gas concentration (ppmv)'"
    write(SCRIPTS_FILE_UNIT,*) "set y2label 'Aerosol concentration (kg/m^3)'"
    write(SCRIPTS_FILE_UNIT,*) "set ytics nomirror"
    write(SCRIPTS_FILE_UNIT,*) "set y2tics nomirror"

    write(SCRIPTS_FILE_UNIT,*) "set logscale y"
    write(SCRIPTS_FILE_UNIT,*) "set logscale y2"
    write(SCRIPTS_FILE_UNIT,*) "set xrange [", start_time, ":", end_time, "]"

    i_spec=1
    spec_name = species_names(i_spec)%string
    forall (i_char = 1:len(spec_name), spec_name(i_char:i_char).eq.'/') &
          spec_name(i_char:i_char) = '_'

    write(SCRIPTS_FILE_UNIT,*) "set key top left"
    write(SCRIPTS_FILE_UNIT,"(A)",advance="no") "set output 'out/monarch_plot.jpg'"
    write(SCRIPTS_FILE_UNIT,*)
    write(SCRIPTS_FILE_UNIT,"(A)",advance="no") "plot for [col=2:"&
    //trim(n_gas_species_plot_str)//"] '"//file_prefix//"_results.txt' &
    using 1:col axis x1y1 title columnheader, for [col2=" &
    //trim(n_aerosol_species_start_plot_str)//":" &
    //trim(n_aerosol_species_plot_str)//"] &
    '"//file_prefix//"_results.txt' &
    using " &
    //trim(n_aerosol_species_time_plot_str)// &
    ":col2 axis x1y2 title columnheader"

    close(SCRIPTS_FILE_UNIT)

    deallocate(species_names)
    deallocate(tracer_ids)
    deallocate(file_name)
    deallocate(spec_name)

  end subroutine create_gnuplot_persist

end program mock_monarch
