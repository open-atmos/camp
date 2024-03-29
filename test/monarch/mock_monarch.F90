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

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Parameters for mock MONARCH model !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Number of total species in mock MONARCH
  integer, parameter :: NUM_MONARCH_SPEC = 800
  !> Number of vertical cells in mock MONARCH
  integer, parameter :: NUM_VERT_CELLS = 3
  !> Starting W-E cell for camp-chem call
  integer, parameter :: I_W = 1
  !> Ending W-E cell for camp-chem call
  integer, parameter :: I_E = 5
  !> Starting S-N cell for camp-chem call
  integer, parameter :: I_S = 1
  !> Ending S-N cell for camp-chem call
  integer, parameter :: I_N = 5
  !> Number of W-E cells in mock MONARCH
  integer, parameter :: NUM_WE_CELLS = I_E-I_W+1
  !> Number of S-N cells in mock MONARCH
  integer, parameter :: NUM_SN_CELLS = I_N-I_S+1
  !> Starting index for camp-chem species in tracer array
  integer, parameter :: START_CAMP_ID = 100
  !> Ending index for camp-chem species in tracer array
  integer, parameter :: END_CAMP_ID = 650
  !> Time step (min)
  real, parameter :: TIME_STEP = 1.6
  !> Number of time steps to integrate over
  integer, parameter :: NUM_TIME_STEP = 5
  !> Index for water vapor in water_conc()
  integer, parameter :: WATER_VAPOR_ID = 5
  !> Start time
  real, parameter :: START_TIME = 360.0
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

  character(len=500) :: arg
  integer :: status_code, i_time, i_spec, i, j, k
  !> Partmc nº of cases to test
  integer :: camp_cases = 1


  ! Check the command line arguments
  call assert_msg(129432506, command_argument_count().eq.3, "Usage: "// &
          "./mock_monarch camp_input_file_list.json "// &
          "interface_input_file.json output_file_prefix")

  ! initialize mpi (to take the place of a similar MONARCH call)
  call camp_mpi_init()

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

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! **** end initialization modification **** !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ! Set conc from mock_model
    call camp_interface%get_init_conc(species_conc, water_conc, WATER_VAPOR_ID, &
            air_density)

    ! call camp_interface%print( )

    ! Run the model
    do i_time=0, NUM_TIME_STEP

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! **** Add to MONARCH during runtime for each time step **** !
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      call output_results(curr_time)
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
                                   pressure)            ! Air pressure (Pa)
      curr_time = curr_time + TIME_STEP

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! **** end runtime modification **** !
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    end do

    write(*,*) "Model run time: ", comp_time, " s"

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
                  trim( to_string( species_conc_copy(i,j,k,i_spec) ) )//", got: "// &
                  trim( to_string( species_conc(i,j,k,i_spec) ) ) )
          end do
        end do
      end do
    end do
  end if

  write(*,*) "MONARCH interface tests - PASS"

  ! Output results and scripts
  if (camp_mpi_rank().eq.0) then
    call output_results(curr_time)
    call create_gnuplot_script(camp_interface, output_file_prefix, &
            plot_start_time, curr_time)
  end if

  ! TODO I would still like to implement this once the results are stable
  ! The evaluation is based on a run with reasonable seeming values and
  ! few solver modifications. It is used to make sure future modifications
  ! to the solver do not affect the results
#if 0
  do i_spec = START_CAMP_ID, END_CAMP_ID
    call assert_msg( 394742768, &
        almost_equal( real( species_conc(10,15,1,i_spec), kind=dp ), &
                      real( comp_species_conc(i_time,i_spec), kind=dp ), &
                      1.d-4, 1d-3 ), &
        "Concentration species mismatch for species "// &
        trim( to_string( i_spec ) )//" at time step "// &
        trim( to_string( i_time ) )//". Expected: "// &
        trim( to_string( comp_species_conc(i_time,i_spec) ) )//", got: "// &
        trim( to_string( species_conc(10,15,1,i_spec) ) ) )
  end do
#endif

  ! Deallocation
  deallocate(camp_input_file)
  deallocate(interface_input_file)

  ! Free the interface and the solver
  deallocate(camp_interface)

  ! close the output file
  close(RESULTS_FILE_UNIT)
  deallocate(output_file_prefix)

  ! finalize mpi
  call camp_mpi_finalize()

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

    ! Open the compare file
    ! TODO Implement once results are stable
#if 0
    file_name = file_prefix//"_comp.txt"
    open(COMPARE_FILE_UNIT, file=file_name, action="read")
#endif

    ! TODO refine initial model conditions
    temperature(:,:,:) = 300.614166259766
    species_conc(:,:,:,:) = 0.0
    water_conc(:,:,:,:) = 0.0
    water_conc(:,:,:,WATER_VAPOR_ID) = 0.01
    air_density(:,:,:) = 1.225
    pressure(:,:,:) = 94165.7187500000

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

    deallocate(file_name)

    ! Read the compare file
    ! TODO Implement once results are stable
#if 0
    call read_comp_file()
    close(COMPARE_FILE_UNIT)
#endif

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
  subroutine output_results(curr_time)

    !> Current model time (min since midnight)
    real, intent(in) :: curr_time

    write(RESULTS_FILE_UNIT, *) curr_time, &
            species_conc(2,3,1,START_CAMP_ID:END_CAMP_ID), &
            water_conc(2,3,1,WATER_VAPOR_ID)

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

end program mock_monarch
