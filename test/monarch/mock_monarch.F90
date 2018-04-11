! Copyright (C) 2017 Matt Dawson
! Licensed under the GNU General Public License version 2 or (at your
! option) any later version. See the file COPYING for details.

!> \file
!> The mock_monarch program

!> Mock version of the MONARCH model for testing integration with PartMC
program mock_monarch

  use pmc_util,                                 only : assert_msg
  use pmc_monarch
  use pmc_mpi

  implicit none

  !> File unit for model run-time messages
  integer, parameter :: OUTPUT_FILE_UNIT = 6
  !> File unit for model results
  integer, parameter :: RESULTS_FILE_UNIT = 15

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Parameters for mock MONARCH model !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Number of total species in mock MONARCH
  integer, parameter :: NUM_MONARCH_SPEC = 800
  !> Number of W-E cells in mock MONARCH
  integer, parameter :: NUM_WE_CELLS = 20
  !> Number of S-N cells in mock MONARCH
  integer, parameter :: NUM_SN_CELLS = 30
  !> Number of vertical cells in mock MONARCH
  integer, parameter :: NUM_VERT_CELLS = 1
  !> Starting W-E cell for phlex-chem call
  integer, parameter :: I_W = 9
  !> Ending W-E cell for phlex-chem call
  integer, parameter :: I_E = 11
  !> Starting S-N cell for phlex-chem call
  integer, parameter :: I_S = 14
  !> Ending S-N cell for phlex-chem call
  integer, parameter :: I_N = 16
  !> Starting index for phlex-chem species in tracer array
  integer, parameter :: START_PHLEX_ID = 100
  !> Ending index for phlex-chem species in tracer array
  integer, parameter :: END_PHLEX_ID = 650
  !> Time step (min)
  real, parameter :: TIME_STEP = 0.1
  !> Number of time steps to integrate over
  integer, parameter :: NUM_TIME_STEP = 100
  !> Index for water vapor in water_conc()
  integer, parameter :: WATER_VAPOR_ID = 5

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

  !> Starting time for mock model run (min since midnight) TODO check how time
  !! is tracked in MONARCH
  real :: start_time = 360.0

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Mock model setup and evaluation variables !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Path to phlex-chem input file list
  character(len=:), allocatable :: phlex_input_file
  !> Path to results file
  character(len=:), allocatable :: output_file
 
  character(len=500) :: arg
  integer :: status_code, i_time


  ! Check the command line arguments
  call assert_msg(129432506, command_argument_count().eq.2, "Usage: "// &
          "./mock_monarch phlex_input_file_list.json output_file.txt")

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! **** Add to MONARCH during initialization **** !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  ! Initialize PartMC-phlex
  call get_command_argument(1, arg, status=status_code)
  call assert_msg(678165802, status_code.eq.0, "Error getting PartMC-phlex "//&
          "configuration file name")
  phlex_input_file = trim(arg)
  call pmc_initialize(phlex_input_file, START_PHLEX_ID, END_PHLEX_ID)

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! **** end initialization modification **** !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  ! Initialize the mock model
  call get_command_argument(2, arg, status=status_code)
  call assert_msg(234156729, status_code.eq.0, "Error getting output file name")
  output_file = trim(arg)
  call model_initialize(output_file)

  ! Run the model
  do i_time=0, NUM_TIME_STEP
  
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! **** Add to MONARCH during runtime for each time step **** !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    call output_results(start_time)
    call pmc_integrate(start_time,        & ! Starting time (min)
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
    start_time = start_time + TIME_STEP
  
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! **** end runtime modification **** !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
  end do

  write(*,*) "Model run time: ", comp_time, " s"

  call output_results(start_time)

  ! TODO evaluate results

  write(*,*) "MONARCH interface tests - PASS"

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Initialize the mock model
  subroutine model_initialize(output_file)

    !> Path to output file
    character(len=:), allocatable, intent(in) :: output_file

    integer :: i_spec

    ! Open the output file
    open(RESULTS_FILE_UNIT, file=output_file, status="new", action="write")

    ! Initialize MPI
    call pmc_mpi_init()

    ! TODO refine initial model conditions
    temperature(:,:,:) = 298.0
    species_conc(:,:,:,:) = 0.0
    water_conc(:,:,:,:) = 0.0
    water_conc(:,:,:,WATER_VAPOR_ID) = 0.01
    air_density(:,:,:) = 1.225
    pressure(:,:,:) = 101325.0

    species_conc(:,:,:,START_PHLEX_ID) = 1.0 ! Species A

  end subroutine model_initialize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Output the model results
  subroutine output_results(curr_time)

    !> Current model time (min since midnight)
    real, intent(in) :: curr_time

    write(RESULTS_FILE_UNIT, *) curr_time, species_conc(10,15,1,START_PHLEX_ID:)

  end subroutine output_results

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end program mock_monarch 