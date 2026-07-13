! Copyright (C) 2021 Barcelona Supercomputing Center and University of
! Illinois at Urbana-Champaign
! SPDX-License-Identifier: MIT

!> \file
!> The mock_monarch program

!> Mock version of the MONARCH model for testing integration with PartMC
program mock_monarch

  use camp_util, only: assert_msg, almost_equal, &
                       to_string
  use camp_monarch_interface
  use camp_mpi
  use camp_aero_rep_data, only: aero_rep_data_t
  use camp_constants

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
  !> File unit for aerosol physical properties (size, number)
  integer, parameter :: RESULTS_FILE_AERO_UNIT = 11
  !> File unit for environment properties (pressure, temperature, ...)
  integer, parameter :: RESULTS_FILE_ENV_UNIT = 12

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
  integer, parameter :: NUM_WE_CELLS = I_E - I_W + 1
  !> Number of S-N cells in mock MONARCH
  integer, parameter :: NUM_SN_CELLS = I_N - I_S + 1
  !> Starting index for camp-chem species in tracer array
  integer, parameter :: START_CAMP_ID = 1!100
  !> Ending index for camp-chem species in tracer array
  integer, parameter :: END_CAMP_ID = 286!350
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

  !> relative humidity
  real :: rh

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
  logical :: found
  !> Partmc nº of cases to test
  integer :: camp_cases = 1
  integer :: plot_case

  ! Check the command line arguments
  call assert_msg(129432506, command_argument_count() .eq. 3, "Usage: "// &
                  "./mock_monarch camp_input_file_list.json "// &
                  "interface_input_file.json output_file_prefix")

  ! initialize mpi (to take the place of a similar MONARCH call)
  call camp_mpi_init()

  plot_case = 5
  if (plot_case == 0) then
    size_gas_species_to_print = 4
    size_aerosol_species_to_print = 1
  elseif (plot_case == 1) then
    size_gas_species_to_print = 1
    size_aerosol_species_to_print = 1
  elseif (plot_case == 2 .or. plot_case == 3) then
    size_gas_species_to_print = 3
    size_aerosol_species_to_print = 2
  elseif (plot_case == 4) then
    size_gas_species_to_print = 5
    size_aerosol_species_to_print = 9
    ! if plot_case == 5, everything is printed
  end if

  if (plot_case /= 5) then
    allocate (name_gas_species_to_print(size_gas_species_to_print))
    allocate (name_aerosol_species_to_print(size_aerosol_species_to_print))
    allocate (id_gas_species_to_print(size_gas_species_to_print))
    allocate (id_aerosol_species_to_print(size_aerosol_species_to_print))
  end if

  if (plot_case == 0) then
    name_gas_species_to_print(1)%string = ("O3")
    name_gas_species_to_print(2)%string = ("NO2")
    name_gas_species_to_print(3)%string = ("NO")
    name_gas_species_to_print(4)%string = ("ISOP")
    name_aerosol_species_to_print(1)%string = ("organic_matter.1.organic_matter.POA")
  elseif (plot_case == 1) then
    name_gas_species_to_print(1)%string = ("OH")
    name_aerosol_species_to_print(1)%string = ("organic_matter.1.organic_matter.POA")
  elseif (plot_case == 2) then
    name_gas_species_to_print(1)%string = ("ISOP")
    name_gas_species_to_print(2)%string = ("ISOP-P1")
    name_gas_species_to_print(3)%string = ("ISOP-P2")
    name_aerosol_species_to_print(1)%string = ("organic_matter.1.organic_matter.ISOP-P1_aero")
    name_aerosol_species_to_print(2)%string = ("organic_matter.1.organic_matter.ISOP-P2_aero")
  elseif (plot_case == 3) then
    name_gas_species_to_print(1)%string = ("TERP")
    name_gas_species_to_print(2)%string = ("TERP-P1")
    name_gas_species_to_print(3)%string = ("TERP-P2")
    name_aerosol_species_to_print(1)%string = ("organic_matter.1.organic_matter.TERP-P1_aero")
    name_aerosol_species_to_print(2)%string = ("organic_matter.1.organic_matter.TERP-P2_aero")
  elseif (plot_case == 4) then
    name_gas_species_to_print(1)%string = ("O3")
    name_gas_species_to_print(2)%string = ("NO2")
    name_gas_species_to_print(3)%string = ("ISOP")
    name_gas_species_to_print(4)%string = ("ISOP-P1")
    name_gas_species_to_print(5)%string = ("ISOP-P2")
    name_aerosol_species_to_print(1)%string = ("organic_matter.1.organic_matter.POA")
    name_aerosol_species_to_print(2)%string = ("organic_matter.2.organic_matter.POA")
    name_aerosol_species_to_print(3)%string = ("organic_matter.3.organic_matter.POA")
    name_aerosol_species_to_print(4)%string = ("organic_matter.1.organic_matter.ISOP-P1_aero")
    name_aerosol_species_to_print(5)%string = ("organic_matter.2.organic_matter.ISOP-P1_aero")
    name_aerosol_species_to_print(6)%string = ("organic_matter.3.organic_matter.ISOP-P1_aero")
    name_aerosol_species_to_print(7)%string = ("organic_matter.1.organic_matter.ISOP-P2_aero")
    name_aerosol_species_to_print(8)%string = ("organic_matter.2.organic_matter.ISOP-P2_aero")
    name_aerosol_species_to_print(9)%string = ("organic_matter.3.organic_matter.ISOP-P2_aero")
  end if

  !Check if repeat program to compare n_cells=1 with n_cells=N
  if (check_multiple_cells) then
    camp_cases = 2
  end if

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! **** Add to MONARCH during initialization **** !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  ! Initialize PartMC-camp
  call get_command_argument(1, arg, status=status_code)
  call assert_msg(678165802, status_code .eq. 0, "Error getting PartMC-camp "// &
                  "configuration file name")
  camp_input_file = trim(arg)
  call get_command_argument(2, arg, status=status_code)
  call assert_msg(664104564, status_code .eq. 0, "Error getting PartMC-camp "// &
                  "<-> MONARCH interface configuration file name")
  interface_input_file = trim(arg)

  ! Initialize the mock model
  call get_command_argument(3, arg, status=status_code)
  call assert_msg(234156729, status_code .eq. 0, "Error getting output file prefix")
  output_file_prefix = trim(arg)

  call model_initialize(output_file_prefix)

  !Repeat in case we want create a checksum
  do i = 1, camp_cases

    camp_interface => monarch_interface_t(camp_input_file, interface_input_file, &
                                          START_CAMP_ID, END_CAMP_ID, n_cells)!, n_cells

    if (plot_case == 5) then
      ! get the species names to print them all
      size_gas_species_to_print = size(camp_interface%monarch_species_names)
      size_aerosol_species_to_print = 0

      allocate (name_gas_species_to_print(size_gas_species_to_print))
      allocate (name_aerosol_species_to_print(size_aerosol_species_to_print))
      allocate (id_gas_species_to_print(size_gas_species_to_print))
      allocate (id_aerosol_species_to_print(size_aerosol_species_to_print))

      do z = 1, size_gas_species_to_print
        name_gas_species_to_print(z)%string = camp_interface%monarch_species_names(z)%string
      end do

    end if

    id_gas_species_to_print(:) = -1
    id_aerosol_species_to_print(:) = -1

    do j = 1, size(name_gas_species_to_print)
      found = .false.
      do z = 1, size(camp_interface%monarch_species_names)
        if (camp_interface%monarch_species_names(z)%string .eq. name_gas_species_to_print(j)%string) then
          id_gas_species_to_print(j) = camp_interface%map_monarch_id(z)
          found = .true.
          exit
        end if
      end do
      call assert_msg(826469926, found, "gas species to print "// &
                      name_gas_species_to_print(j)%string//" not found")
    end do

    do j = 1, size(name_aerosol_species_to_print)
      found = .false.
      do z = 1, size(camp_interface%monarch_species_names)
        if (camp_interface%monarch_species_names(z)%string .eq. name_aerosol_species_to_print(j)%string) then
          id_aerosol_species_to_print(j) = camp_interface%map_monarch_id(z)
          found = .true.
          exit
        end if
      end do
      call assert_msg(779635070, found, "aerosol species to print "// &
                      name_aerosol_species_to_print(j)%string//" not found")
    end do

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! **** end initialization modification **** !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ! Set conc from mock_model
    call camp_interface%get_init_conc(species_conc, water_conc, WATER_VAPOR_ID, &
                                      air_density)

    ! Run the model
    do i_time = 1, NUM_TIME_STEP

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! **** Add to MONARCH during runtime for each time step **** !
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      call output_results(curr_time, camp_interface, species_conc, water_conc, WATER_VAPOR_ID, pressure, temperature, rh)

! vary environmental conditions
      call external_forcings(curr_time, air_density, pressure, temperature, water_conc, WATER_VAPOR_ID, rh)

      call camp_interface%integrate(curr_time, & ! Starting time (min)
                                    TIME_STEP, & ! Time step (min)
                                    I_W, & ! Starting W->E grid cell
                                    I_E, & ! Ending W->E grid cell
                                    I_S, & ! Starting S->N grid cell
                                    I_N, & ! Ending S->N grid cell
                                    temperature, & ! Temperature (K)
                                    species_conc, & ! Tracer array
                                    water_conc, & ! Water concentrations (kg_H2O/kg_air)
                                    WATER_VAPOR_ID, & ! Index in water_conc() corresponding to water vapor
                                    air_density, & ! Air density (kg_air/m^3)
                                    pressure, & ! Air pressure (Pa)
                                    conv, &
                                    i_hour)
      curr_time = curr_time + TIME_STEP

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! **** end runtime modification **** !
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    end do

#ifdef CAMP_USE_MPI
    if (camp_mpi_rank() .eq. 0) then
      write (*, *) "Model run time: ", comp_time, " s"
    end if
#else
    write (*, *) "Model run time: ", comp_time, " s"
#endif

    !Save results
    if (i .eq. 1) then
      species_conc_copy(:, :, :, :) = species_conc(:, :, :, :)
    end if

    ! Set 1 cell to get a checksum case
    n_cells = 1

  end do

  !If something to compare
  if (camp_cases .gt. 1) then
    !Compare results
    do i = I_W, I_E
      do j = I_S, I_N
        do k = 1, NUM_VERT_CELLS
          do i_spec = START_CAMP_ID, END_CAMP_ID
            call assert_msg(394742768, &
                            almost_equal(real(species_conc(i, j, k, i_spec), kind=dp), &
                                         real(species_conc_copy(i, j, k, i_spec), kind=dp), &
                                         1.d-5, 1d-4), &
                            "Concentration species mismatch for species "// &
                            trim(to_string(i_spec))//". Expected: "// &
                            trim(to_string(species_conc(i, j, k, i_spec)))//", got: "// &
                            trim(to_string(species_conc_copy(i, j, k, i_spec))))
          end do
        end do
      end do
    end do
  end if

  ! Output results and scripts
  if (camp_mpi_rank() .eq. 0) then
    write (*, *) "MONARCH interface tests - PASS"
    call output_results(curr_time, camp_interface, species_conc, water_conc, WATER_VAPOR_ID, pressure, temperature, rh)
    call create_gnuplot_script(camp_interface, output_file_prefix, &
                               plot_start_time, curr_time)
    call create_gnuplot_persist(camp_interface, output_file_prefix, &
                                plot_start_time, curr_time)
  end if

  close (RESULTS_FILE_UNIT)
  close (RESULTS_FILE_UNIT_TABLE)
  close (RESULTS_FILE_AERO_UNIT)
  close (RESULTS_FILE_ENV_UNIT)

  ! Deallocation
  deallocate (camp_input_file)
  deallocate (interface_input_file)
  deallocate (output_file_prefix)

  ! finalize mpi
  call camp_mpi_finalize()

  ! Free the interface and the solver
  deallocate (camp_interface)

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
    open (RESULTS_FILE_UNIT, file=file_name, status="replace", action="write")
    file_name = file_prefix//"_results_table.txt"
    open (RESULTS_FILE_UNIT_TABLE, file=file_name, status="replace", action="write")
    file_name = file_prefix//"_results_aero.txt"
    open (RESULTS_FILE_AERO_UNIT, file=file_name, status="replace", action="write")
    file_name = file_prefix//"_results_env.txt"
    open (RESULTS_FILE_ENV_UNIT, file=file_name, status="replace", action="write")

    ! TODO refine initial model conditions
    species_conc(:, :, :, :) = 0.0
    water_conc(:, :, :, :) = 0.0
    water_conc(:, :, :, WATER_VAPOR_ID) = 0.01076565! 0.01165447 !this is equal to 95% RH !1e-14 !0.01 !kg_h2o/kg-1_air
    height = 1. !(m)
#ifndef ENABLE_CB05_SOA
    temperature(:, :, :) = 290.016!300.614166259766
    pressure(:, :, :) = 100000.!94165.7187500000
    air_density(:, :, :) = pressure(:, :, :)/(287.04*temperature(:, :, :)* &
                                              (1.+0.60813824*water_conc(:, :, :, WATER_VAPOR_ID))) !kg m-3
    conv = 0.02897/air_density(1, 1, 1)*(TIME_STEP*60.)*1e6/height !units of time_step to seconds
#else
    temperature(:, :, :) = 300.614166259766
    pressure(:, :, :) = 94165.7187500000
    air_density(:, :, :) = 1.225
    conv = 0.02897/air_density(1, 1, 1)*(TIME_STEP*60.)*1e6/height !units of time_step to seconds

    !Initialize different axis values
    !Species_conc is modified in monarch_interface%get_init_conc
    do i = I_W, I_E
      temperature(i, :, :) = temperature(i, :, :) + 0.1*i
      pressure(i, :, :) = pressure(i, :, :) - 1*i
    end do

    do j = I_S, I_N
      temperature(:, j, :) = temperature(:, j, :) + 0.3*j
      pressure(:, :, j) = pressure(:, :, j) - 3*j
    end do

    do k = 1, NUM_VERT_CELLS
      temperature(:, :, k) = temperature(:, :, k) + 0.6*k
      pressure(:, k, :) = pressure(:, k, :) - 6*k
    end do

#endif

    deallocate (file_name)

  end subroutine model_initialize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Read the comparison file (must have same dimensions as current config)
  subroutine read_comp_file()

    integer :: i_time
    real :: time, water

    do i_time = 0, NUM_TIME_STEP + 1
      read (COMPARE_FILE_UNIT, *) time, &
        comp_species_conc(i_time, START_CAMP_ID:END_CAMP_ID), &
        water
    end do

  end subroutine read_comp_file

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Update environmental conditions
  subroutine external_forcings(curr_time__min, air_density__kg_m3, pressure__Pa, temperature__K, water_conc, water_vapor_index, rh)
    real, intent(in) :: curr_time__min
    real, dimension(:, :, :), intent(out) :: temperature__K, pressure__Pa
    real, dimension(:, :, :), intent(inout) :: air_density__kg_m3
    real, intent(out) :: water_conc(:, :, :, :) ! kg(water)/kg(air)
    real, intent(out) :: rh
    integer, intent(in) :: water_vapor_index

    real :: time__h, rh__pc, water__ppm

    time__h = curr_time__min/60.

    ! update pressure, temperature and rh, simple sinusoïds
    pressure__Pa = 100000.00 ! in Pa
    temperature__K = 293. !+ 5. * sin(2. * time__h * const%pi / 24.) ! K
    rh__pc = 0.90 ! 0.75 + 0.19*sin(2. * time__h * const%pi/24)

    rh = rh__pc

    ! update air density
    air_density__kg_m3(:, :, :) = pressure__Pa(:, :, :)/(287.04*temperature__K(:, :, :)* &
                                                         (1.+0.60813824*water_conc(:, :, :, water_vapor_index))) !kg m-3

    ! convert rh to ppm
    water_conc(:, :, :, water_vapor_index) = rh__pc/ppm_to_rh(temperature__K, pressure__Pa)

    ! convert ppm to kg(water)/kg(air)
    water_conc(:, :, :, water_vapor_index) = water_conc(:, :, :, water_vapor_index)*1e-9*const%water_density &
                                             /air_density__kg_m3(:, :, :)

  end subroutine external_forcings

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> ppm to rh conversion factor
  pure elemental function ppm_to_rh(temperature__K, pressure__Pa) result(ppm_to_rh__1_ppm)
    real, intent(in) :: temperature__K, pressure__Pa
    real :: ppm_to_rh__1_ppm

    real, parameter :: tsteam__K = 373.15
    real :: a, water_vp__Pa

    a = 1.0 - tsteam__K/temperature__K

    a = (((-0.1299*a - 0.6445)*a - 1.976)*a + 13.3185)*a
    water_vp__Pa = 101325.0*exp(a)

    ppm_to_rh__1_ppm = pressure__Pa/water_vp__Pa/1.0e6

  end function ppm_to_rh

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Output the model results
  subroutine output_results(curr_time, camp_interface, species_conc, water_conc, water_vapor_index, pressure, temperature, rh)

    !> Current model time (min since midnight)
    real, intent(in) :: curr_time
    type(monarch_interface_t), intent(in) :: camp_interface
    integer :: z, i, j, k
    real, intent(inout) :: species_conc(:, :, :, :)
    real, intent(in)    :: water_conc(:, :, :, :)
    real, intent(in)    :: rh
    integer, intent(in)    :: water_vapor_index
    real, dimension(:, :, :), intent(in) :: pressure, temperature

    character(len=:), allocatable :: aux_str
    real, allocatable :: aux_real
    logical, save :: first_time = .true.

    !> for saving aerosol information
    character(len=:), allocatable :: aero_rep_name
    class(aero_rep_data_t), pointer :: aero_rep
    logical :: has_aero_rep
    integer :: i_section, i_bin

    has_aero_rep = .false.
    aero_rep_name = "multiphase sectional"
    has_aero_rep = camp_interface%camp_core%get_aero_rep(aero_rep_name, aero_rep)

    write (RESULTS_FILE_UNIT_TABLE, *) "Time_step:", curr_time

    do i = I_W, I_E
      do j = I_W, I_E
        do k = I_W, I_E
          write (RESULTS_FILE_UNIT_TABLE, *) "i:", i, "j:", j, "k:", k
          write (RESULTS_FILE_UNIT_TABLE, *) "Spec_name, Concentrations, Map_monarch_id"
          do z = 1, size(camp_interface%monarch_species_names)
            write (RESULTS_FILE_UNIT_TABLE, *) camp_interface%monarch_species_names(z)%string &
              , species_conc(i, j, k, camp_interface%map_monarch_id(z)) &
              , camp_interface%map_monarch_id(z)
          end do
        end do
      end do
    end do

    !print Titles
    if (first_time) then
      aux_str = "Time(min)"
      do z = 1, size(name_gas_species_to_print)
        aux_str = aux_str//" "//name_gas_species_to_print(z)%string
      end do
      aux_str = aux_str//" H2O"

      do z = 1, size(name_aerosol_species_to_print)
        aux_str = aux_str//" "//name_aerosol_species_to_print(z)%string
      end do

      write (RESULTS_FILE_UNIT, "(A)", advance="no") aux_str
      write (RESULTS_FILE_UNIT, "(A)", advance="yes") " "

      if (has_aero_rep) then
        write (RESULTS_FILE_AERO_UNIT, '(A)') "Time(min) section bin/mode reff__m number"
      end if

      write (RESULTS_FILE_ENV_UNIT, "(A)") "Time(min) temperature__K pressure__Pa rh"
      first_time = .false.
    end if

    write (RESULTS_FILE_ENV_UNIT, "(I4.4, 1X, ES13.6, 1X, ES13.6, 1X, ES13.6)") int(curr_time), temperature(1, 1, 1),&
         & pressure(1, 1, 1), rh

    write (RESULTS_FILE_UNIT, "(i4.4)", advance="no") int(curr_time)
    do z = 1, size(name_gas_species_to_print)
      write (RESULTS_FILE_UNIT, "(ES13.6)", advance="no") &
        species_conc(1, 1, 1, id_gas_species_to_print(z))
    end do
    do z = 1, size(name_aerosol_species_to_print)
      write (RESULTS_FILE_UNIT, "(ES13.6)", advance="no") &
        species_conc(1, 1, 1, id_aerosol_species_to_print(z))
    end do

    write (RESULTS_FILE_UNIT, "(ES13.6)", advance="no") &
      water_conc(1, 1, 1, water_vapor_index)

    write (RESULTS_FILE_UNIT, "(A)", advance="yes") " "

#define NUM_SECTION_ aero_rep%condensed_data_int(1)
#define NUM_INT_PROP_ 4
#define MODE_INT_PROP_LOC_(x) aero_rep%condensed_data_int(NUM_INT_PROP_+x)
#define MODE_REAL_PROP_LOC_(x) aero_rep%condensed_data_int(NUM_INT_PROP_+NUM_SECTION_+x)
! For modes, NUM_BINS_ = 1
#define NUM_BINS_(x) aero_rep%condensed_data_int(MODE_INT_PROP_LOC_(x)+1)
! Number of aerosol phases in this mode/bin set
#define NUM_PHASE_(x) aero_rep%condensed_data_int(MODE_INT_PROP_LOC_(x)+2)
! Real-time number concetration - used for modes and bins - for modes, b=1
#define NUMBER_CONC_(x,b) aero_rep%condensed_data_real(MODE_REAL_PROP_LOC_(x)+(b-1)*3+1)
! Real-time effective radius - for modes, b=1
#define EFFECTIVE_RADIUS_(x,b) aero_rep%condensed_data_real(MODE_REAL_PROP_LOC_(x)+(b-1)*3+2)
! Real-time aerosol phase mass - used for modes and bins - for modes, b=1
#define PHASE_MASS_(x,y,b) aero_rep%condensed_data_real(MODE_REAL_PROP_LOC_(x)+3*NUM_BINS_(x)+(b-1)*NUM_PHASE_(x)+y-1)
! Real-time aerosol phase average MW - used for modes and bins - for modes, b=0
#define PHASE_AVG_MW_(x,y,b) aero_rep%condensed_data_real(MODE_REAL_PROP_LOC_(x)+(3+NUM_PHASE_(x))*NUM_BINS_(x)+(b-1)*NUM_PHASE_(x)+y-1)

    if (has_aero_rep) then
      do i_section = 1, NUM_SECTION_
        do i_bin = 1, NUM_BINS_(i_section)
          write (RESULTS_FILE_AERO_UNIT, "(3(I4.4, 1X), 2(ES13.6, 1X))") &
            int(curr_time), &
            i_section, &
            i_bin, &
            EFFECTIVE_RADIUS_(i_section, i_bin), &
            NUMBER_CONC_(i_section, i_bin)
        end do
      end do
    end if

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
    open (unit=SCRIPTS_FILE_UNIT, file=file_name, status="replace", action="write")
    write (SCRIPTS_FILE_UNIT, *) "# "//file_name
    write (SCRIPTS_FILE_UNIT, *) "# Run as: gnuplot "//file_name
    write (SCRIPTS_FILE_UNIT, *) "set terminal png truecolor"
    write (SCRIPTS_FILE_UNIT, *) "set autoscale"
    write (SCRIPTS_FILE_UNIT, *) "set xrange [", start_time, ":", end_time, "]"
    do i_spec = 1, size(species_names)
      spec_name = species_names(i_spec)%string
      forall (i_char=1:len(spec_name), spec_name(i_char:i_char) .eq. '/') &
        spec_name(i_char:i_char) = '_'
      write (SCRIPTS_FILE_UNIT, *) "set output '"//file_prefix//"_"// &
        spec_name//".png'"
      write (SCRIPTS_FILE_UNIT, *) "plot\"
      write (SCRIPTS_FILE_UNIT, *) " '"//file_prefix//"_results.txt'\"
      write (SCRIPTS_FILE_UNIT, *) " using 1:"// &
        trim(to_string(tracer_ids(i_spec)))//" title '"// &
        species_names(i_spec)%string//" (MONARCH)'"
    end do
    tracer_id = END_CAMP_ID - START_CAMP_ID + 3
    write (SCRIPTS_FILE_UNIT, *) "set output '"//file_prefix//"_H2O.png'"
    write (SCRIPTS_FILE_UNIT, *) "plot\"
    write (SCRIPTS_FILE_UNIT, *) " '"//file_prefix//"_results.txt'\"
    write (SCRIPTS_FILE_UNIT, *) " using 1:"// &
      trim(to_string(tracer_id))//" title 'H2O (MONARCH)'"
    close (SCRIPTS_FILE_UNIT)

    deallocate (species_names)
    deallocate (tracer_ids)
    deallocate (file_name)
    deallocate (spec_name)

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
    integer(kind=i_kind) :: n_gas_species_plot, n_aerosol_species_plot, n_aerosol_species_start_plot &
                            , n_aerosol_species_time_plot
    character(len=100) :: n_gas_species_plot_str
    character(len=100) :: n_aerosol_species_plot_str
    character(len=100) :: n_aerosol_species_start_plot_str
    character(len=100) :: n_aerosol_species_time_plot_str

    ! Get the species names and ids
    call camp_interface%get_MONARCH_species(species_names, tracer_ids)

    ! Adjust the tracer ids to match the results file
    tracer_ids(:) = tracer_ids(:) - START_CAMP_ID + 2

    n_gas_species_plot = size(name_gas_species_to_print)
    n_gas_species_plot = n_gas_species_plot + 1
    write (n_gas_species_plot_str, *) n_gas_species_plot
    n_gas_species_plot_str = adjustl(n_gas_species_plot_str)

    n_aerosol_species_plot = size(name_aerosol_species_to_print)
    n_aerosol_species_plot = n_aerosol_species_plot + n_gas_species_plot + 1
    write (n_aerosol_species_plot_str, *) n_aerosol_species_plot
    n_aerosol_species_plot_str = adjustl(n_aerosol_species_plot_str)

    n_aerosol_species_start_plot = n_gas_species_plot + 2
    write (n_aerosol_species_start_plot_str, *) n_aerosol_species_start_plot
    n_aerosol_species_start_plot_str = adjustl(n_aerosol_species_start_plot_str)

    n_aerosol_species_time_plot = n_gas_species_plot + 1
    write (n_aerosol_species_time_plot_str, *) n_aerosol_species_time_plot
    n_aerosol_species_time_plot_str = adjustl(n_aerosol_species_time_plot_str)

    ! Create the gnuplot script
    file_name = file_prefix//".gnuplot"
    open (unit=SCRIPTS_FILE_UNIT, file=file_name, status="replace", action="write")
    write (SCRIPTS_FILE_UNIT, *) "# "//file_name
    write (SCRIPTS_FILE_UNIT, *) "# Run as: gnuplot -persist "//file_name
    write (SCRIPTS_FILE_UNIT, *) "set terminal jpeg medium size 640,480 truecolor"
    write (SCRIPTS_FILE_UNIT, *) "set title 'Mock_monarch_cb05_soa'"
    write (SCRIPTS_FILE_UNIT, *) "set xlabel 'Time (min)'"
    write (SCRIPTS_FILE_UNIT, *) "set ylabel 'Gas concentration (ppmv)'"
    write (SCRIPTS_FILE_UNIT, *) "set y2label 'Aerosol concentration (kg/m^3)'"
    write (SCRIPTS_FILE_UNIT, *) "set ytics nomirror"
    write (SCRIPTS_FILE_UNIT, *) "set y2tics nomirror"

    write (SCRIPTS_FILE_UNIT, *) "set logscale y"
    write (SCRIPTS_FILE_UNIT, *) "set logscale y2"
    write (SCRIPTS_FILE_UNIT, *) "set xrange [", start_time, ":", end_time, "]"

    i_spec = 1
    spec_name = species_names(i_spec)%string
    forall (i_char=1:len(spec_name), spec_name(i_char:i_char) .eq. '/') &
      spec_name(i_char:i_char) = '_'

    write (SCRIPTS_FILE_UNIT, *) "set key top left"
    write (SCRIPTS_FILE_UNIT, "(A)", advance="no") "set output 'out/monarch_plot.jpg'"
    write (SCRIPTS_FILE_UNIT, *)
    write (SCRIPTS_FILE_UNIT, "(A)", advance="no") "plot for [col=2:" &
    //trim(n_gas_species_plot_str)//"] "//file_prefix//"_results.txt' &
&    using 1:col axis x1y1 title columnheader, for [col2=" &
    //trim(n_aerosol_species_start_plot_str)//":" &
    //trim(n_aerosol_species_plot_str)//"] &
&    '"//file_prefix//"_results.txt' &
&    using " &
    //trim(n_aerosol_species_time_plot_str)// &
    ":col2 axis x1y2 title columnheader"

    close (SCRIPTS_FILE_UNIT)

    deallocate (species_names)
    deallocate (tracer_ids)
    deallocate (file_name)
    deallocate (spec_name)

  end subroutine create_gnuplot_persist

end program mock_monarch
