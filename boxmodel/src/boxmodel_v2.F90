! Copyright (C) 2023 Barcelona Supercomputing Center and University of
! Illinois at Urbana-Champaign
! SPDX-License-Identifier: MIT

!> \file
!> The boxmodel_v2 program

!> Mock version of the MONARCH model for testing integration with PartMC
program boxmodel_v2
  use mpi, only: MPI_COMM_WORLD

  use camp_util, only: assert_msg, almost_equal, &
    to_string, open_file_write, close_file, string_t
  use camp_boxmodel_interface
  use camp_mpi
  use camp_aero_rep_data, only: aero_rep_data_t
  use camp_constants
  use boxmodel_constraints
  use boxmodel_time_control
  use boxmodel_io, only: ncdf_writer
  use boxmodel_update_netcdf
  use boxmodel_photolysis, only: zenith
  use boxmodel_log


  implicit none

  !> File unit for model run-time messages
  integer :: OUTPUT_FILE_UNIT
  !> model results netcdf writer
  type(ncdf_writer)  :: output_writer


  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Parameters for mock MONARCH model !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Number of cells to compute simultaneously
  integer :: n_cells
  !> Check multiple cells results are correct?
  logical :: check_multiple_cells = .false.

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! State variables for mock MONARCH model !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> NMMB style arrays (W->E, S->N, top->bottom, ...)
  !> Temperature (K)
  real(kind=dp), allocatable :: temperature(:)
  !> Species conc (various units)
  real(kind=dp), allocatable :: species_conc(:, :)
  !> Water concentrations (kg_H2O/kg_air)
  real(kind=dp), allocatable :: water_conc(:)
  !> Air density (kg_air/m^3)
  real(kind=dp), allocatable :: air_density(:)
  !> Air pressure (Pa)
  real(kind=dp), allocatable :: pressure(:)
  !> Cell height (cm)
  real(kind=dp), allocatable :: height(:)
  !> cell altitude (m)
  real(kind=dp), allocatable :: altitude(:)
  !> cell latitude and longitude (degrees)
  real(kind=dp), allocatable :: latitude(:), longitude(:)
  !> relative humidity
  real(kind=dp), allocatable :: rh(:)
  !> solar zenith angle
  real(kind=dp), allocatable :: sza(:)

  !> Emissions hour counter
  integer :: i_hour = 0

  !> !!! Add to boxmodel variables !!!
  type(boxmodel_interface_t), pointer :: camp_interface

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Mock model setup and evaluation variables !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> CAMP-chem input file file
  character(len=:), allocatable :: camp_input_file
  !> camp <-> boxmodel interface configuration file
  character(len=:), allocatable :: interface_input_file
  !> Results file prefix
  character(len=:), allocatable :: output_file_prefix

  ! MPI
#ifdef CAMP_USE_MPI
  integer(kind=i_kind) :: mpi_size, mpi_rank, local_comm
#endif

  character(len=500) :: arg
  integer :: status_code, i_time, i_spec, i, j, k, z, i_cell
  logical :: found

  ! Computation time variable
  real(kind=dp) :: timing_start, timing_end

  ! Check the command line arguments
  call assert_msg(129432506, command_argument_count() .eq. 3, "Usage: "// &
    "./boxmodel_v2 camp_input_file_list.json "// &
    "interface_input_file.json output_file_prefix")

  ! initialize mpi
  call camp_mpi_init()
  call init_log()

  local_comm = MPI_COMM_WORLD

  mpi_size = camp_mpi_size()
  call thread_log%info("n mpi processes="//to_string(mpi_size))
  mpi_rank = camp_mpi_rank()

  call cpu_time(timing_start)

  ! Initialize camp
  call get_command_argument(1, arg, status=status_code)
  call assert_msg(678165802, status_code .eq. 0, "Error getting camp "// &
    "configuration file name")
  camp_input_file = trim(arg)
  call thread_log%info("camp config file: "//camp_input_file)
  call get_command_argument(2, arg, status=status_code)
  call assert_msg(664104564, status_code .eq. 0, "Error getting camp "// &
    "<-> boxmodel interface configuration file name")
  interface_input_file = trim(arg)
  call thread_log%info("interface file: "//interface_input_file)

  ! Initialize the mock model
  call get_command_argument(3, arg, status=status_code)
  call assert_msg(234156729, status_code .eq. 0, "Error getting output file prefix")
  output_file_prefix = trim(arg)
  call thread_log%info("outputting results to: "//output_file_prefix)

  call thread_log%debug("camp_interface starting initialization")
  camp_interface => boxmodel_interface_t(camp_input_file, interface_input_file, &
    n_cells, local_comm)
    
  n_cells = camp_interface%n_cells
  call thread_log%debug("camp_interface initialized")
  call thread_log%info("n_cells="//to_string(n_cells))

  ! allocate species concentrations and environmental variable arrays
  allocate (species_conc(n_cells, size(camp_interface%camp_state%state_var)))
  allocate (temperature(n_cells))
  allocate (water_conc(n_cells))
  allocate (air_density(n_cells))
  allocate (pressure(n_cells))
  allocate (height(n_cells))
  allocate (altitude(n_cells))
  allocate (latitude(n_cells))
  allocate (longitude(n_cells))
  allocate (rh(n_cells))
  allocate (sza(n_cells))

  ! initialize external forcings (T, P, RH, density, SZA, pblh)
  call external_forcings(camp_interface%time_control, &
    air_density, pressure, temperature, water_conc, rh, sza, height)
  species_conc(:, camp_interface%gas_phase_water_id) = water_conc

  ! TODO: different initialization for each i_cell
  i_cell = 1
  ! initialize photolysis
  if (camp_interface%photolysis_model /= "NONE") then
    call camp_interface%photolysis_map%init(camp_interface%photolysis_model,&
      "atmosphere_profile.dat", &
      camp_interface%time_control, 5.0_dp, 0.0_dp, &
      real(temperature(i_cell), kind=dp), &
      real(rh(i_cell)/100., kind=dp), &
      real(pressure(i_cell)/100., kind=dp), &
      1, &
      real(altitude(i_cell), kind=dp), &
      longitude(i_cell), latitude(i_cell), &
      0.1_dp, 300._dp, 0._dp, 0._dp, &
      0._dp, 4._dp, 5._dp, &
      0.235_dp, 0.99_dp, 1.0_dp, &
      real(air_density(i_cell), kind=dp))   ! clear sky
  end if

  call model_initialize(output_file_prefix, local_comm)
  ! update initial photolysis rates
  if (camp_interface%photolysis_model /= "NONE") then
    call camp_interface%update_photorates(sza)
  end if
  ! update emission rate
  call camp_interface%update_emission_rates(height, pressure, temperature)

  ! update microphysics
  call camp_interface%update_microphysics()

  ! Set initial concentrations
  call camp_interface%set_init_conc(species_conc, water_conc)

  ! output initial conditions
  call output_results(camp_interface%time_control)

  call cpu_time(timing_end)

  call thread_log%info("Initialization time: "//trim(to_string(timing_end - timing_start))//" s")

  call cpu_time(timing_start)
  ! Run the model
  do
    ! vary environmental conditions
    call external_forcings(camp_interface%time_control, &
      air_density, pressure, temperature, water_conc, rh, sza, height)
    species_conc(:, camp_interface%gas_phase_water_id) = water_conc

    call camp_interface%integrate( &
      camp_interface%time_control%get_current_time(), & ! Starting time (sec)
      camp_interface%time_control%get_time_step(), & ! Time step (sec)
      temperature, & ! Temperature (K)
      species_conc, & ! Tracer array
      water_conc, & ! Water concentrations (kg_H2O/kg_air)
      air_density, & ! Air density (kg_air/m^3)
      pressure, & ! Air pressure (Pa)
      sza, & ! solar zenith angle (deg)
      height, & ! box height (cm)
      i_hour)

    if (.not. camp_interface%time_control%increment()) exit
    ! #ifdef CAMP_USE_MPI
    !     if (camp_mpi_rank() .eq. 0) then
    ! #endif
    ! at the moment, all processes write in separate files

    if (camp_interface%time_control%is_output_time()) then
      call thread_log%debug("writing timestep")
      call output_results(camp_interface%time_control)
    endif
    ! #ifdef CAMP_USE_MPI
    !     end if
    ! #endif
  end do

  call cpu_time(timing_end)

  call thread_log%info("Number of timesteps: "//to_string(camp_interface%time_control%output_time_step_index))

  call thread_log%info("Computation time: "//to_string(timing_end - timing_start)//" s")
  call thread_log%info("Computation time per timestep: "//to_string((timing_end - timing_start)/ &
    camp_interface%time_control%output_time_step_index)//" s")

  call output_writer%close()

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
  subroutine model_initialize(file_prefix, mpi_comm)

    !> File prefix for model results
    character(len=:), allocatable, intent(in) :: file_prefix
    integer(kind=i_kind), intent(in) :: mpi_comm

    integer :: i_spec
    character(len=:), allocatable :: file_name

    ! Open the output file
    file_name = file_prefix//"_results"

    species_conc(:, :) = 0.0

    call external_forcings(camp_interface%time_control, &
      air_density, pressure, temperature, water_conc, rh, sza, height)
    species_conc(:, camp_interface%gas_phase_water_id) = water_conc

    call output_writer%init(file_name, &
      camp_interface%camp_core, &
      n_cells, &
      camp_interface%time_control, &
      camp_interface%camp_core%state_size_per_cell(), &
      mpi_comm)

    deallocate (file_name)

  end subroutine model_initialize

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Update environmental conditions
  subroutine external_forcings(time_control, air_density__kg_m3, pressure__Pa, &
    temperature__K, water_conc, rh, sza, height__cm)
    type(time_control_t), intent(in) :: time_control
    real(kind=dp), dimension(n_cells), intent(out) :: temperature__K, pressure__Pa
    real(kind=dp), dimension(n_cells), intent(out) :: air_density__kg_m3
    real(kind=dp), dimension(n_cells), intent(out) :: water_conc! kg(water)/kg(air)
    real(kind=dp), dimension(n_cells), intent(out) :: rh
    real(kind=dp), dimension(n_cells), intent(out) :: sza
    real(kind=dp), dimension(n_cells), intent(out) :: height__cm
    real(kind=dp), dimension(n_cells) :: water_vapor_pressure
    real(kind=dp)   :: curr_time__s

    real(kind=dp) :: rh__pc(n_cells), water__ppm(n_cells)

    curr_time__s = time_control%get_current_time()


    latitude(:) = 41.5
    longitude(:) = 2.1
    altitude(:) = 1000
    pressure__Pa(:) = 101325
    temperature__K(:) = 298
    height__cm(:) = 100000
    rh(:) = 60.0
    rh__pc(:) = rh(:)
    ! TODO: remove debug values and reinstate constraint calls
    do i_cell = 1, n_cells

      ! update lon and lat (degrees)
      latitude(i_cell) = camp_interface%latitude_constraint(i_cell)%val%get(curr_time__s)
      select case (camp_interface%latitude_constraint(i_cell)%val%value_unit)
      case ("deg", "degree", "degrees", "°") ! do nothing
      case default
        call die_msg(347091569, &
          "Error in latitude constraint, unknown unit: "// &
          camp_interface%latitude_constraint(i_cell)%val%value_unit)
      end select

      longitude(i_cell) = camp_interface%longitude_constraint(i_cell)%val%get(curr_time__s)
      select case (camp_interface%longitude_constraint(i_cell)%val%value_unit)
      case ("deg", "degree", "degrees", "°") ! do nothing
      case default
        call die_msg(347091569, &
          "Error in longitude constraint, unknown unit: "// &
          camp_interface%longitude_constraint(i_cell)%val%value_unit)
      end select

      ! update altitude (m)
      altitude(i_cell) = camp_interface%altitude_constraint(i_cell)%val%get(curr_time__s)
      select case (camp_interface%altitude_constraint(i_cell)%val%value_unit)
      case ("m", "meter", "meters") ! do nothing
      case ("cm", "centimeter", "centimeters")
        altitude(i_cell) = altitude(i_cell)/100.0
      case ("km", "kilometer", "kilometers")
        altitude(i_cell) = altitude(i_cell)*1000.0
      case default
        call die_msg(864305898, &
          "Error in altitude constraint, unknown unit: "// &
          camp_interface%altitude_constraint(i_cell)%val%value_unit)
      end select

      ! update pressure, temperature and rh
      pressure__Pa(i_cell) = camp_interface%pressure_constraint(i_cell)%val%get(curr_time__s)

      ! convert to Pa
      select case (camp_interface%pressure_constraint(i_cell)%val%value_unit)
      case ("Pa") ! do nothing
      case ("hPa", "mbar")
        pressure__Pa(i_cell) = pressure__Pa(i_cell)*100.
      case ("atm")
        pressure__Pa(i_cell) = pressure__Pa(i_cell)*101325.0
      case default
        call die_msg(800766944, &
          "Error in pressure constraint, unknown unit: "// &
          camp_interface%pressure_constraint(i_cell)%val%value_unit)
      end select

      temperature__K(i_cell) = camp_interface%temperature_constraint(i_cell)%val%get(curr_time__s)
      ! convert to K
      select case (camp_interface%temperature_constraint(i_cell)%val%value_unit)
      case ("K") ! do nothing
      case ("C")
        temperature__K(i_cell) = temperature__K(i_cell) + 273.15
      case ("F")
        temperature__K(i_cell) = (temperature__K(i_cell) - 32.0)/1.8 + 273.15
      case default
        call die_msg(785849517, &
          "Error in temperature constraint, unknown unit: "// &
          camp_interface%temperature_constraint(i_cell)%val%value_unit)
      end select

      height__cm(i_cell) = camp_interface%height_constraint(i_cell)%val%get(curr_time__s)
      ! convert to cm
      select case (camp_interface%height_constraint(i_cell)%val%value_unit)
      case ("cm") ! do nothing
      case ("m")
        height__cm(i_cell) = height__cm(i_cell)*100.
      case ("km")
        height__cm(i_cell) = height__cm(i_cell) + 100000.
      case default
        call die_msg(192286936, &
          "Error in height constraint, unknown unit: "// &
          camp_interface%height_constraint(i_cell)%val%value_unit)
      end select

      rh__pc(i_cell) = camp_interface%humidity_constraint(i_cell)%val%get(curr_time__s) ! in pc
      ! convert to %
      select case (camp_interface%humidity_constraint(i_cell)%val%value_unit)
      case ("pc", "%") ! do nothing
      case ("dec", "decimal")
        rh__pc(i_cell) = rh__pc(i_cell)*100.
      case default
        call die_msg(386052629, &
          "Error in humidity constraint, unknown unit: "// &
          camp_interface%humidity_constraint(i_cell)%val%value_unit)
      end select
      rh(i_cell) = rh__pc(i_cell)

      if (camp_interface%constrained_sza) then
        sza(i_cell) = camp_interface%sza_constraint(i_cell)%val%get(curr_time__s)
        select case (camp_interface%sza_constraint(i_cell)%val%value_unit)
        case ("deg", "dg", "d", "º")
          ! do nothing
        case default
          call die_msg(386052630, &
            "Error in sza constraint, unknown unit: "// &
            camp_interface%sza_constraint(i_cell)%val%value_unit)
        end select
      else
        ! calculate sza
        sza(i_cell) = zenith(latitude(i_cell), longitude(i_cell), time_control%year, &
          time_control%month, time_control%day, time_control%hour)
      end if
    end do

    ! update air density
    water_vapor_pressure(:) = rh__pc(:)*water_saturation_pvap__Pa(temperature__K(:))/100.
    air_density__kg_m3(:) = (pressure__Pa(:) - water_vapor_pressure(:))/(287.058*temperature__K(:)) + &
      water_vapor_pressure(:)/(461.495*temperature__K(:))

    ! convert rh to ppm
    water_conc(:) = rh__pc/ppm_to_rh(temperature__K, pressure__Pa)

    ! convert ppm to kg(water)/kg(air)
    water_conc(:) = water_conc(:)*1e-9*const%water_density &
      /air_density__kg_m3(:)

  end subroutine external_forcings

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> ppm to rh conversion factor
  pure elemental function ppm_to_rh(temperature__K, pressure__Pa) result(ppm_to_rh__1_ppm)
    real(kind=dp), intent(in) :: temperature__K, pressure__Pa
    real(kind=dp) :: ppm_to_rh__1_ppm

    ppm_to_rh__1_ppm = pressure__Pa/water_saturation_pvap__Pa(temperature__K)/1.0e6

  end function ppm_to_rh

  !> Return the saturation vapor pressure of water in Pa
  !! using the formula from Seinfeld and Pandis (2006)
  elemental real function water_saturation_pvap__Pa(temperature__K)
    real(kind=dp), intent(in) :: temperature__K
    real(kind=dp), parameter :: tsteam__K = 373.15
    real(kind=dp) :: a

    a = 1.0 - tsteam__K/temperature__K
    a = (((-0.1299*a - 0.6445)*a - 1.976)*a + 13.3185)*a

    water_saturation_pvap__Pa = 101325.0*exp(a)

  end function water_saturation_pvap__Pa

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!> Output the model results
  subroutine output_results(time_control)
    class(time_control_t), intent(in)  :: time_control
    integer(kind=i_kind) :: i_cell



    call put_environment_variables(output_writer,  &
      latitude, &
      longitude, &
      altitude, &
      pressure, &
      temperature, &
      rh, &
      time_control%get_current_time(), &
      time_control%output_time_step_index, &
      sza, &
      height)

    call put_species_concentrations( output_writer, &
      camp_interface%camp_state, &
      time_control%output_time_step_index)

    do i_cell = 1, camp_interface%n_cells
      call put_emission_rates( output_writer, &
        camp_interface%emissions_map(i_cell), &
        i_cell, &
        time_control%output_time_step_index)
    end do

    ! \todo implement aerosol size distribution output

    !call output_writer%sync()

  end subroutine output_results

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end program boxmodel_v2
