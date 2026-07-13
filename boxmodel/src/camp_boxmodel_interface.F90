! Copyright (C) 2021 Barcelona Supercomputing Center and University of
! Illinois at Urbana-Champaign
! SPDX-License-Identifier: MIT

!> \file
!> The monarch_interface_t object and related functions

!> Interface for the MONACH model and PartMC-camp
module camp_boxmodel_interface
  use boxmodel_constraints, only: constraint_ptr
  use boxmodel_constant_constraint, only: constant_constraint_t
  use boxmodel_file_constraint, only: file_constraint_t
  use boxmodel_monarch_constraint, only: monarch_constraint_from_trajectory
  use boxmodel_constraints_utils, only: constraint_from_type_id
  use boxmodel_time_control
  use boxmodel_photolysis, only: photolysis_map_t
  use boxmodel_emissions, only: emissions_map_t
  use boxmodel_deposition, only: deposition_map_t
  use boxmodel_sum_species, only: sum_species_t
  use boxmodel_microphysics, only: microphysics_map_t
  use boxmodel_montecarlo_gaussian, only: montecarlo_gaussian_t
  use boxmodel_forced_species, only: forced_species_map_t

  use boxmodel_montecarlo, only: N_SOBOL_MONTECARLO, RANDOM, SOBOL
  use boxmodel_sobol_numbers, only: generate_sobol_numbers, broadcast_sobol_numbers

  use boxmodel_log

  use camp_constants, only: i_kind, const
  use camp_mpi
  use camp_util, only: assert_msg, string_t, &
                       warn_assert_msg, die_msg
  use camp_rand, only: camp_srand
  use camp_camp_core
  use camp_camp_state
  use camp_aero_rep_data
  use camp_aero_rep_factory
  use camp_aero_rep_modal_binned_mass
  use camp_chem_spec_data
  use camp_property
  use camp_camp_solver_data
  use camp_mechanism_data, only: mechanism_data_t
  use camp_rxn_data, only: rxn_data_t, GAS_AERO_RXN, GAS_RXN
  use camp_rxn_photolysis
  use camp_rxn_emission, only: rxn_emission_t
  use camp_rxn_first_order_loss, only: rxn_first_order_loss_t
  use camp_solver_stats
#ifdef CAMP_USE_MPI
  use mpi
#endif
#ifdef CAMP_USE_JSON
  use json_module
#endif

  implicit none
  private

  public :: boxmodel_interface_t

  integer(kind=i_kind), parameter, public :: SINGLE_BOX = 0, MONTE_CARLO = 1

  !> CAMP <-> boxmodel interface
  !!
  !! Contains all data required to intialize and run camp from boxmodel data
  !! and map state variables between camp and boxmodel
  type :: boxmodel_interface_t
    !private
    !> CAMP-chem core
    type(camp_core_t), pointer :: camp_core
    !> CAMP-chem state
    type(camp_state_t), pointer :: camp_state
    !> PartMC-camp ids for initial concentrations
    integer(kind=i_kind), allocatable :: init_conc_camp_id(:)
    !> Initial species concentrations
    type(constant_constraint_t), allocatable :: init_conc(:)
    !> Number of cells to compute simultaneously with one process
    integer(kind=i_kind) :: n_cells
    !> type of parallel simulation to run
    integer(kind=i_kind) :: multiple_boxes_type
    !> percentage of the cells to run on the GPUs
    integer(kind=i_kind) :: load_gpu
    !> automatically balance the load between CPUs and GPUs
    integer(kind=i_kind) :: is_load_balance
    !> type of random number generation
    integer(kind=i_kind) :: random_type = RANDOM
    !> sobol offset
    integer(kind=i_kind) :: sobol_offset = 0
    !> species map data for camp-boxmodel
    type(property_t), pointer :: species_map_data
    !> Gas-phase water id in camp
    integer(kind=i_kind) :: gas_phase_water_id
    !> Gas-phase water dimer id in camp
    integer(kind=i_kind) :: gas_phase_water_dimer_id
    !> Interface input data
    type(property_t), pointer :: property_set
    !> Initial concentration data
    type(property_t), pointer :: init_conc_data
    !> Mechanism name
    character(len=:), allocatable :: mechanism_name
    !> Flag to activate aerosol chemistry (condensation and inorganic chemistry)
    !> default is true
    logical :: aerosol_flag
    !> pressure, temperature, humidity constraint
    type(constraint_ptr), allocatable, dimension(:) :: pressure_constraint, temperature_constraint, humidity_constraint
    !> pbl height constraint
    type(constraint_ptr), allocatable, dimension(:) :: height_constraint
    !> geographical constraints
    type(constraint_ptr), allocatable, dimension(:) :: latitude_constraint, longitude_constraint, altitude_constraint
    !> optional sza constraint
    type(constraint_ptr), allocatable, dimension(:) :: sza_constraint
    logical :: constrained_sza
    !> Time control
    type(time_control_t), pointer :: time_control
    !> data for initializing emissions
    type(property_t), pointer :: emissions_data
    !> emissions map, allocate one for each cell
    type(emissions_map_t), allocatable, dimension(:) :: emissions_map
    !> deposition
    type(deposition_map_t) :: deposition_map
    !> data for initializing constrained species
    type(property_t), pointer :: forced_species_data
    !> species constraints map
    type(forced_species_map_t) :: forced_species_map
    !>
    type(sum_species_t), dimension(:), allocatable :: sum_species
    !> update data for modal aerosol distributions
    type(property_t), pointer :: microphysics_data
    type(microphysics_map_t) :: microphysics_map
    !>
    type(photolysis_map_t) :: photolysis_map
    character(len=:), allocatable :: photolysis_model
    !> Solve multiple grid cells at once?
    logical :: solve_multiple_cells = .true.
    type(solver_stats_t), pointer :: solver_stats

  contains
    !> Integrate CAMP for the current boxmodel state over a specified time step
    procedure :: integrate
    !> set initial concentrations
    procedure :: set_init_conc
    !> update photolysis reaction rates
    procedure :: update_photorates
    !> update emissions reaction rates
    procedure :: update_emission_rates
    !> update depisition rates
    procedure :: update_deposition_rates
    !> update microphysics
    procedure :: update_microphysics
    !> update forced species
    procedure :: update_forced_species
    !> Print the boxmodel-camp data
    procedure :: print => do_print
    !> Load information about parallelization from interface file
    procedure, private :: load_parallel_config
    !> Load interface data from a set of input files
    procedure, private :: load
    !> Create the CAMP <-> boxmodel species map
    procedure, private :: create_map
    !> Load the initial concentrations
    procedure, private :: load_init_conc
    !> initialize cloud_j map
    procedure, private :: init_photolysis_reactions
    !> initialize deposition map
    procedure, private :: init_deposition_reactions
    !> check that trajectory constraining elements are initialized
    procedure, private :: check_traj_initialization
    !> load emissions constraints
    procedure, private :: load_emissions_constraints
    !> initialize emissions
    procedure, private :: load_emission_data
    !> load microphysics constraints
    procedure, private :: load_microphysics_constraints
    !> load forced species constraints
    procedure, private :: load_forced_species_constraints
    !> load different types of constraints
    procedure, private :: load_constraints
    !> load sum species
    procedure, private :: load_sum_species
    !> update sum species concentrations
    procedure, private :: update_sum_species_concentrations
    !> Determine the number of bytes required to pack the information to send to other processes
    procedure, private :: pack_size
    !> Pack the information to send to other processes
    procedure, private :: bin_pack
    !> Unpack the information received from the main processes
    procedure, private :: bin_unpack
    !> Finalize the interface
    final :: finalize
  end type boxmodel_interface_t

  !> CAMP <-> boxmodel interface constructor
  interface boxmodel_interface_t
    procedure :: constructor
  end interface boxmodel_interface_t

  integer(kind=i_kind)  :: BOXMOD_PROCESS

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Create and initialize a new boxmodel_interface_t object
  !!
  !! Create a boxmodel_interface_t object at the beginning of the  model run
  !! for each node. The master node should pass a string containing the path
  !! to the CAMP confirguration file list, the path to the interface
  !! configuration file.
  function constructor(camp_config_file, interface_config_file, &
                       n_cells, mpi_comm) result(this)

    !> A new MONARCH interface
    type(boxmodel_interface_t), pointer :: this
    !> Path to the camp configuration file list
    character(len=:), allocatable, optional :: camp_config_file
    !> Path to the camp <-> boxmodel interface input file
    character(len=:), allocatable, optional :: interface_config_file
    !> MPI communicator
    integer, intent(in), optional :: mpi_comm
    !> Num cells to compute simulatenously
    integer, optional :: n_cells

    type(camp_solver_data_t), pointer :: camp_solver_data
    character, allocatable :: buffer(:)
    integer(kind=i_kind) :: pos
    integer :: pack_size, ierr
    integer(kind=i_kind) :: i_spec, i_photo_rxn, i, i_cell

    class(aero_rep_data_t), pointer :: aero_rep
    integer(kind=i_kind) :: i_sect_om_mode1, i_sect_om_mode2, i_sect_om_mode3, &
                            i_sect_bc, i_sect_sulf, i_sect_opm
    type(aero_rep_factory_t) :: aero_rep_factory

    ! Computation time variable
    real(kind=dp) :: comp_start, comp_end

#ifdef CAMP_USE_MPI
    integer :: local_comm

    if (present(mpi_comm)) then
      local_comm = mpi_comm
    else
      local_comm = MPI_COMM_WORLD
    end if
#endif
    pack_size = 0
    ! Set the MPI rank
    BOXMOD_PROCESS = camp_mpi_rank()

    call camp_srand(0, offset=BOXMOD_PROCESS*739859)

    ! Create a new interface object
    call thread_log%debug("allocating the interface")
    allocate (this)

    !TODO: set this%solve_multiple_cells and this%n_cells in this function
    ! and remove commented code above
    call this%load_parallel_config(interface_config_file, BOXMOD_PROCESS)

    call thread_log%debug("solve_multiple_cells:"//to_string(this%solve_multiple_cells))
    call thread_log%debug("n_cells="//to_string(this%n_cells))

    ! allocate constrating for each cell
    !TODO   ensure all constraints are created for all cells
    allocate (this%emissions_map(this%n_cells))

    ! Check for an available solver
    camp_solver_data => camp_solver_data_t()
    call assert_msg(332298164, camp_solver_data%is_solver_available(), &
                    "No solver available")
    deallocate (camp_solver_data)
    call thread_log%debug("found solver")
    ! Initialize the time-invariant model data on each node
    if (BOXMOD_PROCESS .eq. 0) then

      ! Start the computation timer on the primary node
      call cpu_time(comp_start)
      ! each process reads the interface config file and sets up its own initial conditions
      call assert_msg(194027509, present(interface_config_file), &
                      "Missing CAMP <-> Boxmodel interface configuration file")

      ! Load the interface data
      call this%load(interface_config_file, BOXMOD_PROCESS)
      call thread_log%debug("interface_config file loaded")

      call assert_msg(304676624, present(camp_config_file), &
                      "Missing CAMP configuration file list")

      ! Initialize the camp-chem core
      this%camp_core => camp_core_t(camp_config_file, this%n_cells)

      call thread_log%debug("config_file loaded and camp_core initialized")
      call this%camp_core%initialize()

      ! create species map
      call this%create_map()

      call this%init_photolysis_reactions()

      call this%init_deposition_reactions()

      call thread_log%info("loading the initial concentrations")
      call this%load_init_conc()

      call thread_log%info("loading the sum species")
      call this%load_sum_species()

      call thread_log%info("loading emissions")
      call this%load_emission_data()

      call thread_log%info("loading microphysics constraints")
      call this%load_microphysics_constraints()

      call thread_log%info("loading the forced species")
      call this%load_forced_species_constraints()

      ! after all constraints are loaded, request sobol numbers
      if (N_SOBOL_MONTECARLO > 0) then
        call thread_log%info("generating requested sobol numbers")
        call generate_sobol_numbers(N_SOBOL_MONTECARLO, camp_mpi_size(local_comm), this%sobol_offset, .true.)
      end if

#ifdef CAMP_USE_MPI

      call thread_log%info("packing data to be sent to other processes")
      pack_size = this%pack_size(comm=local_comm)
      allocate (buffer(pack_size))
      pos = 0
      call this%bin_pack(buffer, pos, comm=local_comm)
    end if

    ! broadcast the buffer size
    call camp_mpi_bcast_integer(pack_size, comm=local_comm)

    if (BOXMOD_PROCESS .ne. 0) then
      call thread_log%info("receiving packed data")
      ! allocate the buffer to receive data
      allocate (buffer(pack_size))
    end if
    ! broadcast the buffer
    call camp_mpi_bcast_packed(buffer, comm=local_comm)

    if (BOXMOD_PROCESS .ne. 0) then
      ! unpack the data
      call thread_log%info("unpacking data")
      pos = 0
      call this%bin_unpack(buffer, pos, comm=local_comm)
      call thread_log%debug("done unpacking data")
      call thread_log%debug("deallocating buffer")
      deallocate (buffer)
      call thread_log%debug("done deallocating buffer")
#endif
    end if

#ifdef CAMP_USE_MPI
    ! broadcast sobol numbers to each process
    call camp_mpi_bcast_integer(N_SOBOL_MONTECARLO, local_comm)
    call thread_log%debug("n_sobol_montecarlo="//to_string(N_SOBOL_MONTECARLO))
    if (N_SOBOL_MONTECARLO > 0) then
      call thread_log%info("broadcasting sobol numbers")
      call broadcast_sobol_numbers(local_comm)
      ! randomize all constraints for all processes
      do i = 1, size(this%init_conc)
        call this%init_conc(i)%randomize()
      end do

      do i_cell = 1, this%n_cells
        call this%longitude_constraint(i_cell)%val%randomize()
        call this%latitude_constraint(i_cell)%val%randomize()
        call this%altitude_constraint(i_cell)%val%randomize()
        call this%pressure_constraint(i_cell)%val%randomize()
        call this%temperature_constraint(i_cell)%val%randomize()
        call this%humidity_constraint(i_cell)%val%randomize()
        call this%height_constraint(i_cell)%val%randomize()
        if (this%constrained_sza) then
          call this%sza_constraint(i_cell)%val%randomize()
        end if

        do i = 1, this%emissions_map(i_cell)%n_emission
          call this%emissions_map(i_cell)%emissions_constraints(i)%val%randomize()
        end do

      end do

      call thread_log%info("done broadcasting sobol numbers")
    end if
#endif

    ! Initialize the solver on all nodes
    call thread_log%debug("initializing solver")
    call thread_log%debug("load_gpu="//to_string(this%load_gpu))
    call thread_log%debug("is_load_balance="//to_string(this%is_load_balance))
    ! TODO: move to user input

    call this%camp_core%solver_initialize(this%load_gpu, this%is_load_balance)
    call thread_log%debug("done initializing solver")

    ! Create a state variable on each node
    this%camp_state => this%camp_core%new_state()

    this%solver_stats => solver_stats_t()

    ! Calculate the intialization time
    if (BOXMOD_PROCESS .eq. 0) then
      call cpu_time(comp_end)
      call thread_log%info("Initialization time: "//to_string(comp_end - comp_start)//" s")
    end if

  end function constructor

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Integrate the CAMP mechanism for a particular set of cells and timestep
  subroutine integrate(this, start_time, time_step, temperature, boxmod_conc, water_conc, &
                       air_density, pressure, sza, height, i_hour)

    !> PartMC-camp <-> MONARCH interface
    class(boxmodel_interface_t) :: this
    !> Integration start time (s)
    real(kind=dp), intent(in) :: start_time
    !> Integration time step [s]
    real(kind=dp), intent(in) :: time_step

    !> NMMB style arrays (W->E, S->N, top->bottom, ...)
    !> Temperature (K)
    real(kind=dp), intent(in) :: temperature(this%n_cells)
    !> boxmod species concentration (ppm or ug/m^3)
    real(kind=dp), intent(inout) :: boxmod_conc(:, :)
    !> Atmospheric water concentrations (kg_H2O/kg_air)
    real(kind=dp), intent(in) :: water_conc(this%n_cells)
    !> Air density (kg_air/m^3)
    real(kind=dp), intent(in) :: air_density(this%n_cells)
    !> Pressure (Pa)
    real(kind=dp), intent(in) :: pressure(this%n_cells)
    !> solar zenith angle (deg)
    real(kind=dp), intent(in) :: sza(this%n_cells)
    !> box height (cm)
    real(kind=dp), intent(in) :: height(this%n_cells)
    integer, intent(inout) :: i_hour
    !> water dimerization equilibrium constant
    real(kind=dp) :: Kd

    type(chem_spec_data_t), pointer :: chem_spec_data

    ! MPI
    character, allocatable :: buffer(:)
    integer(kind=i_kind) :: pos, pack_size
    integer :: local_comm
    real(kind=dp), allocatable :: mpi_conc(:)

    integer :: i, j, k, k_flip, i_spec, z, o, i2, i_cell, i_photo_rxn
    integer :: k_end

    ! Computation time variables
    real(kind=dp) :: comp_start, comp_end

    integer :: state_size_per_cell, n_cell_check

    if (this%n_cells .eq. 1) then
      state_size_per_cell = 0
    else
      state_size_per_cell = this%camp_core%state_size_per_cell()
    end if

    call assert_msg(731700229, &
                    this%camp_core%get_chem_spec_data(chem_spec_data), &
                    "No chemical species data in camp_core.")

    call cpu_time(comp_start)

    if (.not. this%solve_multiple_cells) then
      call thread_log%debug("560220905 in CPU integration")
      do i_cell = 1, this%n_cells
        ! Update the environmental state
        call thread_log%debug("updating temperature: "//to_string(temperature(i_cell)))
        call this%camp_state%env_states(1)%set_temperature_K(temperature(i_cell))

        call thread_log%debug("updating pressure: "//to_string(pressure(i_cell)))
        call this%camp_state%env_states(1)%set_pressure_Pa(pressure(i_cell))

        if (this%time_control%last_photolysis_update > this%time_control%photolysis_update_timestep) then
          call thread_log%debug("updating photolysis rates for sza="//to_string(sza(i_cell)))
          this%time_control%last_photolysis_update = 0.0
          call this%update_photorates(sza(i_cell))
        end if

        if (this%time_control%last_emissions_update > this%time_control%emissions_update_timestep) then
          call thread_log%debug("updating emission rates")
          this%time_control%last_emissions_update = 0.0
          call this%update_emission_rates(height, pressure(i_cell), temperature(i_cell))
        end if

        if (this%time_control%last_microphysics_update > this%time_control%microphysics_update_timestep) then
          call thread_log%debug("updating microphysics")
          this%time_control%last_microphysics_update = 0.0
          call this%update_microphysics()
        end if

        call thread_log%debug("loading data to state_var")
        this%camp_state%state_var(:) = boxmod_conc(i_cell, :)

        if (this%time_control%last_forcing_species_update > this%time_control%forcing_species_timestep) then
          call thread_log%debug("updating species forcings")
          this%time_control%last_forcing_species_update = 0.0
          call this%update_forced_species(pressure(i_cell), temperature(i_cell))
        end if

        if (this%time_control%last_deposition_update > this%time_control%deposition_update_timestep) then
          call thread_log%debug("updating deposition rates")
          this%time_control%last_deposition_update = 0.0
          call this%update_deposition_rates()
        end if

        call thread_log%debug("updating water vapor and dimer concentrations")
        this%camp_state%state_var(this%gas_phase_water_id) = &
          water_conc(i_cell)* &
          const%air_molec_weight/const%water_molec_weight*1.e6
        ! water dimer mixing ratio (Kd according to Scribano et al., 2006)
        ! (H2O)2 = Kd * H2O * H2O
        Kd = 4.7856e-4*exp(1851/temperature(i_cell) - 5.10485e-3*temperature(i_cell))    ! atm^(-1)
        Kd = Kd*1e6 ! ppm^(-1)
        this%camp_state%state_var(this%gas_phase_water_dimer_id) = &
          Kd*(this%camp_state%state_var(this%gas_phase_water_id)**2)

        ! update sum species concentrations
        call thread_log%debug("updating sum species")
        call this%update_sum_species_concentrations()

        if (mod(int(start_time), 3600) .eq. 0) then
          call thread_log%info("i_hour loop"//to_string(i_hour))
          i_hour = i_hour + 1
        end if
        call thread_log%debug("solving ODEs at time:"//to_string(start_time))

        ! Start the computation timer
#ifdef CAMP_DEBUG
        if (BOXMOD_PROCESS == 0 .and. i == 1) then
          this%solver_stats%debug_out = .true.
        else
          this%solver_stats%debug_out = .false.
        end if
#endif

        if (this%aerosol_flag) then
          call thread_log%debug("solving GAS+AEROSOL")
          call this%camp_core%solve( &
            this%camp_state, &
            time_step, &
            rxn_phase=GAS_AERO_RXN, &
            solver_stats=this%solver_stats)
        else
          call thread_log%debug("solving GAS only")
          call this%camp_core%solve( &
            this%camp_state, &
            time_step, &
            rxn_phase=GAS_RXN, &
            solver_stats=this%solver_stats)
        end if

        call assert_msg(376450931, this%solver_stats%status_code(1) .eq. 0, &
                        "Solver failed with code "// &
                        to_string(this%solver_stats%solver_flag(1)))

        call thread_log%debug("updating boxmodel concentrations with new state")
        boxmod_conc(i_cell, :) = &
          this%camp_state%state_var(:)

      end do

    else ! for CAMP-GPU
      call thread_log%debug("560220906 in CAMP-GPU integration")
      ! Set initial conditions and environmental parameters for each grid cell

      !Reset state conc
      this%camp_state%state_var(:) = 0.0
      call thread_log%debug("state_size_per_cell="//trim(to_string(state_size_per_cell)))
      do i_cell = 1, this%n_cells
        ! Update the environmental state
        call thread_log%debug("in cell "//trim(to_string(i_cell)))
        call this%camp_state%env_states(i_cell)%set_temperature_K( &
          temperature(i_cell))
        call thread_log%debug(" set temperature to "//trim(to_string(temperature(i_cell))))
        call this%camp_state%env_states(i_cell)%set_pressure_Pa( &
          pressure(i_cell))
        call thread_log%debug(" set pressure to "//trim(to_string(pressure(i_cell))))

        do i_spec = 1, state_size_per_cell
          this%camp_state%state_var(i_spec + (i_cell - 1)*state_size_per_cell) = boxmod_conc(i_cell, i_spec)
        end do
        this%camp_state%state_var(this%gas_phase_water_id + &
                                  ((i_cell - 1)*state_size_per_cell)) = water_conc(i_cell)* &
                                              const%air_molec_weight/const%water_molec_weight*1.e6
        call thread_log%debug(" set water_conc to "//trim(to_string(water_conc(i_cell))))

        ! water dimer mixing ratio (Kd according to Scribano et al., 2006)
        ! (H2O)2 = Kd * H2O * H2O
        Kd = 4.7856e-4*exp(1851/temperature(i_cell) - 5.10485e-3*temperature(i_cell))    ! atm^(-1)
        Kd = Kd*1e6 ! ppm^(-1)
        this%camp_state%state_var(this%gas_phase_water_dimer_id + &
                                  ((i_cell - 1)*state_size_per_cell)) = &
          Kd*(this%camp_state%state_var(this%gas_phase_water_id + &
                                        ((i_cell - 1)*state_size_per_cell))**2)
      end do

      if (this%time_control%last_photolysis_update > this%time_control%photolysis_update_timestep) then
        call thread_log%debug("updating photolysis rates")
        this%time_control%last_photolysis_update = 0.0
        call this%update_photorates(sza)
      end if

      if (this%time_control%last_emissions_update > this%time_control%emissions_update_timestep) then
        call thread_log%debug("updating emission rates")
        this%time_control%last_emissions_update = 0.0
        call this%update_emission_rates(height, pressure, temperature)
      end if

      ! Integrate the CAMP mechanism
      ! print *, "====="
      ! print *, allocated(this%camp_state%state_var), allocated(this%camp_state%env_var)
      ! print *, this%camp_state%state_var(:)
      ! print *, this%camp_state%env_var(:)
      ! print *, "====="

      call this%camp_core%solve(this%camp_state, time_step, solver_stats=this%solver_stats)

      do i_cell = 1, this%n_cells
        do i_spec = 1, state_size_per_cell
          boxmod_conc(i_cell, i_spec) = &
            this%camp_state%state_var(i_spec + (i_cell - 1)*state_size_per_cell)
        end do
      end do

    end if

    !W8 until all process to send data and measure correctly times
#ifdef CAMP_USE_MPI
    !call camp_mpi_barrier()
#endif

    call cpu_time(comp_end)

  end subroutine integrate

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> load information about parallelization
  !> sets this%multiple_boxes_type, this%n_cells, this%load_gpu, this%is_load_balance
  !> and this%solve_multiple_cells
  subroutine load_parallel_config(this, config_file, boxmod_process)
    !> CAMP <-> boxmodel interface
    class(boxmodel_interface_t) :: this
    !> Interface configuration file path
    character(len=:), allocatable, intent(in) :: config_file
    !> rank of the current process
    integer(kind=i_kind), intent(in) :: boxmod_process

    ! json stuff
    logical :: found
    type(json_core), pointer :: json
    type(json_file) :: j_file
    type(json_value), pointer :: j_obj, j_next
    integer(kind=i_kind) :: i_var
    character(len=:), allocatable :: str_val

#ifdef CAMP_USE_JSON
    allocate (json)
    ! Initialize the json objects
    j_obj => null()
    j_next => null()

    call j_file%initialize()
    call j_file%get_core(json)
    call assert_msg(207035903, allocated(config_file), &
                    "Received non-allocated string for file path")
    call assert_msg(368569727, trim(config_file) .ne. "", &
                    "Received empty string for file path")
    inquire (file=config_file, exist=found)
    call assert_msg(134309013, found, "Cannot find file: "// &
                    config_file)
    call j_file%load_file(filename=config_file)

    ! find information about multiple boxes
    call j_file%get("multiple-boxes", str_val)

    select case (str_val)
    case ("single", "SINGLE")
      this%multiple_boxes_type = SINGLE_BOX
    case ("monte carlo", "montecarlo", "monte-carlo", "MONTECARLO")
      this%multiple_boxes_type = MONTE_CARLO
    case DEFAULT
      call die_msg(45102662, str_val//" is an invalid parallel simulation type, choose among 'single', 'monte carlo',")
    end select

    if (this%multiple_boxes_type == MONTE_CARLO) then
      call j_file%get("ncells", i_var)
      this%n_cells = i_var
      this%solve_multiple_cells = .TRUE.
      call j_file%get("load_gpu", i_var)
      call assert_msg(758423964, (i_var <= 100) .and. (i_var >= 0), "load_gpu must be between 0 and 100")
      this%load_gpu = i_var
      call j_file%get("is_load_balance", i_var)
      call assert_msg(758423965, (i_var == 0) .or. (i_var == 1), "is_load_balance must be 0 or 1")
      this%is_load_balance = i_var
    else
      this%n_cells = 1
      this%solve_multiple_cells = .FALSE.
      this%load_gpu = 0.0
      this%is_load_balance = 0
    end if

    ! Clean up the json objects
    call j_file%destroy()
    call json%destroy()
    deallocate (json)

#else
    call die_msg(635417227, "camp <-> boxmodel interface requires "// &
                 "JSON file support.")
#endif

  end subroutine load_parallel_config

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Load the boxmodel <-> camp interface input data
  subroutine load(this, config_file, boxmod_process)

    !> PartMC-camp <-> MONARCH interface
    class(boxmodel_interface_t) :: this
    !> Interface configuration file path
    character(len=:), allocatable, intent(in) :: config_file
    !> rank of the current process
    integer(kind=i_kind), intent(in) :: boxmod_process

#ifdef CAMP_USE_JSON

    type(json_core), pointer :: json
    type(json_file) :: j_file
    type(json_value), pointer :: j_obj, j_next, j_child
    character(kind=json_ck, len=:), allocatable :: key, unicode_str_val

    character(len=:), allocatable :: str_val
    integer(kind=i_kind) :: var_type
    integer(kind=i_kind) :: i_var
    logical :: found
    integer(kind=i_kind) :: i_cell

    ! Initialize the property sets
    this%init_conc_data => property_t()
    this%property_set => property_t()
    this%species_map_data => property_t()
    this%emissions_data => property_t()
    this%microphysics_data => property_t()
    this%forced_species_data => property_t()

    this%time_control => time_control_t()

    this%constrained_sza = .false.

    ! Get a new json core
    allocate (json)

    ! Initialize the json objects
    j_obj => null()
    j_next => null()

    ! Initialize the json file
    call j_file%initialize()
    call j_file%get_core(json)
    call assert_msg(207035903, allocated(config_file), &
                    "Received non-allocated string for file path")
    call assert_msg(368569727, trim(config_file) .ne. "", &
                    "Received empty string for file path")
    inquire (file=config_file, exist=found)
    call assert_msg(134309013, found, "Cannot find file: "// &
                    config_file)
    call j_file%load_file(filename=config_file)

    ! find default random type
    call j_file%get("random_type", str_val)

    select case (str_val)
    case ("RANDOM", "RAND", "random", "rand")
      this%random_type = RANDOM
    case ("SOBOL", "sobol")
      this%random_type = SOBOL
    case DEFAULT
      call die_msg(45102663, str_val//" is not a random number generator type, choose among 'random', 'sobol',")
    end select
    call thread_log%debug("random_type:"//to_string(this%random_type))

    call j_file%get("sobol_offset", this%sobol_offset, found)

    if (found) then
      call assert_msg(45102664, this%sobol_offset >= 0, &
                      to_string(this%sobol_offset)//" is not a valid sobol_offset, should be non-negative number")
    else
      this%sobol_offset = 0
    end if

    ! Find the interface data
    call j_file%get('boxmodel-data(1)', j_obj, found)

    if (.not. found) then
      call die_msg(45102664, "no boxmodel-data was found, check that the initialization data is included in &
        & a `boxmodel-data: [...]` key")
    end if

    ! Load the data to the property_set
    do while (associated(j_obj))

      ! Find the object type
      call json%get(j_obj, 'type', unicode_str_val, found)
      call assert_msg(236838162, found, "Missing type in json input file "// &
                      config_file)
      str_val = unicode_str_val

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!! Load property sets according to type !!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      ! Initial concentration data
      if (str_val .eq. "INIT_CONC") then
        call thread_log%debug("initial concentrations")
        call json%get_child(j_obj, j_child)
        do while (associated(j_child))
          call json%info(j_child, name=key, var_type=var_type)
          if (key .ne. "type" .and. key .ne. "name") then
            call this%init_conc_data%load(json, j_child, .false., key)
          end if
          j_next => j_child
          call json%get_next(j_next, j_child)
        end do
        ! constrained species data
      else if (str_val .eq. "FORCED_CONC") then
        call thread_log%debug("forced concentrations")
        call json%get_child(j_obj, j_child)
        do while (associated(j_child))
          call json%info(j_child, name=key, var_type=var_type)
          if (key .ne. "type" .and. key .ne. "name") then
            call this%forced_species_data%load(json, j_child, .false., key)
          end if
          j_next => j_child
          call json%get_next(j_next, j_child)
        end do
        ! emissions data
      else if (str_val .eq. "EMISSIONS") then
        call thread_log%debug("emissions")
        call json%get_child(j_obj, j_child)
        do while (associated(j_child))
          call json%info(j_child, name=key, var_type=var_type)
          if (key .ne. "type" .and. key .ne. "name") then
            call this%emissions_data%load(json, j_child, .false., key)
          end if
          j_next => j_child
          call json%get_next(j_next, j_child)
        end do
        ! microphysics data
      else if (str_val .eq. "MICROPHYSICS") then
        call thread_log%debug("microphysics")
        call json%get_child(j_obj, j_child)
        do while (associated(j_child))
          call json%info(j_child, name=key, var_type=var_type)
          if (key .ne. "type" .and. key .ne. "name") then
            call this%microphysics_data%load(json, j_child, .false., key)
          end if
          j_next => j_child
          call json%get_next(j_next, j_child)
        end do
        ! time control data
      else if (str_val .eq. "TIME_CONTROL") then
        call thread_log%debug("time control")
        call json%get_child(j_obj, j_child)
        do while (associated(j_child))
          call json%info(j_child, name=key, var_type=var_type)
          if (key .eq. "time_control") then
            this%time_control => time_control_t()
            call this%time_control%load(json, j_child)
            call this%time_control%initialize()
          end if
          j_next => j_child
          call json%get_next(j_next, j_child)
        end do
        ! Species Map data
      else if (str_val .eq. "SPECIES_MAP") then
        call thread_log%debug("species map")
        call json%get_child(j_obj, j_child)
        do while (associated(j_child))
          call json%info(j_child, name=key, var_type=var_type)
          if (key .ne. "type" .and. key .ne. "name") then
            call this%species_map_data%load(json, j_child, .false., key)
          end if
          j_next => j_child
          call json%get_next(j_next, j_child)
        end do
      else if (str_val .eq. "LONGITUDE") then
        call thread_log%debug("longitude")
        call json%get_child(j_obj, j_child)
        do while (associated(j_child))
          call this%load_constraints(this%longitude_constraint, json, j_child)
          j_next => j_child
          call json%get_next(j_next, j_child)
        end do
        do i_cell = 1, this%n_cells
          call this%longitude_constraint(i_cell)%val%initialize(this%random_type)
        end do
      else if (str_val .eq. "LATITUDE") then
        call thread_log%debug("latitude")
        call json%get_child(j_obj, j_child)
        do while (associated(j_child))
          call this%load_constraints(this%latitude_constraint, json, j_child)
          j_next => j_child
          call json%get_next(j_next, j_child)
        end do
        do i_cell = 1, this%n_cells
          call this%latitude_constraint(i_cell)%val%initialize(this%random_type)
        end do
      else if (str_val .eq. "ALTITUDE") then
        call thread_log%debug("altitude")
        call json%get_child(j_obj, j_child)
        do while (associated(j_child))
          call this%load_constraints(this%altitude_constraint, json, j_child)
          j_next => j_child
          call json%get_next(j_next, j_child)
        end do
        do i_cell = 1, this%n_cells
          call this%altitude_constraint(i_cell)%val%initialize(this%random_type)
        end do
      else if (str_val .eq. "PRESSURE") then
        call thread_log%debug("pressure")
        call json%get_child(j_obj, j_child)
        do while (associated(j_child))
          call this%load_constraints(this%pressure_constraint, json, j_child)
          j_next => j_child
          call json%get_next(j_next, j_child)
        end do
      else if (str_val .eq. "TEMPERATURE") then
        call thread_log%debug("temperature")
        call json%get_child(j_obj, j_child)
        do while (associated(j_child))
          call this%load_constraints(this%temperature_constraint, json, j_child)
          j_next => j_child
          call json%get_next(j_next, j_child)
        end do
      else if (str_val .eq. "HUMIDITY") then
        call thread_log%debug("humidity")
        call json%get_child(j_obj, j_child)
        do while (associated(j_child))
          call this%load_constraints(this%humidity_constraint, json, j_child)
          j_next => j_child
          call json%get_next(j_next, j_child)
        end do
      else if (str_val .eq. "HEIGHT") then
        call thread_log%debug("height")
        call json%get_child(j_obj, j_child)
        do while (associated(j_child))
          call this%load_constraints(this%height_constraint, json, j_child)
          j_next => j_child
          call json%get_next(j_next, j_child)
        end do
      else if (str_val .eq. "SZA") then
        call thread_log%debug("sza")
        call json%get_child(j_obj, j_child)
        do while (associated(j_child))
          call this%load_constraints(this%sza_constraint, json, j_child)
          this%constrained_sza = .true.
          j_next => j_child
          call json%get_next(j_next, j_child)
        end do
      else
        call this%property_set%load(json, j_obj, .false., str_val)
      end if

      j_next => j_obj
      call json%get_next(j_next, j_obj)
    end do

    do i_cell = 1, this%n_cells
      call assert_msg(440149419, &
                      associated(this%pressure_constraint(i_cell)%val), &
                      "No pressure constraint was found for cell "//to_string(i_cell))

      call assert_msg(105429399, &
                      associated(this%temperature_constraint(i_cell)%val), &
                      "No temperature constraint was found for cell"//to_string(i_cell))

      call assert_msg(185209350, &
                      associated(this%humidity_constraint(i_cell)%val), &
                      "No humidity constraint was found for cell"//to_string(i_cell))

      call assert_msg(324041440, &
                      associated(this%altitude_constraint(i_cell)%val), &
                      "No altitude constraint was found for cell"//to_string(i_cell))

      call assert_msg(214224156, &
                      associated(this%latitude_constraint(i_cell)%val), &
                      "No latitude constraint was found for cell"//to_string(i_cell))

      call assert_msg(403396571, &
                      associated(this%longitude_constraint(i_cell)%val), &
                      "No longitude constraint was found for cell"//to_string(i_cell))

      call assert_msg(387196367, &
                      associated(this%height_constraint(i_cell)%val), &
                      "No pbl height constraint was found for cell"//to_string(i_cell))

      if (this%constrained_sza) then
        call assert_msg(195553092, &
                        associated(this%sza_constraint(i_cell)%val), &
                        "the constrained_sza flag is true, but no sza constraint was found for cell"//to_string(i_cell))
      end if
    end do

    call assert_msg(875032324, &
                    associated(this%time_control%property_set), &
                    "No time control was found")

    do i_cell = 1, this%n_cells
      call this%pressure_constraint(i_cell)%val%initialize(this%random_type)
      call this%temperature_constraint(i_cell)%val%initialize(this%random_type)
      call this%humidity_constraint(i_cell)%val%initialize(this%random_type)
      call this%height_constraint(i_cell)%val%initialize(this%random_type)
      if (this%constrained_sza) then
        call this%sza_constraint(i_cell)%val%initialize(this%random_type)
      end if
    end do

    call this%load_emissions_constraints()

    ! Clean up the json objects
    call j_file%destroy()
    call json%destroy()
    deallocate (json)

#else
    call die_msg(635417227, "camp <-> boxmodel interface requires "// &
                 "JSON file support.")
#endif

  end subroutine load

  subroutine create_map(this)
    !> CAMP <-> boxmodel interface
    class(boxmodel_interface_t) :: this

    type(chem_spec_data_t), pointer :: chem_spec_data
    character(len=:), allocatable :: key_name, spec_name, photolysis_model

    ! Get the chemical species data
    call assert_msg(731700229, &
                    this%camp_core%get_chem_spec_data(chem_spec_data), &
                    "No chemical species data in camp_core.")

    ! Set the gas-phase water id
    key_name = "gas-phase water"
    call assert_msg(413656652, &
                    this%species_map_data%get_string(key_name, spec_name), &
                    "Missing gas-phase water species for MONARCH interface.")
    this%gas_phase_water_id = chem_spec_data%gas_state_id(spec_name)
    call assert_msg(910692272, this%gas_phase_water_id .gt. 0, &
                    "Could not find gas-phase water species '"//spec_name//"'.")

    ! Set the gas-phase water dimer id
    key_name = "gas-phase water dimer"
    call assert_msg(148170138, &
                    this%species_map_data%get_string(key_name, spec_name), &
                    "Missing gas-phase water dimer species for MONARCH interface.")
    this%gas_phase_water_dimer_id = chem_spec_data%gas_state_id(spec_name)
    call assert_msg(925412080, this%gas_phase_water_id .gt. 0, &
                    "Could not find gas-phase water dimer species '"//spec_name//"'.")

    ! Get the mechanism name
    key_name = "mechanism_name"
    call assert_msg(643754355, &
                    this%species_map_data%get_string(key_name, this%mechanism_name), &
                    "Missing mechanism name.")

    key_name = "aerosol_flag"
    if (.not. this%species_map_data%get_logical(key_name, this%aerosol_flag)) then
      this%aerosol_flag = .TRUE.
    end if

    ! get photolysis model
    key_name = "photolysis_method"
    call assert_msg(776509727, &
                    this%species_map_data%get_string(key_name, this%photolysis_model), &
                    "missing photolysis_method key ('CLOUDJ', 'TUV', 'TABLE', 'NONE')")

  end subroutine create_map

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  logical function check_traj_initialization(this)

    !> CAMP <-> boxmodel interface
    class(boxmodel_interface_t), intent(in) :: this

    integer(kind=i_kind) :: i_cell

    if (.not. allocated(this%latitude_constraint) .or. &
        .not. allocated(this%longitude_constraint) .or. &
        .not. allocated(this%altitude_constraint)) then
      check_traj_initialization = .FALSE.
      return
    end if

    check_traj_initialization = associated(this%time_control)

    do i_cell = 1, this%n_cells
      check_traj_initialization = check_traj_initialization .and. &
                                  associated(this%latitude_constraint(i_cell)%val) .and. &
                                  associated(this%longitude_constraint(i_cell)%val) .and. &
                                  associated(this%altitude_constraint(i_cell)%val)
    end do

  end function check_traj_initialization
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine load_emissions_constraints(this)
    class(boxmodel_interface_t) :: this

    type(property_t), pointer :: gas_species_emissions, aer_species_emissions
    type(property_t), pointer :: species_data
    character(len=:), allocatable :: key_name, spec_name
    integer(kind=i_kind)  :: i_spec, i_cell
    type(property_t), pointer :: constraint_set => null()

    this%emissions_map(:)%n_emission = 0

    ! get gas phase species emissions
    key_name = "gas-phase species"
    if (this%emissions_data%get_property_t(key_name, gas_species_emissions)) then
      this%emissions_map(:)%n_emission = this%emissions_map(:)%n_emission + gas_species_emissions%size()
    end if
    ! get aerosol phase species emissiosn
    key_name = "aer-phase species"
    if (this%emissions_data%get_property_t(key_name, aer_species_emissions)) then
      call die_msg(125791688, &
                   "Aerosol emissions not yet implemented")
    end if

    do i_cell = 1, this%n_cells
      allocate (this%emissions_map(i_cell)%emission_rxn_ind(this%emissions_map(i_cell)%n_emission))
      allocate (this%emissions_map(i_cell)%emissions_constraints(this%emissions_map(i_cell)%n_emission))
      allocate (this%emissions_map(i_cell)%emissions_rates_updates(this%emissions_map(i_cell)%n_emission))
      allocate (this%emissions_map(i_cell)%emitted_species(this%emissions_map(i_cell)%n_emission))
      allocate (this%emissions_map(i_cell)%emitted_id(this%emissions_map(i_cell)%n_emission))
      allocate (this%emissions_map(i_cell)%current_emissions_rates(this%emissions_map(i_cell)%n_emission))
    end do

    i_spec = 0
    if (associated(gas_species_emissions)) then
      call gas_species_emissions%iter_reset()
      do while (gas_species_emissions%get_key(spec_name))
        i_spec = i_spec + 1

        call assert_msg(651273586, &
                        gas_species_emissions%get_property_t(val=species_data), &
                        "Missing data for '"//spec_name//"' for emission initialization.")

        do i_cell = 1, this%n_cells
          this%emissions_map(i_cell)%emitted_species(i_spec) = string_t(spec_name)
        end do

        if (associated(constraint_set)) then
          deallocate (constraint_set)
        end if
        if (species_data%get_property_t("constant_constraint", constraint_set)) then
          do i_cell = 1, this%n_cells
            this%emissions_map(i_cell)%emissions_constraints(i_spec)%val => constant_constraint_t()
            this%emissions_map(i_cell)%emissions_constraints(i_spec)%val%property_set => constraint_set
          end do
        else if (species_data%get_property_t("file_constraint", constraint_set)) then
          do i_cell = 1, this%n_cells
            this%emissions_map(i_cell)%emissions_constraints(i_spec)%val => file_constraint_t()
            this%emissions_map(i_cell)%emissions_constraints(i_spec)%val%property_set => constraint_set
          end do
        else if (species_data%get_property_t("monarch_constraint", constraint_set)) then
          call die_msg(343285089, &
                       "emission constrained by monarch is not yet available for species "//spec_name)
        else
          call die_msg(127660373, "unknown constraint type for "//spec_name)
        end if

        do i_cell = 1, this%n_cells
          call this%emissions_map(i_cell)%emissions_constraints(i_spec)%val%initialize(this%random_type)
        end do
        call gas_species_emissions%iter_next()
      end do
    end if

    ! copy initialized emissions_map(1) into the other cells maps
    ! they can be modified at a later stage for randomization
    do i_cell = 2, this%n_cells
      this%emissions_map(i_cell) = this%emissions_map(1)
    end do

  end subroutine load_emissions_constraints

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine load_forced_species_constraints(this)
    class(boxmodel_interface_t), intent(inout) :: this
    character(len=:), allocatable :: key_name, representation_name, spec_name

    type(property_t), pointer     :: gas_species_forcing => null()
    type(property_t), pointer     :: aerosol_species_forcing => null()
    type(property_t), pointer     :: species_data => null()

    type(chem_spec_data_t), pointer :: chem_spec_data => null()
    class(aero_rep_data_t), pointer :: aero_rep => null()

    integer(kind=i_kind)          :: i_spec
    type(property_t), pointer :: constraint_set => null()

    this%forced_species_map%n_species = 0

    key_name = "gas-phase species"
    if (this%forced_species_data%get_property_t(key_name, gas_species_forcing)) then
      this%forced_species_map%n_species = this%forced_species_map%n_species + gas_species_forcing%size()
    end if

    key_name = "aerosol-phase species"
    if (this%forced_species_data%get_property_t(key_name, aerosol_species_forcing)) then
      this%forced_species_map%n_species = this%forced_species_map%n_species + aerosol_species_forcing%size()
    end if

    ! Get the chemical species data
    call assert_msg(279150286, &
                    this%camp_core%get_chem_spec_data(chem_spec_data), &
                    "No chemical species data in camp_core.")

    allocate (this%forced_species_map%species_constraints(this%forced_species_map%n_species))
    allocate (this%forced_species_map%forced_species_ind(this%forced_species_map%n_species))
    allocate (this%forced_species_map%is_aerosol(this%forced_species_map%n_species))

    i_spec = 0
    if (associated(gas_species_forcing)) then
      call gas_species_forcing%iter_reset()
      do while (gas_species_forcing%get_key(spec_name))
        i_spec = i_spec + 1

        call assert_msg(313992533, &
                        gas_species_forcing%get_property_t(val=species_data), &
                        "Missing data for '"//spec_name//"' for species forcing.")

        this%forced_species_map%forced_species_ind(i_spec) = chem_spec_data%gas_state_id(spec_name)
        call assert_msg(847336279, &
                        this%forced_species_map%forced_species_ind(i_spec) > 0, &
                        "Could not find forced species "//spec_name)

        this%forced_species_map%is_aerosol(i_spec) = .FALSE.

        if (associated(constraint_set)) then
          deallocate (constraint_set)
        end if
        if (species_data%get_property_t("constant_constraint", constraint_set)) then
          this%forced_species_map%species_constraints(i_spec)%val => constant_constraint_t()
          this%forced_species_map%species_constraints(i_spec)%val%property_set => constraint_set
        else if (species_data%get_property_t("file_constraint", constraint_set)) then
          this%forced_species_map%species_constraints(i_spec)%val => file_constraint_t()
          this%forced_species_map%species_constraints(i_spec)%val%property_set => constraint_set
        else if (species_data%get_property_t("monarch_constraint", constraint_set)) then
          call die_msg(206208389, &
                       "forced species constrained by monarch is not yet available for species "//spec_name)
        else
          call die_msg(394709515, "unknown constraint type for forced species "//spec_name)
        end if

        call this%forced_species_map%species_constraints(i_spec)%val%initialize(this%random_type)

        call this%forced_species_map%species_constraints(i_spec)%val%print()
        call gas_species_forcing%iter_next()
      end do
    end if

    if (associated(aerosol_species_forcing)) then
      call aerosol_species_forcing%iter_reset()

      do while (aerosol_species_forcing%get_key(spec_name))
        i_spec = i_spec + 1

        call assert_msg(902013240, &
                        aerosol_species_forcing%get_property_t(val=species_data), &
                        "Missing data for '"//spec_name//"' for species forcing.")

        call assert_msg(218776362, &
                        species_data%get_string("aerosol representation name", representation_name), &
                        "MIssing aerosol representation name for forced species '"//spec_name//"'.")

        call assert_msg(198539576, &
                        this%camp_core%get_aero_rep(representation_name, aero_rep), &
                        "The aerosol representation '"//representation_name//"' was not found.")

        this%forced_species_map%forced_species_ind(i_spec) = aero_rep%spec_state_id(spec_name)
        call assert_msg(306705867, &
                        this%forced_species_map%forced_species_ind(i_spec) > 0, &
                        "The forced species '"//spec_name// &
                        "' could not be found in aerosol representation '"//representation_name//"'")

        this%forced_species_map%is_aerosol(i_spec) = .TRUE.

        if (associated(constraint_set)) then
          deallocate (constraint_set)
        end if
        if (species_data%get_property_t("constant_constraint", constraint_set)) then
          this%forced_species_map%species_constraints(i_spec)%val => constant_constraint_t()
          this%forced_species_map%species_constraints(i_spec)%val%property_set => constraint_set
        else if (species_data%get_property_t("file_constraint", constraint_set)) then
          this%forced_species_map%species_constraints(i_spec)%val => file_constraint_t()
          this%forced_species_map%species_constraints(i_spec)%val%property_set => constraint_set
        else if (species_data%get_property_t("monarch_constraint", constraint_set)) then
          call die_msg(193417056, &
                       "forced species constrained by monarch is not yet available for species "//spec_name)
        else
          call die_msg(373991387, "unknown constraint type for forced species "//spec_name)
        end if

        call this%forced_species_map%species_constraints(i_spec)%val%initialize(this%random_type)

        call this%forced_species_map%species_constraints(i_spec)%val%print()
        call aerosol_species_forcing%iter_next()
      end do
    end if

  end subroutine load_forced_species_constraints

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine load_microphysics_constraints(this)
    class(boxmodel_interface_t), intent(inout) :: this
    character(len=:), allocatable :: key_name, representation_name, section_name

    class(aero_rep_data_t), pointer :: aero_rep => null()
    type(aero_rep_modal_binned_mass_t), pointer :: modal_rep => null()

    type(property_t), pointer :: sections_subset => null()
    type(property_t), pointer :: constraints_subset => null()
    type(property_t), pointer :: global_diameter_constraint_subset => null()
    type(property_t), pointer :: global_stdev_constraint_subset => null()
    type(property_t), pointer :: diameter_constraint_subset => null()
    type(property_t), pointer :: stdev_constraint_subset => null()

    integer(kind=i_kind) :: i_distrib, i_section

    ! check if there are any modal/binned aerosol representation to constrain
    if (this%microphysics_data%size() == 0) then
      return
    end if

    this%microphysics_map%n_distrib = 0

    key_name = "aerosol_representation"
    call assert_msg( &
      146926295, &
      this%microphysics_data%get_string(key_name, representation_name), &
      "in MICROPHYSICS section, could not find key "//key_name)

    call assert_msg( &
      89209936, &
      this%camp_core%get_aero_rep(representation_name, aero_rep), &
      "in MICROPHYSICS section, could not find aerosol representation "//trim(representation_name))

    select type (rep => aero_rep)
    type is (aero_rep_modal_binned_mass_t)
      modal_rep => rep

    class default
      call die_msg( &
        675913908, &
        "in MICROPHYSICS section, can only define microphysics for modal/binned aerosol representation type")
    end select

    call this%camp_core%initialize_update_object(modal_rep, &
                                                 this%microphysics_map%update_data_GMD)
    call this%camp_core%initialize_update_object(modal_rep, &
                                                 this%microphysics_map%update_data_GSD)

    ! count the number of modal section, we expect one constraint for each
    i_distrib = 0
    do i_section = 1, size(modal_rep%section_name)

      ! each modal section of the aerosol representation needs constraints
      if (modal_rep%section_type(i_section) == "MODAL") then

        i_distrib = i_distrib + 1
      end if
    end do

    this%microphysics_map%n_distrib = i_distrib
    allocate (this%microphysics_map%section_ids(this%microphysics_map%n_distrib))
    allocate (this%microphysics_map%diameter_constraints(this%microphysics_map%n_distrib))
    allocate (this%microphysics_map%stdev_constraints(this%microphysics_map%n_distrib))

    key_name = "constraints"
    call assert_msg( &
      201850287, &
      this%microphysics_data%get_property_t(key_name, sections_subset), &
      "missing 'constraints' key in MICROPHYSICS section")

    i_distrib = 0
    do i_section = 1, size(modal_rep%section_name)
      ! each modal section of the aerosol representation needs constraints
      if (modal_rep%section_type(i_section) == "MODAL") then
        section_name = modal_rep%section_name(i_section)%string

        if (associated(constraints_subset)) then
          deallocate (constraints_subset)
        end if

        call assert_msg( &
          294495387, &
          sections_subset%get_property_t(section_name, constraints_subset), &
          " a MICROPHYSICS constraint is required for model section "//section_name)

        i_distrib = i_distrib + 1

        this%microphysics_map%section_ids(i_distrib) = i_section

        key_name = "mean_diameter"

        if (associated(global_diameter_constraint_subset)) then
          deallocate (global_diameter_constraint_subset)
        end if
        if (associated(diameter_constraint_subset)) then
          deallocate (diameter_constraint_subset)
        end if

        call assert_msg( &
          185680119, &
          constraints_subset%get_property_t(key=key_name, val=global_diameter_constraint_subset), &
          "in MICROPHYSICS, section "//section_name//", key not found: "//key_name)

        if (global_diameter_constraint_subset%get_property_t("constant_constraint", diameter_constraint_subset)) then
          this%microphysics_map%diameter_constraints(i_distrib)%val => constant_constraint_t()
          this%microphysics_map%diameter_constraints(i_distrib)%val%property_set => diameter_constraint_subset

        else if (global_diameter_constraint_subset%get_property_t("file_constraint", diameter_constraint_subset)) then
          this%microphysics_map%diameter_constraints(i_distrib)%val => file_constraint_t()
          this%microphysics_map%diameter_constraints(i_distrib)%val%property_set => diameter_constraint_subset
        else if (global_diameter_constraint_subset%get_property_t("monarch_constraint", diameter_constraint_subset)) then
          call die_msg(165617190, &
                       "microphysics constrained by monarch is not yet available for section "//section_name)
        else
          call die_msg(245670670, "unknown constraint type for section"//section_name)
        end if

        call this%microphysics_map%diameter_constraints(i_distrib)%val%initialize(this%random_type)

        key_name = "stdev"

        if (associated(global_stdev_constraint_subset)) then
          deallocate (global_stdev_constraint_subset)
        end if

        if (associated(stdev_constraint_subset)) then
          deallocate (stdev_constraint_subset)
        end if

        call assert_msg( &
          185680119, &
          constraints_subset%get_property_t(key=key_name, val=global_stdev_constraint_subset), &
          "in MICROPHYSICS, section "//section_name//", key not found: "//key_name)

        if (global_stdev_constraint_subset%get_property_t("constant_constraint", stdev_constraint_subset)) then
          this%microphysics_map%stdev_constraints(i_distrib)%val => constant_constraint_t()
          this%microphysics_map%stdev_constraints(i_distrib)%val%property_set => stdev_constraint_subset
        else if (global_stdev_constraint_subset%get_property_t("file_constraint", stdev_constraint_subset)) then
          this%microphysics_map%stdev_constraints(i_distrib)%val => file_constraint_t()
          this%microphysics_map%stdev_constraints(i_distrib)%val%property_set => stdev_constraint_subset
        else if (global_stdev_constraint_subset%get_property_t("monarch_constraint", stdev_constraint_subset)) then
          call die_msg(165617190, &
                       "microphysics constrained by monarch is not yet available for section "//section_name)
        else
          call die_msg(245670670, "unknown constraint type for section"//section_name)
        end if
        call this%microphysics_map%stdev_constraints(i_distrib)%val%initialize(this%random_type)

      end if
    end do

    this%microphysics_map%n_distrib = i_distrib

  end subroutine load_microphysics_constraints

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine load_constraints(this, constraint, json, data)
    class(boxmodel_interface_t), intent(in) :: this
    type(constraint_ptr), intent(inout), allocatable :: constraint(:)
    type(json_core), pointer, intent(in) :: json
    type(json_value), pointer, intent(in) :: data

    character(len=:), allocatable :: key

    integer(kind=i_kind) :: var_type, i_cell

    call json%info(data, name=key, var_type=var_type)

    ! check that this is a constraint name key we are looking for
    select case (key)
    case ("constant_constraint", "file_constraint", "monarch_constraint")
      call thread_log%debug("allocating constraint")
      allocate (constraint(this%n_cells))
      call thread_log%debug("constraint allocated on "//trim(to_string(this%n_cells))//" cells")
    case default
      return
    end select

    do i_cell = 1, this%n_cells
      if (key == "constant_constraint") then
        constraint(i_cell)%val => constant_constraint_t()
        call constraint(i_cell)%val%load(json, data)
      else if (key == "file_constraint") then
        constraint(i_cell)%val => file_constraint_t()
        call constraint(i_cell)%val%load(json, data)
      else if (key == "monarch_constraint") then
        call assert_msg(551071447, &
                        this%check_traj_initialization(), &
                        "trajectory elements (time, lon, lat, alt) need to be initialized before any monarch constraint")
        constraint(i_cell)%val => monarch_constraint_from_trajectory( &
                                  this%time_control, &
                                  this%latitude_constraint(i_cell), &
                                  this%longitude_constraint(i_cell), &
                                  this%altitude_constraint(i_cell))
        call constraint(i_cell)%val%load(json, data)
      end if
    end do
  end subroutine load_constraints
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Load initial concentrations
  subroutine load_init_conc(this)

    !> CAMP <-> boxmodel interface
    class(boxmodel_interface_t) :: this

    type(chem_spec_data_t), pointer :: chem_spec_data
    class(aero_rep_data_t), pointer :: aero_rep_ptr
    type(property_t), pointer :: gas_species_list, aero_species_list, species_data, init_conc_subset
    character(len=:), allocatable :: key_name, spec_name, rep_name, unit_name
    integer(kind=i_kind) :: i_spec, num_spec
    real, parameter :: factor_ppb_to_ppm = 1.0E-3, factor_ppp_to_ppm = 1.0E6, factor_ppt_to_ppm = 1e-6
    real(kind=dp) :: conversion_factor

    num_spec = 0

    ! Get the gas-phase species
    key_name = "gas-phase species"
    if (this%init_conc_data%get_property_t(key_name, gas_species_list)) then
      num_spec = num_spec + gas_species_list%size()
    end if

    ! Get the aerosol-phase species
    key_name = "aerosol-phase species"
    if (this%init_conc_data%get_property_t(key_name, aero_species_list)) then
      num_spec = num_spec + aero_species_list%size()
    end if

    call thread_log%info("found initial concentrations for "//trim(to_string(num_spec))//" species")

    ! Get the chemical species data
    call assert_msg(885063268, &
                    this%camp_core%get_chem_spec_data(chem_spec_data), &
                    "No chemical species data in camp_core.")

    ! Allocate space for the initial concentrations and indices
    allocate (this%init_conc_camp_id(num_spec))
    allocate (this%init_conc(num_spec))
    this%init_conc(:)%value = 0.
    this%init_conc_camp_id = -1

    ! Add the gas-phase initial concentrations
    if (associated(gas_species_list)) then

      ! Loop through the gas-phase species and load the initial concentrations
      call gas_species_list%iter_reset()
      i_spec = 0
      do while (gas_species_list%get_key(spec_name))
        i_spec = i_spec + 1

        call assert_msg(325582312, &
                        gas_species_list%get_property_t(val=species_data), &
                        "Missing species data for '"//spec_name//"' for "// &
                        "camp initial concentrations.")

        key_name = "init conc"
        call assert_msg(445070498, &
                        species_data%get_property_t(key_name, this%init_conc(i_spec)%property_set), &
                        "Missing 'init conc' for species '"//spec_name//" for "// &
                        "camp initial concentrations.")

        call this%init_conc(i_spec)%initialize(this%random_type)

        ! Unit change - camp works with ppm
        conversion_factor = 0.0
        select case (this%init_conc(i_spec)%value_unit)
        case ("ppm")
          ! nothing to do
          conversion_factor = 1.0
        case ("ppb")
          conversion_factor = factor_ppb_to_ppm
        case ("ppt")
          conversion_factor = factor_ppt_to_ppm
        case ("ppp")
          conversion_factor = factor_ppp_to_ppm
        case DEFAULT
          call die_msg(479948915, "unit '"//this%init_conc(i_spec)%value_unit &
                       //"' is not a valid unit"// &
                       "for species '"//spec_name//"'.")
        end select

        this%init_conc(i_spec)%value = this%init_conc(i_spec)%value*conversion_factor

        ! if we are using the gaussian montecarlo
        ! we need to convert the standard deviation too
        if (this%init_conc(i_spec)%montecarlo_fg) then
          select type (mc => this%init_conc(i_spec)%montecarlo%val)
          type is (montecarlo_gaussian_t)
            mc%stdev = mc%stdev*conversion_factor
          end select
        end if

        this%init_conc(i_spec)%value_unit = "ppm"

        this%init_conc_camp_id(i_spec) = &
          chem_spec_data%gas_state_id(spec_name)
        call assert_msg(940200584, this%init_conc_camp_id(i_spec) .gt. 0, &
                        "Could not find species '"//spec_name//"' in camp.")

        call gas_species_list%iter_next()
      end do
    end if

    ! Add the aerosol-phase species initial concentrations
    if (associated(aero_species_list)) then

      call aero_species_list%iter_reset()
      do while (aero_species_list%get_key(spec_name))
        i_spec = i_spec + 1
        call assert_msg(331096555, &
                        aero_species_list%get_property_t(val=species_data), &
                        "Missing species data for '"//spec_name//"' for "// &
                        "camp initial concentrations.")

        key_name = "init conc"
        call assert_msg(782275469, &
                        species_data%get_property_t(key_name, this%init_conc(i_spec)%property_set), &
                        "Missing 'init conc' for species '"//spec_name//"' for "// &
                        "camp initial concentrations.")

        call this%init_conc(i_spec)%initialize(this%random_type)

        ! Unit change - camp works with kg/m3 for aerosols
        select case (this%init_conc(i_spec)%value_unit)
        case ("ug/m3")
          conversion_factor = 1e-9
        case ("kg/m3")
          conversion_factor = 1.0
          ! nothing to do
        case DEFAULT
          call die_msg(228484637, "unit '"//this%init_conc(i_spec)%value_unit// &
                       "' is not a valid unit"// &
                       "for species '"//spec_name//"'.")
        end select

        this%init_conc(i_spec)%value = this%init_conc(i_spec)%value*conversion_factor

        ! if we are using the gaussian montecarlo
        ! we need to convert the standard deviation too
        if (this%init_conc(i_spec)%montecarlo_fg) then
          select type (mc => this%init_conc(i_spec)%montecarlo%val)
          type is (montecarlo_gaussian_t)
            mc%stdev = mc%stdev*conversion_factor
          end select
        end if

        this%init_conc(i_spec)%value_unit = "kg/m3"

        key_name = "aerosol representation name"
        call assert_msg(150863332, &
                        species_data%get_string(key_name, rep_name), &
                        "Missing aerosol representation name for species '"// &
                        spec_name//"' for camp initial concentrations.")

        ! Find the species camp id
        this%init_conc_camp_id(i_spec) = 0
        call assert_msg(258814777, &
                        this%camp_core%get_aero_rep(rep_name, aero_rep_ptr), &
                        "Could not find aerosol representation '"//rep_name//"'")
        this%init_conc_camp_id(i_spec) = &
          aero_rep_ptr%spec_state_id(spec_name)
        call assert_msg(437149649, this%init_conc_camp_id(i_spec) .gt. 0, &
                        "Could not find aerosol species '"//spec_name//"' in "// &
                        "aerosol representation '"//rep_name//"'.")

        call aero_species_list%iter_next()
      end do
    end if

  end subroutine load_init_conc

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Get initial concentrations for the mock MONARCH model (for testing only)
  subroutine set_init_conc(this, boxmod_conc, boxmod_water_conc)

    !> PartMC-camp <-> MONARCH interface
    class(boxmodel_interface_t) :: this
    !> MONARCH species concentrations to update
    real(kind=dp), intent(inout) :: boxmod_conc(:, :)
    !> Atmospheric water concentrations (kg_H2O/kg_air)
    real(kind=dp), intent(in) :: boxmod_water_conc(:)

    integer(kind=i_kind) :: i_spec, water_id, i, j, k, i_cell, state_size_per_cell

    state_size_per_cell = this%camp_core%state_size_per_cell()

    ! Reset the species concentrations in CAMP and boxmod
    this%camp_state%state_var = 0.0
    boxmod_conc = 0.0

    ! TODO: rework initial concentrations

    ! Set initial concentrations in CAMP

    do i_cell = 1, this%n_cells
      do i_spec = 1, size(this%init_conc)
        ! the unit conversion has been done in load_init_conc
        ! because we needed to know if the species is gaseous or particulate
        this%camp_state%state_var(this%init_conc_camp_id(i_spec) + (i_cell - 1)*state_size_per_cell) = this%init_conc(i_spec)%value
      end do

      ! Copy species concentrations to boxmod array
      do i_spec = 1, size(this%camp_state%state_var)
        boxmod_conc(i_cell, i_spec) = this%camp_state%state_var(i_spec + (i_cell - 1)*state_size_per_cell)
      end do

      this%camp_state%state_var(this%gas_phase_water_id + (i_cell - 1)*state_size_per_cell) = &
        boxmod_water_conc(i_cell)* &
        const%air_molec_weight/const%water_molec_weight*1.e6

    end do

  end subroutine set_init_conc

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine init_deposition_reactions(this)
    class(boxmodel_interface_t), intent(inout) :: this
    type(mechanism_data_t), pointer :: mechanism
    class(rxn_data_t), pointer :: rxn
    type(property_t), pointer :: properties => null()

    character(len=:), allocatable :: loss_type, loss_type_key_name, species_key_name, species_name
    integer(kind=i_kind) :: i_rxn, i_deposition

    ! find first order loss reactions that have a loss type key
    call assert_msg(826108501, &
                    associated(this%camp_core), &
                    "camp_core in camp_interface not associated")
    call assert_msg(148315040, &
                    associated(this%camp_core%mechanism), &
                    "mechanism in camp_interface%camp_core not associated")

    ! Check if mechanism name is found in the input data
    call assert_msg(664473537, &
                    this%camp_core%get_mechanism(this%mechanism_name, mechanism), &
                    "Could not find the selected mechanism in the input data.")

    loss_type_key_name = "loss type"
    species_key_name = "species"

    this%deposition_map%n_deposition = 0
    do i_rxn = 1, mechanism%size()
      rxn => mechanism%get_rxn(i_rxn)
      select type (rxn)
      class is (rxn_first_order_loss_t)
        if (rxn%property_set%get_string(loss_type_key_name, loss_type)) then
          if (loss_type == "dry dep") then
            this%deposition_map%n_deposition = this%deposition_map%n_deposition + 1
          end if
        end if
      end select
    end do

    allocate (this%deposition_map%deposited_species(this%deposition_map%n_deposition))
    allocate (this%deposition_map%deposition_rxn_ind(this%deposition_map%n_deposition))
    allocate (this%deposition_map%deposition_rates_updates(this%deposition_map%n_deposition))

    i_deposition = 0
    do i_rxn = 1, mechanism%size()
      rxn => mechanism%get_rxn(i_rxn)
      select type (rxn)
      class is (rxn_first_order_loss_t)
        if (rxn%property_set%get_string(loss_type_key_name, loss_type)) then
          if (loss_type == "dry dep") then
            i_deposition = i_deposition + 1
            this%deposition_map%deposition_rxn_ind(i_deposition) = i_rxn
            ! initialize camp update object
            call this%camp_core%initialize_update_object(rxn, &
                                                         this%deposition_map%deposition_rates_updates(i_deposition))
            if (rxn%property_set%get_string(species_key_name, species_name)) then
              this%deposition_map%deposited_species(i_deposition)%string = species_name
            end if
          end if
        end if
      end select
    end do

  end subroutine init_deposition_reactions

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine init_photolysis_reactions(this)
    class(boxmodel_interface_t), intent(inout) :: this
    type(mechanism_data_t), pointer :: mechanism
    class(rxn_data_t), pointer :: rxn
    type(property_t), pointer :: properties => null()
    character(len=:), allocatable :: phot_label, rxn_type, phot_label_key, key_name
    type(property_t), pointer :: phot_subset => null()

    integer(kind=i_kind) :: i_rxn, i_photo_rxn, J, k, max_assigned_values
    integer(kind=i_kind) :: i_cell

    ! find photolysis reactions that have a cloud-j key
    call assert_msg(826108501, &
                    associated(this%camp_core), &
                    "camp_core in camp_interface not associated")
    call assert_msg(148315040, &
                    associated(this%camp_core%mechanism), &
                    "mechanism in camp_interface%camp_core not associated")

    ! Check if mechanism name is found in the input data
    call assert_msg(664473538, &
                    this%camp_core%get_mechanism(this%mechanism_name, mechanism), &
                    "Could not find the selected mechanism in the input data.")

    select case (this%photolysis_model)
    case ("CLOUDJ")
      phot_label_key = "Cloud-J id"
    case ("TUV")
      phot_label_key = "TUV id"
    case ("NONE")
      return
    case default
      call die_msg(369090234, "invalid photolysis model selected: "//this%photolysis_model)
    end select

    this%photolysis_map%nphot_reac = 0
    do i_rxn = 1, mechanism%size()
      rxn => mechanism%get_rxn(i_rxn)
      select type (rxn)
      class is (rxn_photolysis_t)
        if (rxn%property_set%get_property_t(phot_label_key, phot_subset)) then
          this%photolysis_map%nphot_reac = this%photolysis_map%nphot_reac + 1
        end if
      end select
    end do

    allocate (this%photolysis_map%n_assigned_j(this%photolysis_map%nphot_reac))
    allocate (this%photolysis_map%photo_rate_updates(this%n_cells, this%photolysis_map%nphot_reac))

    ! find the maximum number of jvalues that can be assigned to a photolysis reaction
    i_photo_rxn = 0
    max_assigned_values = 0
    do i_rxn = 1, mechanism%size()
      rxn => mechanism%get_rxn(i_rxn)
      select type (rxn)
      class is (rxn_photolysis_t)
        if (rxn%property_set%get_property_t(phot_label_key, phot_subset)) then
          i_photo_rxn = i_photo_rxn + 1
          if (phot_subset%size() .gt. max_assigned_values) then
            max_assigned_values = phot_subset%size()
          end if
          this%photolysis_map%n_assigned_j(i_photo_rxn) = phot_subset%size()
        end if
      end select
    end do

    this%photolysis_map%max_assigned_j = max_assigned_values

    allocate (this%photolysis_map%photolysis_label(this%photolysis_map%nphot_reac, max_assigned_values))
    allocate (this%photolysis_map%weighting_factors(this%photolysis_map%nphot_reac, max_assigned_values))
    this%photolysis_map%weighting_factors = 1.0

    ! set the CAMP and Cloud-J/TUV-id ids
    i_photo_rxn = 0
    key_name = "factor"
    do i_rxn = 1, mechanism%size()
      rxn => mechanism%get_rxn(i_rxn)
      select type (rxn)
      class is (rxn_photolysis_t)
        if (rxn%property_set%get_property_t(phot_label_key, phot_subset)) then
          i_photo_rxn = i_photo_rxn + 1

          call phot_subset%iter_reset()
          k = 0
          do while (phot_subset%get_key(phot_label))
            k = k + 1
            this%photolysis_map%photolysis_label(i_photo_rxn, k)%string = phot_label

            if (.not. phot_subset%get_real(key_name, this%photolysis_map%weighting_factors(i_photo_rxn, k))) then
              this%photolysis_map%weighting_factors(i_photo_rxn, k) = 1.0
            end if
            call phot_subset%iter_next()
          end do
          ! get cloud-j labels and weights
          !this%photolysis_map%photolysis_label(i_photo_rxn)%string = phot_label
          ! initialize camp update object
          do i_cell = 1, this%n_cells
            call this%camp_core%initialize_update_object(rxn, &
                                                         this%photolysis_map%photo_rate_updates(i_cell, i_photo_rxn))
          end do
        end if
      end select
    END DO

  end subroutine init_photolysis_reactions

  subroutine update_photorates(this, sza)
    class(boxmodel_interface_t), intent(inout) :: this
    real(kind=dp), intent(in) :: sza(this%n_cells)
    integer(kind=i_kind) :: i_photo_rxn, i_cell

    if (this%photolysis_model /= "NONE") then
      call this%photolysis_map%update_photolysis_reactions_rates(sza)

      do i_cell = 1, this%n_cells
        do i_photo_rxn = 1, this%photolysis_map%nphot_reac
          call this%camp_core%update_data(this%photolysis_map%photo_rate_updates(i_cell, i_photo_rxn), i_cell)
        end do
      end do
    end if

  end subroutine update_photorates

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!> load emissions data properties to emissions map
  subroutine load_emission_data(this)
    class(boxmodel_interface_t), intent(inout) :: this

    type(mechanism_data_t), pointer :: mechanism
    type(property_t), pointer :: gas_species_emissions, aer_species_emissions
    type(json_value), pointer ::  species_data
    character(len=:), allocatable :: key_name, spec_name, rxn_spec_name, rxn_type, key
    class(rxn_data_t), pointer :: rxn
    integer(kind=i_kind) :: i_spec, i_emissions_rxn, i_rxn, i_cell
    type(property_t), pointer :: constraint_set
    logical :: found

    ! Check if mechanism name is found in the input data
    call assert_msg(713595716, &
                    this%camp_core%get_mechanism(this%mechanism_name, mechanism), &
                    "Could not find the selected mechanism in the input data.")

    do i_cell = 1, this%n_cells
      this%emissions_map(i_cell)%emission_rxn_ind = -1
    end do

    do i_spec = 1, this%emissions_map(1)%n_emission
      found = .false.
      do i_rxn = 1, mechanism%size()
        rxn => mechanism%get_rxn(i_rxn)
        select type (rxn)
        class is (rxn_emission_t)
          if (.not. rxn%property_set%get_string("species", rxn_spec_name)) then
            call die_msg(437547027, &
                         "Emission reaction without species key")
          end if
          if (trim(this%emissions_map(1)%emitted_species(i_spec)%string) == trim(rxn_spec_name)) then
            do i_cell = 1, this%n_cells
              this%emissions_map(i_cell)%emission_rxn_ind(i_spec) = i_rxn
              ! initialize camp update object
              call this%camp_core%initialize_update_object(rxn, &
                                                           this%emissions_map(i_cell)%emissions_rates_updates(i_spec))
              call assert_msg(713595717, &
                              this%camp_core%spec_state_id(trim(rxn_spec_name), this%emissions_map(i_cell)%emitted_id(i_spec)), &
                              "missing emitted species: "//trim(rxn_spec_name))
              found = .true.
            end do
          end if
        end select
      end do
      call assert_msg(328925685, &
                      found, &
                      "could not find emisssion reaction for "//trim(this%emissions_map(1)%emitted_species(i_spec)%string))
    end do

  end subroutine load_emission_data

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine update_emission_rates(this, height, pressure, temperature)
    class(boxmodel_interface_t), intent(inout) :: this
    !> box height in cm
    real(kind=dp), intent(in)       :: height(this%n_cells)
    !> pressure in Pa
    real(kind=dp), intent(in)       :: pressure(this%n_cells)
    !> temperature in K
    real(kind=dp), intent(in)       :: temperature(this%n_cells)
    integer(kind=i_kind) :: i_emi, i_cell
    real(kind=dp)  :: rate ! reaction rate

    do i_cell = 1, this%n_cells
      this%emissions_map(i_cell)%current_emissions_rates = 0.0
      do i_emi = 1, this%emissions_map(i_cell)%n_emission
        rate = this%emissions_map(i_cell)%emissions_constraints(i_emi)%val%get(this%time_control%get_current_time())
        ! convert to molec/cm2/s
        select case (this%emissions_map(i_cell)%emissions_constraints(i_emi)%val%value_unit)
        case ("molec/cm2/s", "molec cm-2 s-1") ! do nothing
        case ("molec/m2/s", "molec m-2 s-1")
          rate = rate/1e4
        case ("mol/cm2/s", "mol cm-2 s-1")
          rate = rate*const%avagadro
        case ("mol/m2/s", "mol m-2 s-1", "m-2.s-1.mol")
          rate = rate*const%avagadro/1e4
        case default
          call die_msg(887932393, &
                       "Error in emission rate, unknown unit: "// &
                       this%emissions_map(i_cell)%emissions_constraints(i_emi)%val%value_unit)
        end select

        ! set rate to print in molec/cm2/s
        this%emissions_map(i_cell)%current_emissions_rates(i_emi) = rate

        ! dilute in box height (cm) to get molec/cm3/s
        rate = rate/height(i_cell)

        ! convert molec/cm3/s to ppm/s
        rate = rate*const%univ_gas_const*temperature(i_cell)*1e12/pressure(i_cell)/const%avagadro

        call thread_log%debug("cell: "//trim(to_string(i_cell))//", height="//trim(to_string(height(i_cell)))// &
                              " , pres="//trim(to_string(pressure(i_cell)))// &
                              ", temp="//trim(to_string(temperature(i_cell)))//", emi_rate="//trim(to_string(rate)))
        ! update reaction rate
        call this%emissions_map(i_cell)%emissions_rates_updates(i_emi)%set_rate(rate)
        call this%camp_core%update_data(this%emissions_map(i_cell)%emissions_rates_updates(i_emi), i_cell)
      end do
    end do

  end subroutine update_emission_rates

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine update_deposition_rates(this)
    class(boxmodel_interface_t), intent(inout) :: this

    integer(kind=i_kind) :: i_deposition

    real(kind=dp) :: rate
    ! /TEMPORARY: set to a few hours lifetime
    ! /TODO: do a proper deposition implementation
    rate = 1e-5 !! s-1

    !TODO: reinstate deposition rates update
    ! do i_deposition = 1, this%deposition_map%n_deposition
    !   ! update deposition rate
    !   call this%deposition_map%deposition_rates_updates(i_deposition)%set_rate(rate)
    !   call this%camp_core%update_data(this%deposition_map%deposition_rates_updates(i_deposition))
    ! end do

  end subroutine update_deposition_rates

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine update_forced_species(this, pressure, temperature)
    class(boxmodel_interface_t), intent(inout) :: this
    !> pressure in Pa
    real(kind=dp), intent(in)       :: pressure
    !> temperature in K
    real(kind=dp), intent(in)       :: temperature

    character(len=:), allocatable   :: unit

    integer(kind=i_kind) :: i_spec
    !> constrained concentration, converted to ppm for gas phase species and kg/m3 for aerosol species
    real(kind=dp) :: concentration

    do i_spec = 1, this%forced_species_map%n_species
      concentration = this%forced_species_map%species_constraints(i_spec)%get(this%time_control%get_current_time())
      unit = this%forced_species_map%species_constraints(i_spec)%value_unit()

      ! do conversion
      if (this%forced_species_map%is_aerosol(i_spec)) then
        ! for aeroosl species, convert to kg/m3
        select case (trim(unit))
        case ("kg/m3", "kg m-3")
          ! do nothing
        case ("ug/m3", "ug m-3")
          concentration = concentration*1e-6
        case default
          call die_msg(112896198, &
                       "Error in forced aerosol species concentration, unknown unit: "// &
                       this%forced_species_map%species_constraints(i_spec)%val%value_unit)
        end select
      else
        ! for gas species, convert to ppm
        select case (trim(unit))
        case ("molec/cm3", "molec cm-3")
          concentration = concentration* &
                          const%univ_gas_const*temperature*1e12/pressure/const%avagadro
        case ("molec/m3", "molec m-3")
          concentration = concentration* &
                          const%univ_gas_const*temperature*1e6/pressure/const%avagadro
        case ("mol/cm3", "mol cm-3")
          concentration = concentration* &
                          const%univ_gas_const*temperature*1e12/pressure
        case ("mol/m3", "mol m-3")
          concentration = concentration* &
                          const%univ_gas_const*temperature*1e6/pressure
        case ("ppm")
          ! do nothing
        case ("ppb")
          concentration = concentration*1e-3
        case ("ppt")
          concentration = concentration*1e-6
        case ("ppp")
          concentration = concentration*1e6
        case default
          call die_msg(112896199, &
                       "Error in forced gas species concentration, unknown unit: "// &
                       this%forced_species_map%species_constraints(i_spec)%val%value_unit)
        end select
      end if

      this%camp_state%state_var(this%forced_species_map%forced_species_ind(i_spec)) = concentration

    end do

  end subroutine update_forced_species

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine update_microphysics(this)
    class(boxmodel_interface_t), intent(inout) :: this

    integer(kind=i_kind) :: i_distrib
    real(kind=dp) :: diameter, stdev, current_time

    ! TODO  reinstate microphysics updates
    ! current_time = this%time_control%get_current_time()

    ! do i_distrib = 1, this%microphysics_map%n_distrib
    !   print *, 179069526, "i_distrib =", i_distrib
    !   diameter = this%microphysics_map%diameter_constraints(i_distrib)%get(current_time)
    !   print *, 179069527, "diameter =", diameter
    !   ! convert to m
    !   select case (this%microphysics_map%diameter_constraints(i_distrib)%val%value_unit)
    !    case ("m")
    !     ! do nothing
    !    case ("cm")
    !     diameter = diameter*0.01
    !    case ("mm")
    !     diameter = diameter*0.001
    !    case ("um")
    !     diameter = diameter*1e-6
    !    case ("nm")
    !     diameter = diameter*1e-9
    !    case default
    !     call die_msg(665434887, &
    !       "Error in diameter constraint, unknown unit: "// &
    !       this%microphysics_map%diameter_constraints(i_distrib)%val%value_unit)
    !   end select

    !   stdev = this%microphysics_map%stdev_constraints(i_distrib)%get(current_time)
    !   print *, 179069527, "stdev =", stdev

    !   ! check that the stdev constraint is dimensionless
    !   select case (this%microphysics_map%stdev_constraints(i_distrib)%val%value_unit)
    !    case ("", "dimensionless", "unitless")
    !    case default
    !     call die_msg(665434888, &
    !       "Error in stdev constraint, unknown unit: "// &
    !       this%microphysics_map%stdev_constraints(i_distrib)%val%value_unit)
    !   end select

    !   ! update mean diameter and stdev of distribution
    !   print *, 179069528, "updating section id: ", this%microphysics_map%section_ids(i_distrib)
    !   call this%microphysics_map%update_data_GMD%set_GMD( &
    !     this%microphysics_map%section_ids(i_distrib), &
    !     diameter)

    !   call this%microphysics_map%update_data_GSD%set_GSD( &
    !     this%microphysics_map%section_ids(i_distrib), &
    !     stdev)
    ! end do
    ! call this%camp_core%update_data(this%microphysics_map%update_data_GMD)
    ! call this%camp_core%update_data(this%microphysics_map%update_data_GSD)

  end subroutine update_microphysics

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> Load the sum species found in the mechanism
  subroutine load_sum_species(this)
    class(boxmodel_interface_t), intent(inout) :: this

    type(property_t), POINTER:: property_set, sum_list
    type(chem_spec_data_t), pointer :: chem_spec_data

    integer :: i_spec, j_spec, i_comp, n_sum_species, i_sum_species
    type(string_t), allocatable, dimension(:) :: spec_names
    character(len=:), allocatable :: spec_name, key_name, component_name
    character(len=30) :: sum_list_key_name
    logical :: test

    n_sum_species = 0

    call assert_msg(250575336, &
                    this%camp_core%get_chem_spec_data(chem_spec_data), &
                    "Trying to initialize sum species with uninitialized chem_spec_data")

    spec_names = chem_spec_data%get_spec_names()

    do i_spec = 1, chem_spec_data%size()
      spec_name = spec_names(i_spec)%string
      call assert_msg(287452699, &
                      chem_spec_data%get_property_set(spec_name, property_set), &
                      "Could not get property_set from species "//spec_name)

      key_name = "sum_list"
      if (property_set%get_property_t("sum_list", sum_list)) then
        n_sum_species = n_sum_species + 1
      end if
    end do

    allocate (this%sum_species(n_sum_species))

    i_sum_species = 0
    do i_spec = 1, chem_spec_data%size()
      spec_name = spec_names(i_spec)%string
      call assert_msg(648886641, &
                      chem_spec_data%get_property_set(spec_name, property_set), &
                      "Could not get property_set from species "//spec_name)

      key_name = "sum_list"
      if (property_set%get_property_t("sum_list", sum_list)) then
        i_sum_species = i_sum_species + 1

        this%sum_species(i_sum_species)%species_idx = chem_spec_data%gas_state_id(spec_name)
        this%sum_species(i_sum_species)%n_components = sum_list%size()
        allocate (this%sum_species(i_sum_species)%components_idx(this%sum_species(i_sum_species)%n_components))

        call sum_list%iter_reset()
        do i_comp = 1, this%sum_species(i_sum_species)%n_components

          call assert_msg(391209470, &
                          sum_list%get_string(val=component_name), &
                          "Could not get component of sum species "//spec_name)

          this%sum_species(i_sum_species)%components_idx(i_comp) = chem_spec_data%gas_state_id(component_name)

          call assert_msg(124783157, &
                          this%sum_species(i_sum_species)%components_idx(i_comp) > 0, &
                          "Could not find species "//component_name//" in gas phase")

          call sum_list%iter_next()

        end do
      end if
    end do

  end subroutine load_sum_species

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> update the sum species concentrations by summing their components concentrations
  subroutine update_sum_species_concentrations(this)
    class(boxmodel_interface_t), intent(inout) :: this

    type(sum_species_t) :: sum_species
    integer :: i_sumspec, i_comp

    if (size(this%sum_species) == 0) return

    do i_sumspec = 1, size(this%sum_species)
      sum_species = this%sum_species(i_sumspec)

      this%camp_state%state_var(sum_species%species_idx) = 0.
      if (sum_species%n_components == 0) continue
      do i_comp = 1, sum_species%n_components
        this%camp_state%state_var(sum_species%species_idx) = &
          this%camp_state%state_var(sum_species%species_idx) + &
          this%camp_state%state_var(sum_species%components_idx(i_comp))
      end do
    end do

  end subroutine update_sum_species_concentrations

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Print the -camp data
  subroutine do_print(this)

    !> PartMC-camp <-> MONARCH interface
    class(boxmodel_interface_t) :: this

    call this%camp_core%print()

  end subroutine do_print

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Finalize the interface
  elemental subroutine finalize(this)

    !> PartMC-camp <-> MONARCH interface
    type(boxmodel_interface_t), intent(inout) :: this

    if (associated(this%camp_core)) &
      deallocate (this%camp_core)
    if (associated(this%camp_state)) &
      deallocate (this%camp_state)
    if (allocated(this%init_conc_camp_id)) &
      deallocate (this%init_conc_camp_id)
    if (allocated(this%init_conc)) &
      deallocate (this%init_conc)
    if (associated(this%species_map_data)) &
      deallocate (this%species_map_data)
    if (associated(this%init_conc_data)) &
      deallocate (this%init_conc_data)
    if (associated(this%property_set)) &
      deallocate (this%property_set)

  end subroutine finalize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  integer(kind=i_kind) function pack_size(this, comm)
    !> Reaction update data
    class(boxmodel_interface_t), intent(in) :: this
    !> MPI communicator
    integer, intent(in), optional :: comm
    integer :: constraint_type_id, i_sumspec, i_init_conc, i_cell
    logical :: allocated_sum_species

#ifdef CAMP_USE_MPI
    integer :: l_comm

    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if

    pack_size = this%camp_core%pack_size(l_comm) + &
                camp_mpi_pack_size_logical(this%aerosol_flag, l_comm) + &
                this%time_control%pack_size(l_comm) + &
                camp_mpi_pack_size_integer_array(this%init_conc_camp_id, l_comm) + &
                camp_mpi_pack_size_integer(this%gas_phase_water_id, l_comm) + &
                camp_mpi_pack_size_integer(this%gas_phase_water_dimer_id, l_comm) + &
                camp_mpi_pack_size_string(this%photolysis_model, l_comm)

    pack_size = pack_size + &
                camp_mpi_pack_size_integer(size(this%init_conc), l_comm)

    if (size(this%init_conc) > 0) then
      do i_init_conc = 1, size(this%init_conc)
        pack_size = pack_size + this%init_conc(i_init_conc)%pack_size(l_comm)
      end do
    end if

    pack_size = pack_size + camp_mpi_pack_size_integer(this%n_cells)

    do i_cell = 1, this%n_cells
      constraint_type_id = this%longitude_constraint(i_cell)%val%constraint_type_id()
      pack_size = pack_size + &
                  camp_mpi_pack_size_integer(constraint_type_id, l_comm) + &
                  this%longitude_constraint(i_cell)%val%pack_size(l_comm)

      constraint_type_id = this%latitude_constraint(i_cell)%val%constraint_type_id()
      pack_size = pack_size + &
                  camp_mpi_pack_size_integer(constraint_type_id, l_comm) + &
                  this%latitude_constraint(i_cell)%val%pack_size(l_comm)

      constraint_type_id = this%altitude_constraint(i_cell)%val%constraint_type_id()
      pack_size = pack_size + &
                  camp_mpi_pack_size_integer(constraint_type_id, l_comm) + &
                  this%altitude_constraint(i_cell)%val%pack_size(l_comm)

      constraint_type_id = this%pressure_constraint(i_cell)%val%constraint_type_id()
      pack_size = pack_size + &
                  camp_mpi_pack_size_integer(constraint_type_id, l_comm) + &
                  this%pressure_constraint(i_cell)%val%pack_size(l_comm)

      constraint_type_id = this%temperature_constraint(i_cell)%val%constraint_type_id()
      pack_size = pack_size + &
                  camp_mpi_pack_size_integer(constraint_type_id, l_comm) + &
                  this%temperature_constraint(i_cell)%val%pack_size(l_comm)

      constraint_type_id = this%humidity_constraint(i_cell)%val%constraint_type_id()
      pack_size = pack_size + &
                  camp_mpi_pack_size_integer(constraint_type_id, l_comm) + &
                  this%humidity_constraint(i_cell)%val%pack_size(l_comm)

      constraint_type_id = this%height_constraint(i_cell)%val%constraint_type_id()
      pack_size = pack_size + &
                  camp_mpi_pack_size_integer(constraint_type_id, l_comm) + &
                  this%height_constraint(i_cell)%val%pack_size(l_comm)
    end do

    pack_size = pack_size + &
                camp_mpi_pack_size_logical(this%constrained_sza, l_comm)

    do i_cell = 1, this%n_cells
      if (this%constrained_sza) then
        constraint_type_id = this%sza_constraint(i_cell)%val%constraint_type_id()
        pack_size = pack_size + &
                    camp_mpi_pack_size_integer(constraint_type_id, l_comm) + &
                    this%sza_constraint(i_cell)%val%pack_size(l_comm)
      end if
    end do

    do i_cell = 1, this%n_cells
      pack_size = pack_size + &
                  this%emissions_map(i_cell)%pack_size(l_comm)
    end do

    allocated_sum_species = allocated(this%sum_species)
    pack_size = pack_size + &
                camp_mpi_pack_size_logical(allocated_sum_species)

    if (allocated_sum_species) then
      pack_size = pack_size + &
                  camp_mpi_pack_size_integer(size(this%sum_species), l_comm)

      if (size(this%sum_species) > 0) then
        do i_sumspec = 1, size(this%sum_species)
          pack_size = pack_size + &
                      this%sum_species(i_sumspec)%pack_size(l_comm)
        end do
      end if
    end if

    pack_size = pack_size + &
                this%microphysics_map%pack_size(l_comm)

    pack_size = pack_size + &
                this%forced_species_map%pack_size(l_comm)

    pack_size = pack_size + &
                this%photolysis_map%pack_size(l_comm)

    pack_size = pack_size + &
                this%deposition_map%pack_size(l_comm)

#else
    pack_size = 0

#endif

  end function pack_size

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine bin_pack(this, buffer, pos, comm)
    class(boxmodel_interface_t), intent(in) :: this
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos
    !> MPI communicator
    integer, intent(in), optional :: comm
    logical :: allocated_sum_species

#ifdef CAMP_USE_MPI
    integer(kind=i_kind) :: prev_position, l_comm, i_sumspec, i_init_conc, i_cell

    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if

    prev_position = pos
    call this%camp_core%bin_pack(buffer, pos, l_comm)
    call camp_mpi_pack_logical(buffer, pos, this%aerosol_flag, l_comm)
    call this%time_control%bin_pack(buffer, pos, l_comm)
    call camp_mpi_pack_integer_array(buffer, pos, this%init_conc_camp_id, l_comm)
    call camp_mpi_pack_integer(buffer, pos, this%gas_phase_water_id, l_comm)
    call camp_mpi_pack_integer(buffer, pos, this%gas_phase_water_dimer_id, l_comm)

    call camp_mpi_pack_string(buffer, pos, this%photolysis_model, l_comm)

    call camp_mpi_pack_integer(buffer, pos, size(this%init_conc), l_comm)

    if (size(this%init_conc) > 0) then
      do i_init_conc = 1, size(this%init_conc)
        call this%init_conc(i_init_conc)%bin_pack(buffer, pos, l_comm)
      end do
    end if

    call camp_mpi_pack_integer(buffer, pos, this%n_cells, l_comm)

    do i_cell = 1, this%n_cells
      call camp_mpi_pack_integer(buffer, pos, this%longitude_constraint(i_cell)%val%constraint_type_id(), l_comm)
      call this%longitude_constraint(i_cell)%val%bin_pack(buffer, pos, l_comm)

      call camp_mpi_pack_integer(buffer, pos, this%latitude_constraint(i_cell)%val%constraint_type_id(), l_comm)
      call this%latitude_constraint(i_cell)%val%bin_pack(buffer, pos, l_comm)

      call camp_mpi_pack_integer(buffer, pos, this%altitude_constraint(i_cell)%val%constraint_type_id(), l_comm)
      call this%altitude_constraint(i_cell)%val%bin_pack(buffer, pos, l_comm)

      call camp_mpi_pack_integer(buffer, pos, this%pressure_constraint(i_cell)%val%constraint_type_id(), l_comm)
      call this%pressure_constraint(i_cell)%val%bin_pack(buffer, pos, l_comm)

      call camp_mpi_pack_integer(buffer, pos, this%temperature_constraint(i_cell)%val%constraint_type_id(), l_comm)
      call this%temperature_constraint(i_cell)%val%bin_pack(buffer, pos, l_comm)

      call camp_mpi_pack_integer(buffer, pos, this%humidity_constraint(i_cell)%val%constraint_type_id(), l_comm)
      call this%humidity_constraint(i_cell)%val%bin_pack(buffer, pos, l_comm)

      call camp_mpi_pack_integer(buffer, pos, this%height_constraint(i_cell)%val%constraint_type_id(), l_comm)
      call this%height_constraint(i_cell)%val%bin_pack(buffer, pos, l_comm)
    end do

    call camp_mpi_pack_logical(buffer, pos, this%constrained_sza, l_comm)

    if (this%constrained_sza) then
      do i_cell = 1, this%n_cells
        call camp_mpi_pack_integer(buffer, pos, this%sza_constraint(i_cell)%val%constraint_type_id(), l_comm)
        call this%sza_constraint(i_cell)%val%bin_pack(buffer, pos, l_comm)
      end do
    end if

    do i_cell = 1, this%n_cells
      call this%emissions_map(i_cell)%bin_pack(buffer, pos, l_comm)
    end do

    allocated_sum_species = allocated(this%sum_species)
    call camp_mpi_pack_logical(buffer, pos, allocated_sum_species, l_comm)

    if (allocated_sum_species) then
      call camp_mpi_pack_integer(buffer, pos, size(this%sum_species), l_comm)
      if (size(this%sum_species) > 0) then
        do i_sumspec = 1, size(this%sum_species)
          call this%sum_species(i_sumspec)%bin_pack(buffer, pos, l_comm)
        end do
      end if
    end if

    call this%microphysics_map%bin_pack(buffer, pos, l_comm)

    call this%forced_species_map%bin_pack(buffer, pos, l_comm)

    call this%photolysis_map%bin_pack(buffer, pos, l_comm)

    call this%deposition_map%bin_pack(buffer, pos, l_comm)

    call assert(179073293, &
                pos - prev_position <= this%pack_size(l_comm))
#endif

  end subroutine bin_pack

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine bin_unpack(this, buffer, pos, comm)
    class(boxmodel_interface_t), intent(inout) :: this
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos
    !> MPI communicator
    integer, intent(in), optional :: comm

#ifdef CAMP_USE_MPI
    integer(kind=i_kind) :: prev_position, l_comm, constraint_type_id, i_sumspec, n_sumspec
    integer(kind=i_kind) :: i_init_conc, n_init_conc, i_cell
    logical :: allocated_sum_species
    character(len=15) :: temp_string

    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if

    this%camp_core => camp_core_t()
    this%time_control => time_control_t()

    prev_position = pos
    call this%camp_core%bin_unpack(buffer, pos, l_comm)
    call camp_mpi_unpack_logical(buffer, pos, this%aerosol_flag, l_comm)
    call this%time_control%bin_unpack(buffer, pos, l_comm)
    call camp_mpi_unpack_integer_array(buffer, pos, this%init_conc_camp_id, l_comm)
    call camp_mpi_unpack_integer(buffer, pos, this%gas_phase_water_id, l_comm)
    call camp_mpi_unpack_integer(buffer, pos, this%gas_phase_water_dimer_id, l_comm)

    call camp_mpi_unpack_string(buffer, pos, temp_string, l_comm)
    this%photolysis_model = trim(temp_string)

    call camp_mpi_unpack_integer(buffer, pos, n_init_conc, l_comm)

    allocate (this%init_conc(n_init_conc))

    if (n_init_conc > 0) then
      do i_init_conc = 1, n_init_conc
        call this%init_conc(i_init_conc)%bin_unpack(buffer, pos, l_comm)
      end do
    end if

    call camp_mpi_unpack_integer(buffer, pos, this%n_cells, l_comm)
    allocate (this%longitude_constraint(this%n_cells))
    allocate (this%latitude_constraint(this%n_cells))
    allocate (this%altitude_constraint(this%n_cells))
    allocate (this%pressure_constraint(this%n_cells))
    allocate (this%temperature_constraint(this%n_cells))
    allocate (this%humidity_constraint(this%n_cells))
    allocate (this%height_constraint(this%n_cells))

    do i_cell = 1, this%n_cells
      call camp_mpi_unpack_integer(buffer, pos, constraint_type_id, l_comm)
      call constraint_from_type_id(this%longitude_constraint(i_cell), constraint_type_id)
      call this%longitude_constraint(i_cell)%val%bin_unpack(buffer, pos, l_comm)

      call camp_mpi_unpack_integer(buffer, pos, constraint_type_id, l_comm)
      call constraint_from_type_id(this%latitude_constraint(i_cell), constraint_type_id)
      call this%latitude_constraint(i_cell)%val%bin_unpack(buffer, pos, l_comm)

      call camp_mpi_unpack_integer(buffer, pos, constraint_type_id, l_comm)
      call constraint_from_type_id(this%altitude_constraint(i_cell), constraint_type_id)
      call this%altitude_constraint(i_cell)%val%bin_unpack(buffer, pos, l_comm)

      call camp_mpi_unpack_integer(buffer, pos, constraint_type_id, l_comm)
      call constraint_from_type_id(this%pressure_constraint(i_cell), constraint_type_id)
      call this%pressure_constraint(i_cell)%val%bin_unpack(buffer, pos, l_comm)

      call camp_mpi_unpack_integer(buffer, pos, constraint_type_id, l_comm)
      call constraint_from_type_id(this%temperature_constraint(i_cell), constraint_type_id)
      call this%temperature_constraint(i_cell)%val%bin_unpack(buffer, pos, l_comm)

      call camp_mpi_unpack_integer(buffer, pos, constraint_type_id, l_comm)
      call constraint_from_type_id(this%humidity_constraint(i_cell), constraint_type_id)
      call this%humidity_constraint(i_cell)%val%bin_unpack(buffer, pos, l_comm)

      call camp_mpi_unpack_integer(buffer, pos, constraint_type_id, l_comm)
      call constraint_from_type_id(this%height_constraint(i_cell), constraint_type_id)
      call this%height_constraint(i_cell)%val%bin_unpack(buffer, pos, l_comm)
    end do

    call camp_mpi_unpack_logical(buffer, pos, this%constrained_sza, l_comm)

    if (this%constrained_sza) then
      allocate (this%sza_constraint(this%n_cells))
      do i_cell = 1, this%n_cells
        call camp_mpi_unpack_integer(buffer, pos, constraint_type_id, l_comm)
        call constraint_from_type_id(this%sza_constraint(i_cell), constraint_type_id)
        call this%sza_constraint(i_cell)%val%bin_unpack(buffer, pos, l_comm)
      end do
    end if

    do i_cell = 1, this%n_cells
      call this%emissions_map(i_cell)%bin_unpack(buffer, pos, l_comm)
    end do

    call camp_mpi_unpack_logical(buffer, pos, allocated_sum_species, l_comm)

    if (allocated_sum_species) then
      call camp_mpi_unpack_integer(buffer, pos, n_sumspec, l_comm)

      allocate (this%sum_species(n_sumspec))
      if (n_sumspec > 0) then
        do i_sumspec = 1, n_sumspec
          call this%sum_species(i_sumspec)%bin_unpack(buffer, pos, l_comm)
        end do
      end if
    end if

    call this%microphysics_map%bin_unpack(buffer, pos, l_comm)

    call this%forced_species_map%bin_unpack(buffer, pos, l_comm)

    call this%photolysis_map%bin_unpack(buffer, pos, l_comm)

    call this%deposition_map%bin_unpack(buffer, pos, l_comm)

    call assert(379116695, &
                pos - prev_position <= this%pack_size(l_comm))

#endif

  end subroutine bin_unpack

end module camp_boxmodel_interface
