! Copyright (C) 2017-2018 Matt Dawson
! Licensed under the GNU General Public License version 2 or (at your
! option) any later version. See the file COPYING for details.

!> \file
!> The pmc_test_wet_deposition program

!> Test of wet_deposition reaction module
program pmc_test_wet_deposition

  use pmc_aero_rep_data
  use pmc_chem_spec_data
  use pmc_mechanism_data
  use pmc_phlex_core
  use pmc_phlex_state
  use pmc_rxn_data
  use pmc_rxn_wet_deposition
  use pmc_rxn_factory
  use pmc_util,                         only: i_kind, dp, assert, &
                                              almost_equal, string_t, &
                                              warn_msg
#ifdef PMC_USE_JSON
  use json_module
#endif
  use pmc_mpi

  use iso_c_binding

  implicit none

  ! Number of timesteps to output in mechanisms
  integer(kind=i_kind) :: NUM_TIME_STEP = 100

  ! initialize mpi
  call pmc_mpi_init()

  if (run_wet_deposition_tests()) then
    if (pmc_mpi_rank().eq.0) write(*,*) &
          "Wet Deposition reaction tests - PASS"
  else
    if (pmc_mpi_rank().eq.0) write(*,*) &
          "Wet Deposition reaction tests - FAIL"
  end if

  ! finalize mpi
  call pmc_mpi_finalize()

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Run all pmc_rxn_wet_deposition tests
  logical function run_wet_deposition_tests() result(passed)

    use pmc_phlex_solver_data

    type(phlex_solver_data_t), pointer :: phlex_solver_data

    phlex_solver_data => phlex_solver_data_t()

    if (phlex_solver_data%is_solver_available()) then
      passed = run_wet_deposition_test()
    else
      call warn_msg(280044966, "No solver available")
      passed = .true.
    end if

    deallocate(phlex_solver_data)

  end function run_wet_deposition_tests

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Solve a mechanism of wet deposition reactions
  !!
  !! The mechanism is of the form:
  !!
  !!   A -k->
  !!   B -k->
  !!
  !! where k is the wet_deposition reaction rate constant.
  logical function run_wet_deposition_test()

    use pmc_constants

    type(phlex_core_t), pointer :: phlex_core
    type(phlex_state_t), pointer :: phlex_state
    character(len=:), allocatable :: input_file_path, key, str_val
    type(string_t), allocatable, dimension(:) :: output_file_path

    real(kind=dp), dimension(0:NUM_TIME_STEP, 8) :: model_conc, true_conc
    integer(kind=i_kind) :: idx_1RA, idx_1RB, idx_1CB, idx_1CC
    integer(kind=i_kind) :: idx_2RA, idx_2RB, idx_2CB, idx_2CC
    integer(kind=i_kind) :: i_time, i_spec, i_rxn, i_rxn_rain, &
                            i_mech_rxn_rain
    real(kind=dp) :: time_step, time, k_rain, k_cloud, temp, pressure, &
                     rate_rain
    class(rxn_data_t), pointer :: rxn
    class(aero_rep_data_t), pointer :: aero_rep
    type(string_t), allocatable :: unique_names(:)
    character(len=:), allocatable :: spec_name, phase_name, rep_name
#ifdef PMC_USE_MPI
    character, allocatable :: buffer(:), buffer_copy(:)
    integer(kind=i_kind) :: pack_size, pos, i_elem, results
#endif

    ! For setting rates
    type(mechanism_data_t), pointer :: mechanism
    type(rxn_factory_t) :: rxn_factory
    type(rxn_update_data_wet_deposition_rate_t) :: rate_update

    run_wet_deposition_test = .true.

    ! Set the rate constants (for calculating the true value)
    temp = 272.5d0
    pressure = 101253.3d0
    rate_rain = 0.954d0
    k_rain = rate_rain
    k_cloud = 1.0d-02 * 12.3d0

    ! Set output time step (s)
    time_step = 1.0

#ifdef PMC_USE_MPI
    ! Load the model data on the root process and pass it to process 1 for solving
    if (pmc_mpi_rank().eq.0) then
#endif

      ! Get the wet_deposition reaction mechanism json file
      input_file_path = 'test_wet_deposition_config.json'

      ! Construct a phlex_core variable
      phlex_core => phlex_core_t(input_file_path)

      deallocate(input_file_path)

      ! Initialize the model
      call phlex_core%initialize()

      ! Find the mechanism
      key = "wet deposition"
      call assert(868882379, phlex_core%get_mechanism(key, mechanism))

      ! Find the A wet_deposition reaction
      key = "rxn id"
      i_rxn_rain = 342
      i_mech_rxn_rain = 0
      do i_rxn = 1, mechanism%size()
        rxn => mechanism%get_rxn(i_rxn)
        if (rxn%property_set%get_string(key, str_val)) then
          if (trim(str_val).eq."rxn rain") then
            i_mech_rxn_rain = i_rxn
            select type (rxn_loss => rxn)
              class is (rxn_wet_deposition_t)
                call rxn_loss%set_rxn_id(i_rxn_rain)
            end select
          end if
        end if
      end do
      call assert(300573108, i_mech_rxn_rain.eq.1)

      ! Find species in the first aerosol representation
      rep_name = "my first particle"
      call assert(229751752, phlex_core%get_aero_rep(rep_name, aero_rep))

      ! Get species indices
      phase_name = "rain"
      spec_name = "A"
      unique_names = aero_rep%unique_names( phase_name = phase_name, &
                                            spec_name = spec_name )
      call assert(635673803, size(unique_names).eq.1)
      idx_1RA = aero_rep%spec_state_id(unique_names(1)%string);

      spec_name = "B"
      unique_names = aero_rep%unique_names( phase_name = phase_name, &
                                            spec_name = spec_name )
      call assert(565853391, size(unique_names).eq.1)
      idx_1RB = aero_rep%spec_state_id(unique_names(1)%string);

      phase_name = "cloud"
      spec_name = "B"
      unique_names = aero_rep%unique_names( phase_name = phase_name, &
                                            spec_name = spec_name )
      call assert(957288212, size(unique_names).eq.1)
      idx_1CB = aero_rep%spec_state_id(unique_names(1)%string);

      spec_name = "C"
      unique_names = aero_rep%unique_names( phase_name = phase_name, &
                                            spec_name = spec_name )
      call assert(388978941, size(unique_names).eq.1)
      idx_1CC = aero_rep%spec_state_id(unique_names(1)%string);

      ! Find species in the second aerosol representation
      rep_name = "my second particle"
      call assert(440099954, phlex_core%get_aero_rep(rep_name, aero_rep))

      phase_name = "rain"
      spec_name = "A"
      unique_names = aero_rep%unique_names( phase_name = phase_name, &
                                            spec_name = spec_name )
      call assert(271848584, size(unique_names).eq.1)
      idx_2RA = aero_rep%spec_state_id(unique_names(1)%string);

      spec_name = "B"
      unique_names = aero_rep%unique_names( phase_name = phase_name, &
                                            spec_name = spec_name )
      call assert(384166929, size(unique_names).eq.1)
      idx_2RB = aero_rep%spec_state_id(unique_names(1)%string);

      phase_name = "cloud"
      spec_name = "B"
      unique_names = aero_rep%unique_names( phase_name = phase_name, &
                                            spec_name = spec_name )
      call assert(778960523, size(unique_names).eq.1)
      idx_2CB = aero_rep%spec_state_id(unique_names(1)%string);

      spec_name = "C"
      unique_names = aero_rep%unique_names( phase_name = phase_name, &
                                            spec_name = spec_name )
      call assert(273754118, size(unique_names).eq.1)
      idx_2CC = aero_rep%spec_state_id(unique_names(1)%string);

      ! Make sure the expected species are in the model
      call assert(100238441, idx_1RA.gt.0)
      call assert(830081536, idx_1RB.gt.0)
      call assert(659924632, idx_1CB.gt.0)
      call assert(489767728, idx_1CC.gt.0)
      call assert(319610824, idx_2RA.gt.0)
      call assert(149453920, idx_2RB.gt.0)
      call assert(879297015, idx_2CB.gt.0)
      call assert(709140111, idx_2CC.gt.0)

#ifdef PMC_USE_MPI
      ! pack the phlex core
      pack_size = phlex_core%pack_size()
      allocate(buffer(pack_size))
      pos = 0
      call phlex_core%bin_pack(buffer, pos)
      call assert(768884204, pos.eq.pack_size)
    end if

    ! broadcast the species ids
    call pmc_mpi_bcast_integer(idx_1RA)
    call pmc_mpi_bcast_integer(idx_1RB)
    call pmc_mpi_bcast_integer(idx_1CB)
    call pmc_mpi_bcast_integer(idx_1CC)
    call pmc_mpi_bcast_integer(idx_2RA)
    call pmc_mpi_bcast_integer(idx_2RB)
    call pmc_mpi_bcast_integer(idx_2CB)
    call pmc_mpi_bcast_integer(idx_2CC)
    call pmc_mpi_bcast_integer(i_rxn_rain)

    ! broadcast the buffer size
    call pmc_mpi_bcast_integer(pack_size)

    if (pmc_mpi_rank().eq.1) then
      ! allocate the buffer to receive data
      allocate(buffer(pack_size))
    end if

    ! broadcast the data
    call pmc_mpi_bcast_packed(buffer)

    if (pmc_mpi_rank().eq.1) then
      ! unpack the data
      phlex_core => phlex_core_t()
      pos = 0
      call phlex_core%bin_unpack(buffer, pos)
      call assert(413229770, pos.eq.pack_size)
      allocate(buffer_copy(pack_size))
      pos = 0
      call phlex_core%bin_pack(buffer_copy, pos)
      call assert(243072866, pos.eq.pack_size)
      do i_elem = 1, pack_size
        call assert_msg(809928898, buffer(i_elem).eq.buffer_copy(i_elem), &
                "Mismatch in element: "//trim(to_string(i_elem)))
      end do

      ! solve and evaluate results on process 1
#endif

      ! Initialize the solver
      call phlex_core%solver_initialize()

      ! Get a model state variable
      phlex_state => phlex_core%new_state()

      ! Set the environmental conditions
      phlex_state%env_state%temp = temp
      phlex_state%env_state%pressure = pressure
      call phlex_state%update_env_state()

      ! Save the initial concentrations
      true_conc(0,idx_1RA) = 1.0
      true_conc(0,idx_1RB) = 2.2
      true_conc(0,idx_1CB) = 1.7
      true_conc(0,idx_1CC) = 2.4
      true_conc(0,idx_2RA) = 2.5
      true_conc(0,idx_2RB) = 0.7
      true_conc(0,idx_2CB) = 1.6
      true_conc(0,idx_2CC) = 1.9
      model_conc(0,:) = true_conc(0,:)

      ! Set the initial concentrations in the model
      phlex_state%state_var(:) = model_conc(0,:)

      ! Set the rain rxn rate
      call rxn_factory%initialize_update_data(rate_update)
      call rate_update%set_rate(i_rxn_rain, rate_rain)
      call phlex_core%update_rxn_data(rate_update)

      ! Integrate the mechanism
      do i_time = 1, NUM_TIME_STEP

        ! Get the modeled conc
        call phlex_core%solve(phlex_state, time_step)
        model_conc(i_time,:) = phlex_state%state_var(:)

        ! Get the analytic conc
        time = i_time * time_step
        true_conc(i_time,idx_1RA) = true_conc(0,idx_1RA) * exp(-(k_rain)*time)
        true_conc(i_time,idx_1RB) = true_conc(0,idx_1RB) * exp(-(k_rain)*time)
        true_conc(i_time,idx_1CB) = true_conc(0,idx_1CB) * exp(-(k_cloud)*time)
        true_conc(i_time,idx_1CC) = true_conc(0,idx_1CC) * exp(-(k_cloud)*time)
        true_conc(i_time,idx_2RA) = true_conc(0,idx_2RA) * exp(-(k_rain)*time)
        true_conc(i_time,idx_2RB) = true_conc(0,idx_2RB) * exp(-(k_rain)*time)
        true_conc(i_time,idx_2CB) = true_conc(0,idx_2CB) * exp(-(k_cloud)*time)
        true_conc(i_time,idx_2CC) = true_conc(0,idx_2CC) * exp(-(k_cloud)*time)

      end do

      ! Save the results
      open(unit=7, file="out/wet_deposition_results.txt", status="replace", &
              action="write")
      do i_time = 0, NUM_TIME_STEP
        write(7,*) i_time*time_step, &
              ' ', true_conc(i_time, idx_1RA),' ', model_conc(i_time, idx_1RA), &
              ' ', true_conc(i_time, idx_1RB),' ', model_conc(i_time, idx_1RB), &
              ' ', true_conc(i_time, idx_1CB),' ', model_conc(i_time, idx_1CB), &
              ' ', true_conc(i_time, idx_1CC),' ', model_conc(i_time, idx_1CC), &
              ' ', true_conc(i_time, idx_2RA),' ', model_conc(i_time, idx_2RA), &
              ' ', true_conc(i_time, idx_2RB),' ', model_conc(i_time, idx_2RB), &
              ' ', true_conc(i_time, idx_2CB),' ', model_conc(i_time, idx_2CB), &
              ' ', true_conc(i_time, idx_2CC),' ', model_conc(i_time, idx_2CC)
      end do
      close(7)

      ! Analyze the results
      do i_time = 1, NUM_TIME_STEP
        do i_spec = 1, size(model_conc, 2)
          call assert_msg(281211082, &
            almost_equal(model_conc(i_time, i_spec), &
            true_conc(i_time, i_spec), real(1.0e-2, kind=dp)).or. &
            (model_conc(i_time, i_spec).lt.1e-5*model_conc(1, i_spec).and. &
            true_conc(i_time, i_spec).lt.1e-5*true_conc(1, i_spec)), &
            "time: "//trim(to_string(i_time))//"; species: "// &
            trim(to_string(i_spec))//"; mod: "// &
            trim(to_string(model_conc(i_time, i_spec)))//"; true: "// &
            trim(to_string(true_conc(i_time, i_spec))))
        end do
      end do

      deallocate(phlex_state)

#ifdef PMC_USE_MPI
      ! convert the results to an integer
      if (run_wet_deposition_test) then
        results = 0
      else
        results = 1
      end if
    end if
    
    ! Send the results back to the primary process
    call pmc_mpi_transfer_integer(results, results, 1, 0)

    ! convert the results back to a logical value
    if (pmc_mpi_rank().eq.0) then
      if (results.eq.0) then
        run_wet_deposition_test = .true.
      else
        run_wet_deposition_test = .false.
      end if
    end if

    deallocate(buffer)
#endif

    deallocate(phlex_core)

  end function run_wet_deposition_test

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end program pmc_test_wet_deposition