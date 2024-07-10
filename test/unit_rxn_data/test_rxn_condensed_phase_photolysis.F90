! Copyright (C) 2021 Barcelona Supercomputing Center and University of
! Illinois at Urbana-Champaign
! SPDX-License-Identifier: MIT

!> \file
!> The camp_test_condensed_phase_photolysis program

!> Test of condensed_phase_photolysis reaction module
program camp_test_condensed_phase_photolysis

  use iso_c_binding

  use camp_util,                         only: i_kind, dp, assert, &
                                              almost_equal, string_t, &
                                              warn_msg
  use camp_camp_core
  use camp_mechanism_data
  use camp_camp_state
  use camp_aero_rep_data
  use camp_aero_rep_factory
  use camp_aero_rep_single_particle
  use camp_aero_rep_modal_binned_mass
  use camp_rxn_data
  use camp_rxn_condensed_phase_photolysis
  use camp_solver_stats
#ifdef CAMP_USE_JSON
  use json_module
#endif
  use camp_mpi

  implicit none

  ! Number of timesteps to output in mechanisms
  integer(kind=i_kind) :: NUM_TIME_STEP = 100

  ! initialize mpi
  call camp_mpi_init()

  if (run_condensed_phase_photolysis_tests()) then
    if (camp_mpi_rank().eq.0) write(*,*) "Condensed-phase photolysis reaction tests - PASS"
  else
    if (camp_mpi_rank().eq.0) write(*,*) "Condensed-phase photolysis reaction tests - FAIL"
    stop 3
  end if

  ! finalize mpi
  call camp_mpi_finalize()

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Run all camp_chem_mech_solver tests
  logical function run_condensed_phase_photolysis_tests() result(passed)

    use camp_camp_solver_data

    type(camp_solver_data_t), pointer :: camp_solver_data

    camp_solver_data => camp_solver_data_t()

    if (camp_solver_data%is_solver_available()) then
      passed = run_condensed_phase_photolysis_test()
    else
      call warn_msg(187891224, "No solver available")
      passed = .true.
    end if

    deallocate(camp_solver_data)

  end function run_condensed_phase_photolysis_tests

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Solve a mechanism consisting of two sets of condensed-phase reactions
  !!
  !! The mechanism is of the form:
  !!
  !!   A + hv -k1-> B + C
  !!
  !! where k1 is a condensed-phase photolysis reaction rate constant.
  !! One set of reactions is done with units
  !! of 'M' and one is with 'kg/m3'.
  !!
  !! One of two scenarios is tested, depending on the passed integer:
  !! (1) single-particle aerosol representation and fixed water concentration
  !! (2) modal aerosol representation and ZSR-calculated water concentration
  logical function run_condensed_phase_photolysis_test()

    use camp_constants

    type(camp_core_t), pointer :: camp_core
    type(camp_state_t), pointer :: camp_state
    character(len=:), allocatable :: input_file_path, key, idx_prefix, str_val
    type(string_t), allocatable, dimension(:) :: output_file_path

    class(aero_rep_data_t), pointer :: aero_rep_ptr
    integer(kind=i_kind) :: num_state_var, state_size
    real(kind=dp), allocatable, dimension(:,:) :: model_conc, true_conc
    integer(kind=i_kind) :: idx_A, idx_B, idx_C, idx_H2O, &
            idx_aq_phase, i_time, i_spec, i_rxn
    integer(kind=i_kind) :: i_rxn_photo_A, i_rxn_photo_B
    real(kind=dp) :: time_step, time, conc_water, MW_A, MW_B, MW_C, &
            j1, j2, temp, pressure, j2_scaling
    class(rxn_data_t), pointer :: rxn
#ifdef CAMP_USE_MPI
    character, allocatable :: buffer(:), buffer_copy(:)
    integer(kind=i_kind) :: pack_size, pos, i_elem, results
#endif

    type(solver_stats_t), target :: solver_stats

    integer(kind=i_kind) :: i_sect_unused, i_sect_the_mode
    type(aero_rep_factory_t) :: aero_rep_factory
    type(aero_rep_update_data_modal_binned_mass_GMD_t) :: update_data_GMD
    type(aero_rep_update_data_modal_binned_mass_GSD_t) :: update_data_GSD
    
    ! rate setting
    type(mechanism_data_t), pointer :: mechanism
    type(rxn_update_data_condensed_phase_photolysis_t) :: rate_update_A, rate_update_B

    run_condensed_phase_photolysis_test = .true.

    ! Allocate space for the results
    num_state_var = 9 * 4 ! particles * species
    allocate(model_conc(0:NUM_TIME_STEP, num_state_var))
    allocate(true_conc(0:NUM_TIME_STEP, num_state_var))

    ! Set the environmental and aerosol test conditions
    temp = 272.5d0              ! temperature (K)
    pressure = 101253.3d0       ! pressure (Pa)

    ! Set the rate constants (for calculating the true values)
    conc_water = 2.3d0
    MW_A = 0.1572
    MW_B = 0.0219
    MW_C = 0.2049
    j1 = 0.05
    j2_scaling = 12.3
    j2 = 0.15

    ! Set output time step (s)
    time_step = 1.0d0

#ifdef CAMP_USE_MPI
    ! Load the model data on root process and pass it to process 1 for solving
    if (camp_mpi_rank().eq.0) then
#endif

      ! Get the condensed_phase_photolysis reaction mechanism json file
      input_file_path = 'test_condensed_phase_photolysis_config.json'

      ! Construct a camp_core variable
      camp_core => camp_core_t(input_file_path)

      deallocate(input_file_path)

      ! Initialize the model
      call camp_core%initialize()

      ! Find the aerosol representation
      key = "my aero rep 2"
      call assert(421062613, camp_core%get_aero_rep(key, aero_rep_ptr))

      ! Find the mechanism
      key = "condensed phase photolysis"
      call assert(214488774, camp_core%get_mechanism(key, mechanism))

      ! Find the reaction indices
      key = "photo id"
      i_rxn_photo_A = 0
      i_rxn_photo_B = 0
      do i_rxn = 1, mechanism%size()
        rxn => mechanism%get_rxn(i_rxn)
        if (rxn%property_set%get_string(key, str_val)) then
          if (trim(str_val).eq."photo A") then
            i_rxn_photo_A = i_rxn
            select type (rxn_photo => rxn)
              class is (rxn_condensed_phase_photolysis_t)
                call camp_core%initialize_update_object(rxn_photo,&
                                                        rate_update_A)
            end select
          end if
          if (trim(str_val).eq."photo B") then
            i_rxn_photo_B = i_rxn
            select type (rxn_photo => rxn)
              class is (rxn_condensed_phase_photolysis_t)
                call camp_core%initialize_update_object(rxn_photo,&
                                                        rate_update_B)
            end select
          end if
        end if
      end do
      call assert(850083242, i_rxn_photo_A.eq.1)
      call assert(262215272, i_rxn_photo_B.eq.2)

      ! Get species indices
      idx_prefix = "P2."
      key = idx_prefix//"one layer.aqueous aerosol.A"
      idx_A = aero_rep_ptr%spec_state_id(key);
      key = idx_prefix//"one layer.aqueous aerosol.B"
      idx_B = aero_rep_ptr%spec_state_id(key);
      key = idx_prefix//"one layer.aqueous aerosol.C"
      idx_C = aero_rep_ptr%spec_state_id(key);
      key = idx_prefix//"one layer.aqueous aerosol.H2O_aq"
      idx_H2O = aero_rep_ptr%spec_state_id(key);

      ! Make sure the expected species are in the model
      call assert(643455452, idx_A.gt.0)
      call assert(917307621, idx_B.gt.0)
      call assert(747150717, idx_C.gt.0)
      call assert(971787407, idx_H2O.gt.0)

#ifdef CAMP_USE_MPI
      ! pack the camp core
      pack_size = camp_core%pack_size() &
                  + update_data_GMD%pack_size() &
                  + update_data_GSD%pack_size() &
                  + rate_update_A%pack_size() &
                  + rate_update_B%pack_size()
      allocate(buffer(pack_size))
      pos = 0
      call camp_core%bin_pack(buffer, pos)
      call update_data_GMD%bin_pack(buffer, pos)
      call update_data_GSD%bin_pack(buffer, pos)
      call rate_update_A%bin_pack(buffer, pos)
      call rate_update_B%bin_pack(buffer, pos)
      call assert(636035849, pos.eq.pack_size)
    end if

    ! broadcast the species ids
    call camp_mpi_bcast_integer(idx_A)
    call camp_mpi_bcast_integer(idx_B)
    call camp_mpi_bcast_integer(idx_C)
    call camp_mpi_bcast_integer(idx_H2O)
    call camp_mpi_bcast_integer(i_sect_unused)
    call camp_mpi_bcast_integer(i_sect_the_mode)

    ! broadcast the buffer size
    call camp_mpi_bcast_integer(pack_size)

    if (camp_mpi_rank().eq.1) then
      ! allocate the buffer to receive data
      allocate(buffer(pack_size))
    end if

    ! broadcast the data
    call camp_mpi_bcast_packed(buffer)

    if (camp_mpi_rank().eq.1) then
      ! unpack the data
      camp_core => camp_core_t()
      pos = 0
      call camp_core%bin_unpack(buffer, pos)
      call update_data_GMD%bin_unpack(buffer, pos)
      call update_data_GSD%bin_unpack(buffer, pos)
      call rate_update_A%bin_unpack(buffer, pos)
      call rate_update_B%bin_unpack(buffer, pos)
      call assert(913246791, pos.eq.pack_size)
      allocate(buffer_copy(pack_size))
      pos = 0
      call camp_core%bin_pack(buffer_copy, pos)
      call update_data_GMD%bin_pack(buffer_copy, pos)
      call update_data_GSD%bin_pack(buffer_copy, pos)
      call rate_update_A%bin_pack(buffer_copy, pos)
      call rate_update_B%bin_pack(buffer_copy, pos)
      call assert(408040386, pos.eq.pack_size)
      do i_elem = 1, pack_size
        call assert_msg(185309230, buffer(i_elem).eq.buffer_copy(i_elem), &
                "Mismatch in element :"//trim(to_string(i_elem)))
      end do

      ! solve and evaluate results on process 1
#endif

      ! Initialize the solver
      call camp_core%solver_initialize()

      ! Get a model state variable
      camp_state => camp_core%new_state()

      ! Check the size of the state array
      state_size = size(camp_state%state_var)
      call assert_msg(235226766, state_size.eq.NUM_STATE_VAR, &
                      "Wrong state size: "//to_string( state_size ))

      ! Set the environmental conditions
      call camp_state%env_states(1)%set_temperature_K(   temp )
      call camp_state%env_states(1)%set_pressure_Pa( pressure )

      ! Save the initial concentrations
      true_conc(:,:) = 0.0
      true_conc(0,idx_A) = 1.0
      true_conc(0,idx_B) = 0.0
      true_conc(0,idx_C) = 0.0
      true_conc(:,idx_H2O) = conc_water
      model_conc(0,:) = true_conc(0,:)

      ! Set the initial state in the model
      camp_state%state_var(:) = model_conc(0,:)

      ! Set the photo B rate
      call rate_update_A%set_rate(j1)
      call rate_update_B%set_rate(j2)
      call camp_core%update_data(rate_update_A)
      call camp_core%update_data(rate_update_B)

#ifdef CAMP_DEBUG
      ! Evaluate the Jacobian during solving
      solver_stats%eval_Jac = .true.
#endif

      ! scale j2 appropriately here
      j2 = j2 * j2_scaling
      ! Integrate the mechanism
      do i_time = 1, NUM_TIME_STEP

        ! Get the modeled conc
        call camp_core%solve(camp_state, time_step, &
                              solver_stats = solver_stats)
        model_conc(i_time,:) = camp_state%state_var(:)

#ifdef CAMP_DEBUG
        ! Check the Jacobian evaluations
        call assert_msg(772386254, solver_stats%Jac_eval_fails.eq.0, &
                        trim( to_string( solver_stats%Jac_eval_fails ) )// &
                        " Jacobian evaluation failures at time step "// &
                        trim( to_string( i_time ) ) )
#endif

        ! Get the analytic concentrations
        time = i_time * time_step
        true_conc(i_time,idx_A) = true_conc(0,idx_A) * exp(-j1*time)
        true_conc(i_time,idx_B) = true_conc(0,idx_A) * &
                (j1/(j2-j1)) * &
                (exp(-j1*time) - exp(-j2*time)) * MW_B / MW_A
        true_conc(i_time,idx_C) = true_conc(0,idx_A) * MW_C / MW_A * &
                (1.0d0 + (j1*exp(-j2*time) - &
                j2*exp(-j1*time))/(j2-j1))
      end do

      ! Save the results
      open(unit=7, file="out/condensed_phase_photolysis_results.txt", &
            status="replace", action="write")
      do i_time = 0, NUM_TIME_STEP
        write(7,*) i_time*time_step, &
              ' ', true_conc(i_time, idx_A), &
              ' ', model_conc(i_time, idx_A), &
              ' ', true_conc(i_time, idx_B), &
              ' ', model_conc(i_time, idx_B), &
              ' ', true_conc(i_time, idx_C), &
              ' ', model_conc(i_time, idx_C), &
              ' ', true_conc(i_time, idx_H2O), &
              ' ', model_conc(i_time, idx_H2O)
      end do
      close(7)

      ! Analyze the results (single-particle only)
      do i_time = 1, NUM_TIME_STEP
        do i_spec = 1, size(model_conc, 2)
          call assert_msg(548109047, &
            almost_equal(model_conc(i_time, i_spec), &
            true_conc(i_time, i_spec), real(1.0e-2, kind=dp)).or. &
            (model_conc(i_time, i_spec).lt.1e-5*model_conc(1, i_spec).and. &
            true_conc(i_time, i_spec).lt.1e-5*true_conc(1, i_spec)).or. &
            (model_conc(i_time, i_spec).lt.1.0e-30.and. &
            true_conc(i_time, i_spec).lt.1.0e-30), &
            "time: "//trim(to_string(i_time))//"; species: "// &
            trim(to_string(i_spec))//"; mod: "// &
            trim(to_string(model_conc(i_time, i_spec)))//"; true: "// &
            trim(to_string(true_conc(i_time, i_spec))))
        end do
      end do

      deallocate(camp_state)

#ifdef CAMP_USE_MPI
      ! convert the results to an integer
      if (run_condensed_phase_photolysis_test) then
        results = 0
      else
        results = 1
      end if
    end if

    ! Send the results back to the primary process
    call camp_mpi_transfer_integer(results, results, 1, 0)

    ! convert the results back to a logical value
    if (camp_mpi_rank().eq.0) then
      if (results.eq.0) then
        run_condensed_phase_photolysis_test = .true.
      else
        run_condensed_phase_photolysis_test = .false.
      end if
    end if

    deallocate(buffer)
#endif

    deallocate(camp_core)

  end function run_condensed_phase_photolysis_test

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end program camp_test_condensed_phase_photolysis
