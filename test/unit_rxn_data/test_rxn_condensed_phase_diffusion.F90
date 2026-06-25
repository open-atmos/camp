! Copyright (C) 2021 Barcelona Supercomputing Center and University of
! Illinois at Urbana-Champaign
! SPDX-License-Identifier: MIT

!> \file
!> The camp_test_condensed_phase_diffusion program

!> Test of condensed_phase_diffusion reaction module
program camp_test_condensed_phase_diffusion

  use iso_c_binding

  use camp_util,                         only: i_kind, dp, assert, &
                                              almost_equal, string_t, &
                                              warn_msg, to_string
  use camp_camp_core
  use camp_camp_state
  use camp_aero_rep_data
  use camp_aero_rep_factory
  use camp_aero_rep_single_particle
  use camp_aero_rep_modal_binned_mass
  use camp_solver_stats
  use camp_rxn_data
  use camp_rxn_condensed_phase_diffusion
  use camp_mechanism_data
#ifdef CAMP_USE_JSON
  use json_module
#endif
  use camp_mpi

  implicit none

  ! Number of timesteps to output in mechanisms
  integer(kind=i_kind) :: NUM_TIME_STEP = 100

  ! initialize mpi
  call camp_mpi_init()

  if (run_condensed_phase_diffusion_tests()) then
    if (camp_mpi_rank().eq.0) write(*,*) "Condensed-phase diffusion reaction tests - PASS"
  else
    if (camp_mpi_rank().eq.0) write(*,*) "Condensed-phase diffusion reaction tests - FAIL"
    stop 3
  end if

  ! finalize mpi
  call camp_mpi_finalize()

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Run all camp_chem_mech_solver tests
  logical function run_condensed_phase_diffusion_tests() result(passed)

    use camp_camp_solver_data

    type(camp_solver_data_t), pointer :: camp_solver_data

    camp_solver_data => camp_solver_data_t()

    if (camp_solver_data%is_solver_available()) then
      passed = run_condensed_phase_diffusion_test(1)
    else
      call warn_msg(723042853, "No solver available")
      passed = .true.
    end if

    deallocate(camp_solver_data)

  end function run_condensed_phase_diffusion_tests

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Solve a mechanism consisting of a solute diffusing through the condensed phase
  !!
  !! One scenario is tested:
  !! (1) single-particle aerosol representation 
  logical function run_condensed_phase_diffusion_test(scenario)

    use camp_constants

    !> Scenario flag
    integer, intent(in) :: scenario

    type(camp_core_t), pointer :: camp_core
    type(camp_state_t), pointer :: camp_state
    character(len=:), allocatable :: input_file_path, key, idx_prefix
    type(string_t), allocatable, dimension(:) :: output_file_path

    class(aero_rep_data_t), pointer :: aero_rep_ptr
    integer(kind=i_kind) :: num_state_var, state_size
    real(kind=dp), allocatable, dimension(:,:) :: model_conc, true_conc
    integer(kind=i_kind) :: idx_solute_l0, idx_solute_l1, idx_solute_l2, &
            idx_solute_l3, idx_H2O_l0, idx_H2O_l1, idx_H2O_l2, idx_H2O_l3, &
            i_time, i_spec, i
    real(kind=dp) :: time_step, time, conc_water, MW_solute, D_solute
#ifdef CAMP_USE_MPI
    character, allocatable :: buffer(:), buffer_copy(:)
    integer(kind=i_kind) :: pack_size, pos, i_elem, results, rank_1_results
#endif

    type(solver_stats_t), target :: solver_stats
    real(kind=dp), target :: radius, number_conc
    real(kind=dp) :: layer_thickness_l0, layer_thickness_l1, layer_thickness_l2, layer_thickness_l3
    real(kind=dp) :: surface_area_l0, surface_area_l1, surface_area_l2
    real(kind=dp) :: volume_phase_l1_p1, volume_phase_l2_p1
    real(kind=dp) :: rate_second, expected_rate_first, expected_rate_second
    real(kind=dp) :: test_tolerance
    real(kind=dp) :: volume_phase_l3, volume_phase_l2

    integer(kind=i_kind) :: i_sect_unused, i_sect_the_mode

    ! For setting particle radius and number concentration
    type(aero_rep_factory_t) :: aero_rep_factory
    type(aero_rep_update_data_single_particle_number_t) :: number_update

    ! Variables for diffusion coefficient testing
    type(mechanism_data_t), pointer :: mechanism
    class(rxn_data_t), pointer :: rxn
    integer(kind=i_kind) :: i_rxn, num_adjacent_pairs
    real(kind=dp) :: expected_diff_coeff, num_int_prop
    real(kind=dp), allocatable :: diff_coeff_first(:), diff_coeff_second(:)
    real(kind=dp), allocatable :: phase_id_first(:), phase_id_second(:)
    real(kind=dp), allocatable :: aero_spec_first(:), aero_spec_second(:)
    real(kind=dp), allocatable :: phase_id_first_expected(:), phase_id_second_expected(:)
    real(kind=dp), allocatable :: aero_spec_first_expected(:), aero_spec_second_expected(:)
    real(kind=dp) :: rate_first

    call assert_msg(227053212, scenario.eq.1, &
              "Invalid scenario specified: "//to_string( scenario ))

    run_condensed_phase_diffusion_test = .true.

    ! Allocate space for the results
    num_state_var = 52 ! particles * layers * species ** this needs to be looked at**
    allocate(model_conc(0:NUM_TIME_STEP, num_state_var))
    allocate(true_conc(0:NUM_TIME_STEP, num_state_var))

    ! Set the rate constants (for calculating the true values)
    MW_solute = 0.058 ! molecular weight of solute (kg/mol)
    D_solute = 1.5e-5 ! diffusion coeff of solute in condensed phase (m2/s)
    conc_water = 2.3d-2 ! molar concentration of water in the condensed phase (mol/m3)
    !!! JJJ: Where is the size of the particle specified???

    ! Set output time step (s)
    time_step = 1.0d0

#ifdef CAMP_USE_MPI
    ! Load the model data on root process and pass it to process 1 for solving
    if (camp_mpi_rank().eq.0) then
#endif

      ! Get the condensed_phase_diffusion reaction mechanism json file
      input_file_path = 'test_condensed_phase_diffusion_config.json'

      ! Construct a camp_core variable
      camp_core => camp_core_t(input_file_path)

      deallocate(input_file_path)

      ! Initialize the model
      call camp_core%initialize()

      ! Find the aerosol representation
      key = "my aero rep 2"
      call assert(337943171, camp_core%get_aero_rep(key, aero_rep_ptr))

      select type (aero_rep_ptr)
        type is (aero_rep_single_particle_t)
          call camp_core%initialize_update_object( aero_rep_ptr, &
                                                     number_update )
        class default
          call die_msg(594380626, "Incorrect aerosol representation type")
      end select

      ! Get species indices
      idx_prefix = "P1.zero layer."
      key = idx_prefix//"aqueous aerosol.solute_aq"
      idx_solute_l0 = aero_rep_ptr%spec_state_id(key);

      key = idx_prefix//"aqueous aerosol.H2O_aq"
      idx_H2O_l0 = aero_rep_ptr%spec_state_id(key);

      idx_prefix = "P1.one layer."
      key = idx_prefix//"aqueous aerosol.solute_aq"
      idx_solute_l1 = aero_rep_ptr%spec_state_id(key);

      key = idx_prefix//"aqueous aerosol.H2O_aq"
      idx_H2O_l1 = aero_rep_ptr%spec_state_id(key);

      idx_prefix = "P1.two layer."
      key = idx_prefix//"aqueous aerosol.solute_aq"
      idx_solute_l2 = aero_rep_ptr%spec_state_id(key);

      key = idx_prefix//"aqueous aerosol.H2O_aq"
      idx_H2O_l2 = aero_rep_ptr%spec_state_id(key);

      idx_prefix = "P1.three layer."
      key = idx_prefix//"aqueous aerosol.solute_aq"
      idx_solute_l3 = aero_rep_ptr%spec_state_id(key);

      key = idx_prefix//"aqueous aerosol.H2O_aq"
      idx_H2O_l3 = aero_rep_ptr%spec_state_id(key);

      ! Make sure the expected species are in the model
      call assert(050889938, idx_solute_l0.gt.0)
      call assert(599205790, idx_solute_l1.gt.0)
      call assert(434244152, idx_solute_l2.gt.0)
      call assert(260481458, idx_solute_l3.gt.0)
      call assert(149096792, idx_H2O_l0.gt.0)
      call assert(011666208, idx_H2O_l1.gt.0)
      call assert(465533442, idx_H2O_l2.gt.0)
      call assert(250659956, idx_H2O_l3.gt.0)

#ifdef CAMP_USE_MPI
      ! pack the camp core
      pack_size = camp_core%pack_size()
      pack_size = pack_size &
                + number_update%pack_size()
      allocate(buffer(pack_size))
      pos = 0
      call camp_core%bin_pack(buffer, pos)
      call number_update%bin_pack(buffer, pos)
      call assert(761722462, pos.eq.pack_size)
    end if

    ! broadcast the species ids
    call camp_mpi_bcast_integer(idx_solute_l0)
    call camp_mpi_bcast_integer(idx_solute_l1)
    call camp_mpi_bcast_integer(idx_solute_l2)
    call camp_mpi_bcast_integer(idx_solute_l3)
    call camp_mpi_bcast_integer(idx_H2O_l0)
    call camp_mpi_bcast_integer(idx_H2O_l1)
    call camp_mpi_bcast_integer(idx_H2O_l2)
    call camp_mpi_bcast_integer(idx_H2O_l3)

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
      call number_update%bin_unpack(buffer, pos)
      call assert(967359696, pos.eq.pack_size)
      allocate(buffer_copy(pack_size))
      pos = 0
      call camp_core%bin_pack(buffer_copy, pos)
      call number_update%bin_pack(buffer_copy, pos)
      call assert(524365939, pos.eq.pack_size)
      do i_elem = 1, pack_size
        call assert_msg(811596732, buffer(i_elem).eq.buffer_copy(i_elem), &
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
      call assert_msg(093459490, state_size.eq.NUM_STATE_VAR, &
                      "Wrong state size: "//to_string( state_size ))

      ! Save the initial concentrations
      true_conc(:,:) = 0.0
      true_conc(0,idx_solute_l0) = 0.0
      true_conc(0,idx_solute_l1) = 0.0
      true_conc(0,idx_solute_l2) = 0.0
      true_conc(0,idx_solute_l3) = 1.0d-2
      true_conc(:,idx_H2O_l0) = conc_water
      true_conc(:,idx_H2O_l1) = conc_water
      true_conc(:,idx_H2O_l2) = conc_water
      true_conc(:,idx_H2O_l3) = conc_water
      number_conc = 1.3e6         ! particle number concentration (#/cc)
      true_conc(0,:) = true_conc(0,:) / (number_conc * 1000.0) ! convert to kg/m3 per particle
      model_conc(0,:) = true_conc(0,:)

      ! single particle aerosol mass concentrations are per particle
      ! radius (m) calculated based on particle mass
      radius = ( ( true_conc(0,idx_solute_l0) +  &
                   true_conc(0,idx_solute_l1) +  &
                    true_conc(0,idx_solute_l2) +  &
                   true_conc(0,idx_solute_l3) + &
                   true_conc(0,idx_H2O_l0) + & 
                   true_conc(0,idx_H2O_l1) + & 
                   true_conc(0,idx_H2O_l2) + & 
                   true_conc(0,idx_H2O_l3) ) &
                   * 3.0 / 4.0 / 3.14159265359 )**(1.0/3.0)
      layer_thickness_l3 = ( ( true_conc(0,idx_solute_l0) +  &
                   true_conc(0,idx_solute_l1) +  &
                   true_conc(0,idx_solute_l2) +  &
                   true_conc(0,idx_solute_l3) + &
                   true_conc(0,idx_H2O_l0) + & 
                   true_conc(0,idx_H2O_l1) + & 
                   true_conc(0,idx_H2O_l2) + & 
                   true_conc(0,idx_H2O_l3) ) &
                   * 3.0 / 4.0 / 3.14159265359 )**(1.0/3.0) - &
                   ( ( true_conc(0,idx_solute_l0) +  &
                   true_conc(0,idx_solute_l1) +  &
                   true_conc(0,idx_solute_l2) + &
                   true_conc(0,idx_H2O_l0) + & 
                   true_conc(0,idx_H2O_l1) + & 
                   true_conc(0,idx_H2O_l2) ) &
                   * 3.0 / 4.0 / 3.14159265359 )**(1.0/3.0)
        layer_thickness_l2 = ( ( true_conc(0,idx_solute_l0) +  &
                   true_conc(0,idx_solute_l1) +  &
                   true_conc(0,idx_solute_l2) + &
                   true_conc(0,idx_H2O_l0) + & 
                   true_conc(0,idx_H2O_l1) + & 
                   true_conc(0,idx_H2O_l2) ) &
                   * 3.0 / 4.0 / 3.14159265359 )**(1.0/3.0) - &
                   ( ( true_conc(0,idx_solute_l0) +  &
                   true_conc(0,idx_solute_l1) + &
                   true_conc(0,idx_H2O_l0) + & 
                   true_conc(0,idx_H2O_l1) ) &
                   * 3.0 / 4.0 / 3.14159265359 )**(1.0/3.0)
        layer_thickness_l1 = ( ( true_conc(0,idx_solute_l0) +  &
                   true_conc(0,idx_solute_l1) + &
                   true_conc(0,idx_H2O_l0) + & 
                   true_conc(0,idx_H2O_l1) ) &
                   * 3.0 / 4.0 / 3.14159265359 )**(1.0/3.0) - &
                   ( (true_conc(0,idx_solute_l0) + &
                   true_conc(0,idx_H2O_l0) ) &
                   * 3.0 / 4.0 / 3.14159265359 )**(1.0/3.0)
        layer_thickness_l0 = ( ( true_conc(0,idx_solute_l0) +  &
                   true_conc(0,idx_H2O_l0) ) &
                   * 3.0 / 4.0 / 3.14159265359 )**(1.0/3.0)
        surface_area_l2 = 4.0 * 3.14159265359 * ( ( true_conc(0,idx_solute_l0) +  &
                   true_conc(0,idx_solute_l1) +  &
                   true_conc(0,idx_solute_l2) + &
                   true_conc(0,idx_H2O_l0) + & 
                   true_conc(0,idx_H2O_l1) + & 
                   true_conc(0,idx_H2O_l2) ) &
                   * 3.0 / 4.0 / 3.14159265359 )**(2.0/3.0)
        surface_area_l1 = 4.0 * 3.14159265359 * ( ( true_conc(0,idx_solute_l0) +  &
                   true_conc(0,idx_solute_l1) + &
                   true_conc(0,idx_H2O_l0) + & 
                   true_conc(0,idx_H2O_l1) ) &
                   * 3.0 / 4.0 / 3.14159265359 )**(2.0/3.0)
        surface_area_l0 = 4.0 * 3.14159265359 * ( ( true_conc(0,idx_solute_l0) +  &
                   true_conc(0,idx_H2O_l0) ) &
                   * 3.0 / 4.0 / 3.14159265359 )**(2.0/3.0)

      ! Update the aerosol representation (single particle only)
      call number_update%set_number__n_m3(1, number_conc)
      call camp_core%update_data(number_update)

      ! Set the initial state in the model
      camp_state%state_var(:) = model_conc(0,:)

#ifdef CAMP_DEBUG
      ! Evaluate the Jacobian during solving
      !solver_stats%eval_Jac = .true.
#endif

      ! Integrate the mechanism
      do i_time = 1, NUM_TIME_STEP

        ! Get the modeled conc
        call camp_core%solve(camp_state, time_step, &
                              solver_stats = solver_stats)
        model_conc(i_time,:) = camp_state%state_var(:)
      end do

#ifdef CAMP_DEBUG
        ! Check the Jacobian evaluations
        !call assert_msg(567403530, solver_stats%Jac_eval_fails.eq.0, &
        !                trim( to_string( solver_stats%Jac_eval_fails ) )// &
        !                " Jacobian evaluation failures at time step "// &
        !                trim( to_string( i_time ) ) )
#endif

      ! Test diffusion coefficients
      key = "condensed phase diffusion"
      if (camp_core%get_mechanism(key, mechanism)) then
        ! Find the condensed phase diffusion reaction
        do i_rxn = 1, mechanism%size()
          rxn => mechanism%get_rxn(i_rxn)
          select type (rxn_diffusion => rxn)
            class is (rxn_condensed_phase_diffusion_t)
              ! Get the number of adjacent phase pairs
              num_adjacent_pairs = rxn_diffusion%condensed_data_int(1)
              
              ! Allocate arrays for diffusion coefficients
              if (.not. allocated(diff_coeff_first)) then
                allocate(diff_coeff_first(num_adjacent_pairs))
                allocate(diff_coeff_second(num_adjacent_pairs))
              end if
              if (.not. allocated(phase_id_first)) then
                allocate(phase_id_first(num_adjacent_pairs))
                allocate(phase_id_first_expected(num_adjacent_pairs))
                allocate(phase_id_second(num_adjacent_pairs))
                allocate(phase_id_second_expected(num_adjacent_pairs))
              end if
              if (.not. allocated(aero_spec_first)) then
                allocate(aero_spec_first(num_adjacent_pairs))
                allocate(aero_spec_first_expected(num_adjacent_pairs))
                allocate(aero_spec_second(num_adjacent_pairs))
                allocate(aero_spec_second_expected(num_adjacent_pairs))
              end if
              
              ! Extract diffusion coefficients from condensed data
              num_int_prop = 1
              do i = 1, num_adjacent_pairs
                diff_coeff_first(i) = rxn_diffusion%condensed_data_real(i)
                diff_coeff_second(i) = rxn_diffusion%condensed_data_real(num_adjacent_pairs + i)
                phase_id_first(i) = rxn_diffusion%condensed_data_int(i + num_int_prop)
                phase_id_second(i) = rxn_diffusion%condensed_data_int(num_adjacent_pairs + i + num_int_prop)
                aero_spec_first(i) = rxn_diffusion%condensed_data_int((2 * num_adjacent_pairs) + i + num_int_prop)
                aero_spec_second(i) = rxn_diffusion%condensed_data_int((3 * num_adjacent_pairs) + i + num_int_prop)
              end do
              
              ! Test that all diffusion coefficients match the expected value
              expected_diff_coeff = 1.5d-5
              phase_id_first_expected = (/1,3,5,1,2,3,5,6,7,9,10,11,13,14,15,1,3/)
              phase_id_second_expected = (/2,4,6,2,3,4,6,7,8,10,11,12,14,15,16,2,4/)
              aero_spec_first_expected = (/1,5,9,13,15,17,21,23,25,29,31,33,37,39,41,45,49/)
              aero_spec_second_expected = (/3,7,11,15,17,19,23,25,27,31,33,35,39,41,43,47,51/)
              do i = 1, num_adjacent_pairs - 1
                call assert_msg(449021345, almost_equal(diff_coeff_first(i), expected_diff_coeff, 1.0d-15), &
                                "DIFF_COEFF_FIRST_ for pair "//trim(to_string(i))//" is "// &
                                trim(to_string(diff_coeff_first(i)))//" expected "// &
                                trim(to_string(expected_diff_coeff)))
                call assert_msg(593847156, almost_equal(diff_coeff_second(i), expected_diff_coeff, 1.0d-15), &
                                "DIFF_COEFF_SECOND_ for pair "//trim(to_string(i))//" is "// &
                                trim(to_string(diff_coeff_second(i)))//" expected "// &
                                trim(to_string(expected_diff_coeff)))
                call assert_msg(678901234, almost_equal(phase_id_first(i), phase_id_first_expected(i), 1.0d-15), &
                                "PHASE_ID_FIRST_ for pair "//trim(to_string(i))//" is "// &
                                trim(to_string(phase_id_first(i)))//" expected "// &
                                trim(to_string(phase_id_first_expected(i))))
                call assert_msg(789012345, almost_equal(phase_id_second(i), phase_id_second_expected(i), 1.0d-15), &
                                "PHASE_ID_SECOND_ for pair "//trim(to_string(i))//" is "// &
                                trim(to_string(phase_id_second(i)))//" expected "// &
                                trim(to_string(phase_id_second_expected(i))))
                call assert_msg(890123456, almost_equal(aero_spec_first(i), aero_spec_first_expected(i), 1.0d-15), &
                                "AERO_SPEC_FIRST_ for pair "//trim(to_string(i))//" is "// &
                                trim(to_string(aero_spec_first(i)))//" expected "// &
                                trim(to_string(aero_spec_first_expected(i))))
                call assert_msg(901234567, almost_equal(aero_spec_second(i), aero_spec_second_expected(i), 1.0d-15), &
                                "AERO_SPEC_SECOND_ for pair "//trim(to_string(i))//" is "// &
                                trim(to_string(aero_spec_second(i)))//" expected "// &
                                trim(to_string(aero_spec_second_expected(i))))
              end do
              ! Test rate_first and rate_second calculations based on surface_area and layer_thickness
              ! 
              ! Formula from condensed phase diffusion solver:
              ! rate_first = (eff_sa / volume_phase_first) * (
              !     (-DIFF_COEFF_FIRST_ / layer_thickness_first) * state[PHASE_ID_FIRST_] +
              !     (DIFF_COEFF_SECOND_ / layer_thickness_second) * state[PHASE_ID_SECOND_]
              ! )
              !
              ! rate_second = (eff_sa / volume_phase_second) * (
              !     (DIFF_COEFF_FIRST_ / layer_thickness_first) * state[PHASE_ID_FIRST_] -
              !     (DIFF_COEFF_SECOND_ / layer_thickness_second) * state[PHASE_ID_SECOND_]
              ! )
              !
              ! Where:
              ! - eff_sa = interface surface area between layers (m2)
              ! - volume_phase_first = total mass of first phase (kg per particle)
              ! - volume_phase_second = total mass of second phase (kg per particle)
              ! - state values = concentrations of species in each layer (kg/m3 per particle)
              
              test_tolerance = 1.0d-12
              
              ! Calculate volume (mass) of each phase
              volume_phase_l3 = true_conc(0,idx_solute_l3) + true_conc(0,idx_H2O_l3)
              volume_phase_l2 = true_conc(0,idx_solute_l2) + true_conc(0,idx_H2O_l2)

              ! Calculate expected rate_first
              ! rate_first = (eff_sa / volume_phase_first) * (
              !     (-DIFF_COEFF_FIRST_ / layer_thickness_first) * state_l1 +
              !     (DIFF_COEFF_SECOND_ / layer_thickness_second) * state_l2
              ! )
              expected_rate_first = (surface_area_l2 / volume_phase_l2) * ( &
                  (-diff_coeff_first(6) / layer_thickness_l2) * true_conc(0,idx_solute_l2) + &
                  (diff_coeff_second(6) / layer_thickness_l3) * true_conc(0,idx_solute_l3) )
              print *, "Expected rate_first: ", expected_rate_first
              print *, "surface_area_l2: ", surface_area_l2
              print *, "volume_phase_l2: ", volume_phase_l2
              print *, "diff_coeff_first(6): ", diff_coeff_first(6)
              print *, "layer_thickness_l2: ", layer_thickness_l2
              print *, "diff_coeff_second(6): ", diff_coeff_second(6)
              print *, "layer_thickness_l3: ", layer_thickness_l3
              print *, "true_conc(0,idx_solute_l2): ", true_conc(0,idx_solute_l2)
              print *, "true_conc(0,idx_solute_l3): ", true_conc(0,idx_solute_l3)

              ! Calculate expected rate_second
              ! rate_second = (eff_sa / volume_phase_second) * (
              !     (DIFF_COEFF_FIRST_ / layer_thickness_first) * state_l1 -
              !     (DIFF_COEFF_SECOND_ / layer_thickness_second) * state_l2
              ! )
              expected_rate_second = (surface_area_l2 / volume_phase_l3) * ( &
                  (diff_coeff_first(6) / layer_thickness_l2) * true_conc(0,idx_solute_l2) - &
                  (diff_coeff_second(6) / layer_thickness_l3) * true_conc(0,idx_solute_l3) )
              print *, "Expected rate_second: ", expected_rate_second
              print *, "surface_area_l2: ", surface_area_l2
              print *, "volume_phase_l3: ", volume_phase_l3
              print *, "diff_coeff_first(6): ", diff_coeff_first(6)
              print *, "layer_thickness_l2: ", layer_thickness_l2
              print *, "diff_coeff_second(6): ", diff_coeff_second(6)
              print *, "layer_thickness_l3: ", layer_thickness_l3
              
          end select
        end do
      end if

      deallocate(camp_state)

#ifdef CAMP_USE_MPI
      ! convert the results to an integer
      if (run_condensed_phase_diffusion_test) then
        results = 0
      else
        results = 1
      end if
      rank_1_results = results
    end if

    ! Send the results back to the primary process
    call camp_mpi_transfer_integer(rank_1_results, results, 1, 0)

    ! convert the results back to a logical value
    if (camp_mpi_rank().eq.0) then
      if (results.eq.0) then
        run_condensed_phase_diffusion_test = .true.
      else
        run_condensed_phase_diffusion_test = .false.
      end if
    end if

    deallocate(buffer)
#endif

    ! Deallocate diffusion coefficient arrays if they were allocated
    if (allocated(diff_coeff_first)) deallocate(diff_coeff_first)
    if (allocated(diff_coeff_second)) deallocate(diff_coeff_second)
    if (allocated(phase_id_first)) deallocate(phase_id_first)
    if (allocated(phase_id_second)) deallocate(phase_id_second)
    if (allocated(aero_spec_first)) deallocate(aero_spec_first)
    if (allocated(aero_spec_second)) deallocate(aero_spec_second)

    deallocate(camp_core)

  end function run_condensed_phase_diffusion_test

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end program camp_test_condensed_phase_diffusion
