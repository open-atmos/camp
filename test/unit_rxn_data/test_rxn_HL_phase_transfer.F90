! Copyright (C) 2021 Barcelona Supercomputing Center and University of
! Illinois at Urbana-Champaign
! SPDX-License-Identifier: MIT

!> \file
!> The camp_test_HL_phase_transfer program

!> Test of HL_phase_transfer reaction module
program camp_test_HL_phase_transfer

  use iso_c_binding

  use camp_util,                         only: i_kind, dp, assert, &
                                              almost_equal, string_t, &
                                              warn_msg
  use camp_camp_core
  use camp_camp_state
  use camp_chem_spec_data
  use camp_aero_rep_data
  use camp_aero_rep_factory
  use camp_aero_rep_modal_binned_mass
  use camp_aero_rep_single_particle
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

  if (run_HL_phase_transfer_tests()) then
    if (camp_mpi_rank().eq.0) write(*,*) "Henry's Law phase transfer reaction tests - PASS"
  else
    if (camp_mpi_rank().eq.0) write(*,*) "Henry's Law phase transfer reaction tests - FAIL"
    stop 3
  end if

  ! finalize mpi
  call camp_mpi_finalize()

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Run all camp_chem_mech_solver tests
  logical function run_HL_phase_transfer_tests() result(passed)

    use camp_camp_solver_data

    type(camp_solver_data_t), pointer :: camp_solver_data

    camp_solver_data => camp_solver_data_t()

    if (camp_solver_data%is_solver_available()) then
      passed = run_HL_phase_transfer_test(1)
      passed = passed .and. run_HL_phase_transfer_test(2)
    else
      call warn_msg(713064651, "No solver available")
      passed = .true.
    end if

    deallocate(camp_solver_data)

  end function run_HL_phase_transfer_tests

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Solve a mechanism consisting of two phase transfer reactions
  !!
  !! Ozone (O3) and hydrogen peroxide (H2O2) partitioning based on parameters
  !! and equations from CAPRAM 2.4 reduced mechanism.
  !! (Ervens, B., et al., 2003. "CAPRAM 2.4 (MODAC mechanism) : An extended
  !! and condensed tropospheric aqueous phase mechanism and its
  !! application." J. Geophys. Res. 108, 4426. doi:10.1029/2002JD002202
  !!
  !! One of two scenarios is tested, depending on the passed integer:
  !! (1) single-particle aerosol representation and fixed water concentration
  !! (2) modal aerosol representation and ZSR-calculated water concentration
  logical function run_HL_phase_transfer_test(scenario)

    use camp_constants

    !> Scenario flag
    integer, intent(in) :: scenario

    type(camp_core_t), pointer :: camp_core
    type(camp_state_t), pointer :: camp_state
    character(len=:), allocatable :: input_file_path
    type(string_t), allocatable, dimension(:) :: output_file_path

    type(chem_spec_data_t), pointer :: chem_spec_data
    class(aero_rep_data_t), pointer :: aero_rep_ptr
    real(kind=dp), allocatable, dimension(:,:) :: model_conc, true_conc
    integer(kind=i_kind) :: idx_phase, idx_aero_rep, idx_HNO3, idx_HNO3_aq, &
            idx_NH3, idx_NH3_aq, idx_H2O, idx_Na_p, idx_Cl_m, idx_Ca_pp, &
            i_time, i_spec
    integer(kind=i_kind) :: idx_O3, idx_O3_aq, &
            idx_O3_aq_layer1, idx_O3_aq_layer2, idx_H2O2, idx_H2O2_aq, &
            idx_H2O2_aq_layer1, idx_H2O2_aq_layer2, idx_H2O_aq, &
            idx_H2O_aq_layer1, idx_H2O_aq_layer2
    character(len=:), allocatable :: key, idx_prefix
    real(kind=dp) :: time_step, time, n_star, del_H, del_S, del_G, alpha, &
            mfp, Kn, corr, M_to_ppm, kgm3_to_ppm, K_eq_O3, K_eq_H2O2, &
            k_O3_forward, k_O3_backward, k_H2O2_forward, k_H2O2_backward, &
            equil_O3, equil_O3_aq, equil_H2O2, equil_H2O2_aq, temp, pressure
    real(kind=dp), target :: radius, number_conc
#ifdef CAMP_USE_MPI
    character, allocatable :: buffer(:), buffer_copy(:)
    integer(kind=i_kind) :: pack_size, pos, i_elem, results, rank_1_results
#endif

    real(kind=dp), parameter :: MW_O3 =   0.048     ! [kg mol-1]
    real(kind=dp), parameter :: MW_H2O2 = 0.0340147 ! [kg mol-1]
    real(kind=dp), parameter :: Dg_O3   = 1.48e-5   ! diffusion coeff [m2 s-1]
    real(kind=dp), parameter :: Dg_H2O2 = 1.46e-5   ! diffusion coeff [m2 s-1]

    type(solver_stats_t), target :: solver_stats

    ! For setting particle radius and number concentration
    type(aero_rep_factory_t) :: aero_rep_factory
    type(aero_rep_update_data_single_particle_number_t) :: number_update

    integer(kind=i_kind) :: i_sect_unused, i_sect_the_mode
    type(aero_rep_update_data_modal_binned_mass_GMD_t) :: update_data_GMD
    type(aero_rep_update_data_modal_binned_mass_GSD_t) :: update_data_GSD

    print *, "did we make it here?"
    call assert_msg(144071521, scenario.ge.1 .and. scenario.le.2, &
                    "Invalid scenario specified: "//to_string( scenario ) )

    run_HL_phase_transfer_test = .true.

    ! Allocate space for the results
    if (scenario.eq.1) then
      allocate(model_conc(0:NUM_TIME_STEP, 14))
      allocate(true_conc(0:NUM_TIME_STEP, 14))
    else if (scenario.eq.2) then
      allocate(model_conc(0:NUM_TIME_STEP, 24))
      allocate(true_conc(0:NUM_TIME_STEP, 24))
    endif

    ! Set the environmental conditions
    temp = 272.5d0              ! temperature (K)
    pressure = 101253.3d0       ! pressure (Pa)

    ! Henry's Law equilibrium constants (M/ppm)
    ! O3 HLC Equil Const (M/ppm)
    K_eq_O3 = 1.12509e-7 * pressure / 1.0e6 &
              * exp(2300.0d0 * (1.0d0/temp - 1.0d0/298.0d0))
    ! H2O2 HLC Equil Const (M/ppm)
    K_eq_H2O2 = 1.011596348 * pressure / 1.0e6 &
                * exp(6340.0d0 * (1.0d0/temp - 1.0d0/298.0d0))

    ! Set output time step (s)
    time_step = 10.0

#ifdef CAMP_USE_MPI
    ! Load the model data on root process and pass it to process 1 for solving
    if (camp_mpi_rank().eq.0) then
#endif

      ! Get the HL_phase_transfer reaction mechanism json file
      if (scenario.eq.1) then
        input_file_path = 'test_HL_phase_transfer_config.json'
      else if (scenario.eq.2) then
        input_file_path = 'test_HL_phase_transfer_config_2.json'
      end if

      ! Construct a camp_core variable
      camp_core => camp_core_t(input_file_path)

      deallocate(input_file_path)

      ! Initialize the model
      call camp_core%initialize()

      ! Find the aerosol representation
      key = "my aero rep 2"
      call assert(116793129, camp_core%get_aero_rep(key, aero_rep_ptr))
      if (scenario.eq.1) then
        select type (aero_rep_ptr)
          type is (aero_rep_single_particle_t)
            call camp_core%initialize_update_object( aero_rep_ptr, &
                                                     number_update)
          class default
            call die_msg(866102326, "Incorrect aerosol representation type")
        end select
      else if (scenario.eq.2) then
        select type (aero_rep_ptr)
          type is (aero_rep_modal_binned_mass_t)
            call camp_core%initialize_update_object( aero_rep_ptr, &
                                                     update_data_GMD)
            call camp_core%initialize_update_object( aero_rep_ptr, &
                                                     update_data_GSD)
            call assert_msg(879021282, &
                  aero_rep_ptr%get_section_id("unused mode", i_sect_unused), &
                  "Could not get section id for the unused mode")
            call assert_msg(426389129, &
                  aero_rep_ptr%get_section_id("the mode", i_sect_the_mode), &
                  "Could not get section id for the unused mode")
          class default
            call die_msg(290304323, "Incorrect aerosol representation type")
        end select
      end if

      ! Get the chemical species data
      call assert(191714381, camp_core%get_chem_spec_data(chem_spec_data))

      ! Get species indices
      if (scenario.eq.1) then
        idx_prefix = "P1."
        key = "O3"
        idx_O3 = chem_spec_data%gas_state_id(key);
        key = idx_prefix//"one layer.aqueous aerosol.O3_aq"
        print *, "key", key
        idx_O3_aq_layer1 = aero_rep_ptr%spec_state_id(key);
        key = idx_prefix//"two layer.aqueous aerosol.O3_aq"
        idx_O3_aq_layer2 = aero_rep_ptr%spec_state_id(key);
        key = "H2O2"
        idx_H2O2 = chem_spec_data%gas_state_id(key);
        key = idx_prefix//"one layer.aqueous aerosol.H2O2_aq"
        idx_H2O2_aq_layer1 = aero_rep_ptr%spec_state_id(key);
        key = idx_prefix//"two layer.aqueous aerosol.H2O2_aq"
        idx_H2O2_aq_layer2 = aero_rep_ptr%spec_state_id(key);
        key = idx_prefix//"one layer.aqueous aerosol.H2O_aq"
        idx_H2O_aq_layer1 = aero_rep_ptr%spec_state_id(key);
        key = idx_prefix//"two layer.aqueous aerosol.H2O_aq"
        idx_H2O_aq_layer2 = aero_rep_ptr%spec_state_id(key);
        ! Make sure the expected species are in the model
        call assert(167389209, idx_O3.gt.0)
        call assert(156210838, idx_O3_aq_layer1.gt.0)
        call assert(802672616, idx_O3_aq_layer2.gt.0)
        call assert(844092221, idx_H2O2.gt.0)
        call assert(097661255, idx_H2O2_aq_layer1.gt.0)
        call assert(429463219, idx_H2O2_aq_layer2.gt.0)
        call assert(312315501, idx_H2O_aq_layer1.gt.0)
        call assert(448976280, idx_H2O_aq_layer2.gt.0)
      else if (scenario.eq.2) then
        idx_prefix = "the mode."
        key = "O3"
        idx_O3 = chem_spec_data%gas_state_id(key);
        key = idx_prefix//"aqueous aerosol.O3_aq"
        idx_O3_aq = aero_rep_ptr%spec_state_id(key);
        key = "H2O2"
        idx_H2O2 = chem_spec_data%gas_state_id(key);
        key = idx_prefix//"aqueous aerosol.H2O2_aq"
        idx_H2O2_aq = aero_rep_ptr%spec_state_id(key);

        key = "HNO3"
        idx_HNO3 = chem_spec_data%gas_state_id(key);
        key = idx_prefix//"aqueous aerosol.HNO3_aq"
        idx_HNO3_aq = aero_rep_ptr%spec_state_id(key);
        key = "NH3"
        idx_NH3 = chem_spec_data%gas_state_id(key);
        key = idx_prefix//"aqueous aerosol.NH3_aq"
        idx_NH3_aq = aero_rep_ptr%spec_state_id(key);
        key = "H2O"
        idx_H2O = chem_spec_data%gas_state_id(key);
        key = idx_prefix//"aqueous aerosol.Na_p"
        idx_Na_p = aero_rep_ptr%spec_state_id(key);
        key = idx_prefix//"aqueous aerosol.Cl_m"
        idx_Cl_m = aero_rep_ptr%spec_state_id(key);
        key = idx_prefix//"aqueous aerosol.Ca_pp"
        idx_Ca_pp = aero_rep_ptr%spec_state_id(key);
        key = idx_prefix//"aqueous aerosol.H2O_aq"
        idx_H2O_aq = aero_rep_ptr%spec_state_id(key);
        ! Make sure the expected species are in the model
        call assert(202593939, idx_O3.gt.0)
        call assert(262338032, idx_O3_aq.gt.0)
        call assert(374656377, idx_H2O2.gt.0)
        call assert(704441571, idx_H2O2_aq.gt.0)
        call assert(274044966, idx_HNO3.gt.0)
        call assert(386363311, idx_HNO3_aq.gt.0)
        call assert(833731157, idx_NH3.gt.0)
        call assert(946049502, idx_NH3_aq.gt.0)
        call assert(688163399, idx_H2O.gt.0)
        call assert(758921357, idx_H2O_aq.gt.0)
      end if

#ifdef CAMP_USE_MPI
      ! pack the camp core
      pack_size = camp_core%pack_size()
      if (scenario.eq.1) then
        pack_size = pack_size &
                  + number_update%pack_size()
      else if (scenario.eq.2) then
        pack_size = pack_size &
                  + update_data_GMD%pack_size() &
                  + update_data_GSD%pack_size()
      end if
      allocate(buffer(pack_size))
      pos = 0
      call camp_core%bin_pack(buffer, pos)
      if (scenario.eq.1) then
        call number_update%bin_pack(buffer, pos)
      else if (scenario.eq.2) then
        call update_data_GMD%bin_pack(buffer, pos)
        call update_data_GSD%bin_pack(buffer, pos)
      end if
      call assert(947937853, pos.eq.pack_size)
    end if

    ! broadcast the species ids
    call camp_mpi_bcast_integer(idx_O3)
    call camp_mpi_bcast_integer(idx_H2O2)
    if (scenario.eq.1) then
      call camp_mpi_bcast_integer(idx_O3_aq_layer1)
      call camp_mpi_bcast_integer(idx_O3_aq_layer2)
      call camp_mpi_bcast_integer(idx_H2O2_aq_layer1)
      call camp_mpi_bcast_integer(idx_H2O2_aq_layer2)
      call camp_mpi_bcast_integer(idx_H2O_aq_layer1)
      call camp_mpi_bcast_integer(idx_H2O_aq_layer2)
    else if (scenario.eq.2) then
      call camp_mpi_bcast_integer(idx_O3_aq)
      call camp_mpi_bcast_integer(idx_H2O2_aq)
      call camp_mpi_bcast_integer(idx_H2O_aq)
      call camp_mpi_bcast_integer(idx_HNO3)
      call camp_mpi_bcast_integer(idx_HNO3_aq)
      call camp_mpi_bcast_integer(idx_NH3)
      call camp_mpi_bcast_integer(idx_NH3_aq)
      call camp_mpi_bcast_integer(idx_H2O)
      call camp_mpi_bcast_integer(idx_Na_p)
      call camp_mpi_bcast_integer(idx_Cl_m)
      call camp_mpi_bcast_integer(idx_Ca_pp)
    end if
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
      if (scenario.eq.1) then
        call number_update%bin_unpack(buffer, pos)
      else if (scenario.eq.2) then
        call update_data_GMD%bin_unpack(buffer, pos)
        call update_data_GSD%bin_unpack(buffer, pos)
      end if
      call assert(711319310, pos.eq.pack_size)
      allocate(buffer_copy(pack_size))
      pos = 0
      call camp_core%bin_pack(buffer_copy, pos)
      if (scenario.eq.1) then
        call number_update%bin_pack(buffer_copy, pos)
      else if (scenario.eq.2) then
        call update_data_GMD%bin_pack(buffer_copy, pos)
        call update_data_GSD%bin_pack(buffer_copy, pos)
      end if
      call assert(206112905, pos.eq.pack_size)
      do i_elem = 1, pack_size
        call assert_msg(265856998, buffer(i_elem).eq.buffer_copy(i_elem), &
                "Mismatch in element :"//trim(to_string(i_elem)))
      end do

      ! solve and evaluate results on process 1
#endif

      ! Initialize the solver
      call camp_core%solver_initialize()

      ! Get a model state variable
      camp_state => camp_core%new_state()

      ! Set the environmental conditions
      call camp_state%env_states(1)%set_temperature_K(   temp )
      call camp_state%env_states(1)%set_pressure_Pa( pressure )

      ! Save the initial concentrations
      true_conc(:,:) = 0.0
      if (scenario.eq.1) then
        true_conc(0,idx_O3) = 0.0
        true_conc(0,idx_O3_aq_layer1) = 1.0e-3
        true_conc(0,idx_O3_aq_layer2) = 1.0e-3
        true_conc(0,idx_H2O2) = 1.0
        true_conc(0,idx_H2O2_aq_layer1) = 0.0
        true_conc(0,idx_H2O2_aq_layer2) = 0.0
        true_conc(0,idx_H2O_aq_layer1) = 1.4e-2
        true_conc(0,idx_H2O_aq_layer2) = 1.4e-2
      else if (scenario.eq.2) then
        true_conc(0,idx_O3) = 0.0
        true_conc(0,idx_O3_aq) = 1.0
        true_conc(0,idx_H2O2) = 1.0
        true_conc(0,idx_H2O2_aq) = 0.0
        true_conc(0,idx_HNO3) = 1.0
        true_conc(0,idx_HNO3_aq) = 0.0
        true_conc(0,idx_NH3) = 1.0
        true_conc(0,idx_NH3_aq) = 0.0
        true_conc(0,idx_Na_p) = 2.5
        true_conc(0,idx_Cl_m) = 5.3
        true_conc(0,idx_Ca_pp) = 1.3
        true_conc(0,idx_H2O) = 3000.0
      end if

      ! Calculate the radius and number concentration to use
      !
      ! The modal scenario is only used for Jacobian checking
      if (scenario.eq.1) then
        number_conc = 1.3e6         ! particle number concentration (#/cc)
        ! single particle aerosol mass concentrations are per particle
        true_conc(0,idx_O3_aq_layer1)  = true_conc(0,idx_O3_aq_layer1)  / number_conc
        true_conc(0,idx_O3_aq_layer2)  = true_conc(0,idx_O3_aq_layer2)  / number_conc
        true_conc(0,idx_H2O_aq_layer1) = true_conc(0,idx_H2O_aq_layer1) / number_conc
        true_conc(0,idx_H2O_aq_layer2) = true_conc(0,idx_H2O_aq_layer2) / number_conc
        ! radius (m) calculated based on particle mass
        radius = ( ( true_conc(0,idx_O3_aq_layer1)  / 1000.0 +  &
                     true_conc(0,idx_O3_aq_layer2)  / 1000.0 + &
                     true_conc(0,idx_H2O_aq_layer1)  / 1000.0 + &
                     true_conc(0,idx_H2O_aq_layer2) / 1000.0 )  &
                   * 3.0 / 4.0 / 3.14159265359 )**(1.0/3.0)
      else if (scenario.eq.2) then
        ! radius (m)
        radius = 9.37e-7 / 2.0 * exp(5.0 * log(2.1d0) * log(2.1d0) / 2.0)
        ! number conc
        number_conc = 6.0 / (const%pi * (9.37e-7)**3.0 * &
                             exp(9.0/2.0 * log(2.1d0) * log(2.1d0) ))
        number_conc = number_conc * ( true_conc(0,idx_O3_aq) / 1000.0 )
      end if

      model_conc(0,:) = true_conc(0,:)

      ! Update the aerosol representation (single-particle only)
      if (scenario.eq.1) then
        call number_update%set_number__n_m3(1, number_conc)
        call camp_core%update_data(number_update)
      end if

      ! Update the GMD and GSD for the aerosol modes
      if (scenario.eq.2) then
        ! unused mode
        call update_data_GMD%set_GMD(i_sect_unused, 1.2d-6)
        call update_data_GSD%set_GSD(i_sect_unused, 1.2d0)
        call camp_core%update_data(update_data_GMD)
        call camp_core%update_data(update_data_GSD)
        ! the mode
        call update_data_GMD%set_GMD(i_sect_the_mode, 9.3d-7)
        call update_data_GSD%set_GSD(i_sect_the_mode, 2.1d0)
        call camp_core%update_data(update_data_GMD)
        call camp_core%update_data(update_data_GSD)
      end if

      ! Determine the M -> ppm conversion using the total aerosol water
      if (scenario.eq.1) then
        M_to_ppm = number_conc * 1.0d6 * true_conc(0,idx_H2O_aq_layer1) * &
                const%univ_gas_const * temp / pressure
      else if (scenario.eq.2) then
        M_to_ppm = number_conc * 1.0d6 * true_conc(0,idx_H2O_aq) * &
                const%univ_gas_const * temp / pressure
      end if

      ! O3 rate constants
      n_star = 1.89d0
      del_H = - 10.0d0 * (n_star-1.0d0) + &
            7.53*(n_star**(2.0d0/3.0d0)-1.0d0) - 1.0d0
      del_S = - 32.0d0 * (n_star-1.0d0) + &
            9.21*(n_star**(2.0d0/3.0d0)-1.0d0) - 1.3d0
      del_G = (del_H - temp * del_S/1000.0d0) * 4184.0d0
      alpha = exp(-del_G/(const%univ_gas_const*temp))
      alpha = alpha / (1.0d0 + alpha)
      ! mean free path [m]
      mfp = 3.0 * Dg_O3 / &
            ( sqrt( 8.0 * const%univ_gas_const * temp &
                    / ( const%pi * MW_O3 ) ) )
      Kn = mfp / radius
      corr = ( 0.75 * alpha * ( 1 + Kn ) ) / &
             ( Kn**2 + ( 1.0 + 0.283 * alpha ) * Kn + 0.75 * alpha )
      k_O3_forward = number_conc * 4.0 * const%pi * radius * Dg_O3 * corr ! [s-1]
      k_O3_backward = k_O3_forward / (K_eq_O3 * M_to_ppm)             ! (1/s)

      ! H2O2 rate constants
      n_star = 1.74d0
      del_H = - 10.0d0 * (n_star-1.0d0) + &
            7.53*(n_star**(2.0d0/3.0d0)-1.0d0) - 1.0d0
      del_S = - 32.0d0 * (n_star-1.0d0) + &
            9.21*(n_star**(2.0d0/3.0d0)-1.0d0) - 1.3d0
      del_G = (del_H - temp * del_S/1000.0d0) * 4184.0d0
      alpha = exp(-del_G/(const%univ_gas_const*temp))
      alpha = alpha / (1.0d0 + alpha)
      ! mean free path [m]
      mfp = 3.0 * Dg_H2O2 / &
            ( sqrt( 8.0 * const%univ_gas_const * temp &
                    / ( const%pi * MW_H2O2 ) ) )
      Kn = mfp / radius
      corr = ( 0.75 * alpha * ( 1 + Kn ) ) / &
             ( Kn**2 + ( 1.0 + 0.283 * alpha ) * Kn + 0.75 * alpha )
      k_H2O2_forward = number_conc * 4.0 * const%pi * radius * Dg_H2O2 * corr ! [s-1]
      k_H2O2_backward = k_H2O2_forward / (K_eq_H2O2 * M_to_ppm)       ! (1/s)

      ! Determine the equilibrium concentrations
      ! [A_gas] = [A_total] / (1 + 1/K_HL)
      ! [A_aero] = [A_total] / (K_HL + 1)
      kgm3_to_ppm = const%univ_gas_const * 1.0e6 * temp &
                    / (MW_O3 * pressure)
      if (scenario.eq.1) then
        equil_O3 = (true_conc(0,idx_O3) + &
                (true_conc(0,idx_O3_aq_layer1)+true_conc(0,idx_O3_aq_layer1)) * &
                number_conc*kgm3_to_ppm) / (K_eq_O3*M_to_ppm + 1.0d0)
        equil_O3_aq = (true_conc(0,idx_O3)/kgm3_to_ppm/number_conc + &
                true_conc(0,idx_O3_aq_layer1) + true_conc(0,idx_O3_aq_layer2)) / &
                (1.0d0 + 1.0d0/(K_eq_O3*M_to_ppm))
      else if (scenario.eq.2) then
        equil_O3 = (true_conc(0,idx_O3) + &
                true_conc(0,idx_O3_aq)*number_conc*kgm3_to_ppm) / &
                (K_eq_O3*M_to_ppm + 1.0d0)
        equil_O3_aq = (true_conc(0,idx_O3)/kgm3_to_ppm/number_conc + &
                true_conc(0,idx_O3_aq)) / &
                (1.0d0 + 1.0d0/(K_eq_O3*M_to_ppm))
      end if

      kgm3_to_ppm = const%univ_gas_const * 1.0e6 * temp &
                    / (MW_H2O2 * pressure)
      if (scenario.eq.1) then
        equil_H2O2 = (true_conc(0,idx_H2O2) + &
                (true_conc(0,idx_H2O2_aq_layer1) + true_conc(0,idx_H2O2_aq_layer2)) * &
                number_conc*kgm3_to_ppm) / (K_eq_H2O2*M_to_ppm + 1.0d0)
        equil_H2O2_aq = (true_conc(0,idx_H2O2)/kgm3_to_ppm/number_conc + &
                true_conc(0,idx_H2O2_aq_layer1) + true_conc(0,idx_H2O2_aq_layer2)) / &
                (1.0d0 + 1.0d0/(K_eq_H2O2*M_to_ppm))
      else if (scenario.eq.2) then
        equil_H2O2 = (true_conc(0,idx_H2O2) + &
                true_conc(0,idx_H2O2_aq)*number_conc*kgm3_to_ppm) / &
                (K_eq_H2O2*M_to_ppm + 1.0d0)
        equil_H2O2_aq = (true_conc(0,idx_H2O2)/kgm3_to_ppm/number_conc + &
                true_conc(0,idx_H2O2_aq)) / &
                (1.0d0 + 1.0d0/(K_eq_H2O2*M_to_ppm))
      end if

      ! Set the initial state in the model
      camp_state%state_var(:) = model_conc(0,:)

#ifdef CAMP_DEBUG
      ! Evaluate the Jacobian during solving
      solver_stats%eval_Jac = .true.
#endif

      ! Integrate the mechanism
      do i_time = 1, NUM_TIME_STEP

        ! Get the modeled conc
        call camp_core%solve(camp_state, time_step, &
                              solver_stats = solver_stats)
        model_conc(i_time,:) = camp_state%state_var(:)

#ifdef CAMP_DEBUG
        ! Check the Jacobian evaluations
        call assert_msg(470924483, solver_stats%Jac_eval_fails.eq.0, &
                        trim( to_string( solver_stats%Jac_eval_fails ) )// &
                        " Jacobian evaluation failures at time step "// &
                        trim( to_string( i_time ) ) )
#endif

        ! Get the analytic conc
        ! x = [A_gas] - [A_eq_gas]
        ! x0 = [A_init_gas] - [A_eq_gas]
        ! [A_gas] = x + [A_eq_gas] = x0exp(-t/tau) + [A_eq_gas]
        ! 1/tau = k_f + k_b
        ! [A_gas] = ([A_init_gas] - [A_eq_gas])
        !     * exp(-t *(k_f + k_b)) + [A_eq_gas]
        ! [A_aero] = ([A_init_aero] - [A_eq_aero])
        !     * exp(-t * (k_f + k_b)) + [A_eq_aero]
        time = i_time * time_step
        if (scenario.eq.1) then
          true_conc(i_time,idx_O3) = (true_conc(0,idx_O3) - equil_O3) * &
                  exp(-time * (k_O3_forward + k_O3_backward)) + equil_O3
          true_conc(i_time,idx_O3_aq_layer1) = (true_conc(0,idx_O3_aq_layer1) - equil_O3_aq) * &
                  exp(-time * (k_O3_forward + k_O3_backward)) + equil_O3_aq
          true_conc(i_time,idx_H2O2) = (true_conc(0,idx_H2O2) - equil_H2O2) * &
                  exp(-time * (k_H2O2_forward + k_H2O2_backward)) + equil_H2O2
          true_conc(i_time,idx_H2O2_aq_layer1) = &
                  (true_conc(0,idx_H2O2_aq_layer1) - equil_H2O2_aq) * &
                  exp(-time * (k_H2O2_forward + k_H2O2_backward)) + equil_H2O2_aq
          true_conc(i_time,idx_H2O_aq_layer1) = true_conc(0,idx_H2O_aq_layer1)
          true_conc(i_time,idx_H2O_aq_layer2) = true_conc(0,idx_H2O_aq_layer2) 
        else if (scenario.eq.2) then
          true_conc(i_time,idx_O3) = (true_conc(0,idx_O3) - equil_O3) * &
                  exp(-time * (k_O3_forward + k_O3_backward)) + equil_O3
          true_conc(i_time,idx_O3_aq) = (true_conc(0,idx_O3_aq) - equil_O3_aq) * &
                  exp(-time * (k_O3_forward + k_O3_backward)) + equil_O3_aq
          true_conc(i_time,idx_H2O2) = (true_conc(0,idx_H2O2) - equil_H2O2) * &
                  exp(-time * (k_H2O2_forward + k_H2O2_backward)) + equil_H2O2
          true_conc(i_time,idx_H2O2_aq) = &
                  (true_conc(0,idx_H2O2_aq) - equil_H2O2_aq) * &
                  exp(-time * (k_H2O2_forward + k_H2O2_backward)) + equil_H2O2_aq
          true_conc(i_time,idx_H2O_aq) = true_conc(0,idx_H2O_aq)
        end if
      end do

      ! Save the results
      if (scenario.eq.1) then
        open(unit=7, file="out/HL_phase_transfer_results.txt", status="replace", &
              action="write")
      else if (scenario.eq.2) then
        open(unit=7, file="out/HL_phase_transfer_results_2.txt", status="replace", &
              action="write")
      end if
      if (scenario.eq.1.) then
        do i_time = 0, NUM_TIME_STEP
          write(7,*) i_time*time_step, &
              ' ', true_conc(i_time, idx_O3), &
              ' ', model_conc(i_time, idx_O3), &
              ' ', true_conc(i_time, idx_O3_aq_layer1),&
              ' ', model_conc(i_time, idx_O3_aq_layer1), &
              ' ', true_conc(i_time, idx_O3_aq_layer2),&
              ' ', model_conc(i_time, idx_O3_aq_layer2), &
              ' ', true_conc(i_time, idx_H2O2), &
              ' ', model_conc(i_time, idx_H2O2), &
              ' ', true_conc(i_time, idx_H2O2_aq_layer1), &
              ' ', model_conc(i_time, idx_H2O2_aq_layer1), &
              ' ', true_conc(i_time, idx_H2O2_aq_layer2), &
              ' ', model_conc(i_time, idx_H2O2_aq_layer2), &
              ' ', true_conc(i_time, idx_H2O_aq_layer1), &
              ' ', model_conc(i_time, idx_H2O_aq_layer1), &
              ' ', true_conc(i_time, idx_H2O_aq_layer2), &
              ' ', model_conc(i_time, idx_H2O_aq_layer2)
        end do
      else if (scenario.eq.2) then
        do i_time = 0, NUM_TIME_STEP
          write(7,*) i_time*time_step, &
              ' ', true_conc(i_time, idx_O3), &
              ' ', model_conc(i_time, idx_O3), &
              ' ', true_conc(i_time, idx_O3_aq),&
              ' ', model_conc(i_time, idx_O3_aq), &
              ' ', true_conc(i_time, idx_H2O2), &
              ' ', model_conc(i_time, idx_H2O2), &
              ' ', true_conc(i_time, idx_H2O2_aq), &
              ' ', model_conc(i_time, idx_H2O2_aq), &
              ' ', true_conc(i_time, idx_HNO3), &
              ' ', model_conc(i_time, idx_HNO3), &
              ' ', true_conc(i_time, idx_HNO3_aq), &
              ' ', model_conc(i_time, idx_HNO3_aq), &
              ' ', true_conc(i_time, idx_NH3), &
              ' ', model_conc(i_time, idx_NH3), &
              ' ', true_conc(i_time, idx_NH3_aq), &
              ' ', model_conc(i_time, idx_NH3_aq), &
              ' ', true_conc(i_time, idx_H2O), &
              ' ', model_conc(i_time, idx_H2O), &
              ' ', true_conc(i_time, idx_H2O_aq), &
              ' ', model_conc(i_time, idx_H2O_aq)
        end do
      endif
      close(7)

      ! Analyze the results (single-particle only)
      !
      ! The particle radius changes as the species condense/evaporate, so an
      ! exact solution is not calculated. The tolerances on the comparison
      ! with "true" values are higher to account for this.
      !
      ! scenario 2 is only used for Jacobian checking
      if (scenario.eq.1) then
        do i_time = 1, NUM_TIME_STEP
          do i_spec = 1, size(model_conc, 2)
            if (i_spec.ge.2.and.i_spec.le.8) cycle
            call assert_msg(411096108, &
              almost_equal(model_conc(i_time, i_spec), &
              true_conc(i_time, i_spec), 1.0e-1_dp, 1.0e-14_dp).or. &
              (model_conc(i_time, i_spec).lt.1.2*model_conc(NUM_TIME_STEP, i_spec).and. &
              true_conc(i_time, i_spec).lt.1.2*true_conc(NUM_TIME_STEP, i_spec)).or. &
              (model_conc(i_time, i_spec).lt.1e-2*model_conc(1, i_spec).and. &
              true_conc(i_time, i_spec).lt.1e-2*true_conc(1, i_spec)), &
              "time: "//trim(to_string(i_time))//"; species: "// &
              trim(to_string(i_spec))//"; mod: "// &
              trim(to_string(model_conc(i_time, i_spec)))//"; true: "// &
              trim(to_string(true_conc(i_time, i_spec))))
          end do
        end do
      endif
      deallocate(camp_state)

#ifdef CAMP_USE_MPI
      ! convert the results to an integer
      if (run_HL_phase_transfer_test) then
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
        run_HL_phase_transfer_test = .true.
      else
        run_HL_phase_transfer_test = .false.
      end if
    end if

    deallocate(buffer)
#endif

    deallocate(camp_core)
    deallocate(model_conc)
    deallocate(true_conc)

  end function run_HL_phase_transfer_test

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end program camp_test_HL_phase_transfer
