! Copyright (C) 2023 Barcelona Supercomputing Center, University of
! Illinois at Urbana-Champaign, and National Center for Atmospheric Research
! SPDX-License-Identifier: MIT

!> \file
!> The camp_test_surface program

!> Test of surface reaction module
program camp_test_surface

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

  if (run_surface_tests()) then
    if (camp_mpi_rank().eq.0) write(*,*) &
            "surface reaction tests - PASS"
  else
    if (camp_mpi_rank().eq.0) write(*,*) &
            "surface reaction tests - FAIL"
    stop 3
  end if

  ! finalize mpi
  call camp_mpi_finalize()

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Run all camp_rxn_surface tests
  logical function run_surface_tests() result(passed)

    use camp_camp_solver_data

    type(camp_solver_data_t), pointer :: camp_solver_data

    camp_solver_data => camp_solver_data_t()

    if (camp_solver_data%is_solver_available()) then
      passed = run_surface_test(1)
      passed = passed .and. run_surface_test(2)
    else
      call warn_msg(930415116, "No solver available")
      passed = .true.
    end if

    deallocate(camp_solver_data)

  end function run_surface_tests

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Solve a mechanism consisting of one surface reaction
  !!
  !! One of two scenarios is tested, depending on the passed integer:
  !! one with a single-particle aerosol representation (1)
  !! and one with a modal aerosol representation (2).
  logical function run_surface_test(scenario)

    use camp_constants

    !> Scenario flag
    integer, intent(in) :: scenario

    type(camp_core_t), pointer :: camp_core
    type(camp_state_t), pointer :: camp_state
    character(len=:), allocatable :: input_file_path, key, idx_prefix
    type(string_t), allocatable, dimension(:) :: output_file_path

    type(chem_spec_data_t), pointer :: chem_spec_data
    class(aero_rep_data_t), pointer :: aero_rep_ptr
    real(kind=dp), allocatable, dimension(:,:) :: model_conc, true_conc
    real(kind=dp) :: time, time_step, mean_speed, k_surf
    integer :: i_time, idx_foo, idx_bar, idx_baz, idx_stuff, idx_more_stuff, &
               i_spec
#ifdef CAMP_USE_MPI
    character, allocatable :: buffer(:), buffer_copy(:)
    integer(kind=i_kind) :: i_elem, results, pack_size, pos, rank_solve
#endif

    ! Parameters for calculating true concentrations
    real(kind=dp) :: temperature, pressure
    real(kind=dp), target :: radius, number_conc

    ! For setting particle radius and number concentration
    type(aero_rep_factory_t) :: aero_rep_factory
    type(aero_rep_update_data_single_particle_number_t) :: number_update

    ! For setting the GSD and GMD for modes
    integer(kind=i_kind) :: i_sect_unused, i_sect_the_mode
    type(aero_rep_update_data_modal_binned_mass_GMD_t) :: update_data_GMD
    type(aero_rep_update_data_modal_binned_mass_GSD_t) :: update_data_GSD

    real(kind=dp), parameter :: rxn_gamma          = 2.0e-2_dp  ! [unitless]
    real(kind=dp), parameter :: bar_yield          = 1.0_dp     ! [unitless]
    real(kind=dp), parameter :: baz_yield          = 0.4_dp     ! [unitless]
    real(kind=dp), parameter :: DENSITY_stuff      = 1000.0_dp  ! [kg m-3]
    real(kind=dp), parameter :: DENSITY_more_stuff = 1000.0_dp  ! [kg m-3]
    real(kind=dp), parameter :: MW_stuff           = 0.5_dp     ! [kg mol-1]
    real(kind=dp), parameter :: MW_more_stuff      = 0.2_dp     ! [kg mol-1]
    real(kind=dp), parameter :: MW_foo             = 0.04607_dp ! [kg mol-1]
    real(kind=dp), parameter :: Dg_foo             = 0.95e-5_dp ! diffusion coeff [m2 s-1]
    real(kind=dp), parameter :: sp_number          = 1.3e6      ! single-particle number
                                                                !   concentration [# m-3]
    real(kind=dp), parameter :: mode_GMD           = 1.0e-6_dp  ! mode geometric mean diameter [m]
    real(kind=dp), parameter :: mode_GSD           = 0.1_dp     ! mode geometric standard
                                                                !   deviation [unitless]

    type(solver_stats_t), target :: solver_stats

    call assert_msg(318154673, scenario.ge.1 .and. scenario.le.2, &
                    "Invalid scenario specified: "//to_string( scenario ) )

    run_surface_test = .true.

    ! Allocate space for the results
    if (scenario.eq.1) then
      allocate(model_conc(0:NUM_TIME_STEP, 12))
      allocate(true_conc( 0:NUM_TIME_STEP, 12))
    else if (scenario.eq.2) then
      allocate(model_conc(0:NUM_TIME_STEP, 7))
      allocate(true_conc( 0:NUM_TIME_STEP, 7))
    endif

    ! Set the environmental conditions
    temperature = 272.5d0       ! temperature (K)
    pressure = 101253.3d0       ! pressure (Pa)

    ! Set output time step (s)
    time_step = 10.0

#ifdef CAMP_USE_MPI
    ! Load the model data on root process and pass it to process 1 for solving
    if (camp_mpi_rank().eq.0) then
#endif

      ! Get the surface reaction mechanism json file
      if (scenario.eq.1) then
        input_file_path = 'test_surface_config.json'
      else if (scenario.eq.2) then
        input_file_path = 'test_surface_config_2.json'
      endif

      ! Construct a camp_core variable
      camp_core => camp_core_t(input_file_path)

      deallocate(input_file_path)

      ! Initialize the model
      call camp_core%initialize()

      ! Find the aerosol representation
      key ="my aero rep 2"
      call assert(699060880, camp_core%get_aero_rep(key, aero_rep_ptr))
      if (scenario.eq.1) then
        select type (aero_rep_ptr)
          type is (aero_rep_single_particle_t)
            call camp_core%initialize_update_object( aero_rep_ptr, &
                                                     number_update)
          class default
            call die_msg(528903976, "Incorrect aerosol representation type")
        end select
      else if (scenario.eq.2) then
        select type (aero_rep_ptr)
          type is (aero_rep_modal_binned_mass_t)
            call camp_core%initialize_update_object( aero_rep_ptr, &
                                                     update_data_GMD)
            call camp_core%initialize_update_object( aero_rep_ptr, &
                                                     update_data_GSD)
            call assert_msg(306172820, &
                  aero_rep_ptr%get_section_id("unused mode", i_sect_unused), &
                  "Could not get section id for the unused mode")
            call assert_msg(136015916, &
                  aero_rep_ptr%get_section_id("the mode", i_sect_the_mode), &
                  "Could not get section id for the unused mode")
          class default
            call die_msg(865859011, "Incorrect aerosol representation type")
        end select
      endif

      ! Get chemical species data
      call assert(978177356, camp_core%get_chem_spec_data(chem_spec_data))

      ! Get species indices
      if (scenario.eq.1) then
        idx_prefix = "P1."
      else if (scenario.eq.2) then
        idx_prefix = "the mode."
      end if
      key = "foo"
      idx_foo = chem_spec_data%gas_state_id(key)
      key = "bar"
      idx_bar = chem_spec_data%gas_state_id(key)
      key = "baz"
      idx_baz = chem_spec_data%gas_state_id(key)
      key = idx_prefix//"surface reacting phase.aerosol stuff"
      idx_stuff = aero_rep_ptr%spec_state_id(key)
      key = idx_prefix//"surface reacting phase.more aerosol stuff"
      idx_more_stuff = aero_rep_ptr%spec_state_id(key)

      ! Make sure the expected species are in the model
      call assert(989817309, idx_foo.gt.0)
      call assert(531920849, idx_bar.gt.0)
      call assert(191607041, idx_baz.gt.0)
      call assert(816301632, idx_stuff.gt.0)
      call assert(193512575, idx_more_stuff.gt.0)

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
      call assert(580024989, pos.eq.pack_size)
    end if

    ! broadcast the species ids
    call camp_mpi_bcast_integer(idx_foo)
    call camp_mpi_bcast_integer(idx_bar)
    call camp_mpi_bcast_integer(idx_baz)
    call camp_mpi_bcast_integer(idx_stuff)
    call camp_mpi_bcast_integer(idx_more_stuff)
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

    rank_solve=1
    if(camp_mpi_size() == 1 ) then
      rank_solve=0
    end if

    if (camp_mpi_rank().eq.rank_solve) then
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
      call assert(134562677, pos.eq.pack_size)
      allocate(buffer_copy(pack_size))
      pos = 0
      call camp_core%bin_pack(buffer_copy, pos)
      if (scenario.eq.1) then
        call number_update%bin_pack(buffer_copy, pos)
      else if (scenario.eq.2) then
        call update_data_GMD%bin_pack(buffer_copy, pos)
        call update_data_GSD%bin_pack(buffer_copy, pos)
      end if
      call assert(129298370, pos.eq.pack_size)
      do i_elem = 1, pack_size
        call assert_msg(859141465, buffer(i_elem).eq.buffer_copy(i_elem), &
                "Mismatch in element :"//trim(to_string(i_elem)))
      end do

      ! solve and evaluate results on process 1
#endif

      ! Initialize the solver
      call camp_core%solver_initialize()

      ! Get a model state variable
      camp_state => camp_core%new_state()

      ! Set the environmental conditions
      call camp_state%env_states(1)%set_temperature_K( temperature )
      call camp_state%env_states(1)%set_pressure_Pa( pressure )

      ! Save the initial concentrations
      true_conc(:,:) = 0.0
      true_conc(0,idx_foo) = 1.0
      true_conc(0,idx_stuff) = 2.0e-3
      true_conc(0,idx_more_stuff) = 3.0e-3

      ! Calculate the radius and number concentration to use
      !
      ! The modal scenario is only used for Jacobian checking
      if (scenario.eq.1) then
        number_conc = 1.3e6 ! particle number concetration (#/m3)
        ! single particle aerosol mass concentrations are per particle
        true_conc(0,idx_stuff)      = true_conc(0,idx_stuff)      / number_conc
        true_conc(0,idx_more_stuff) = true_conc(0,idx_more_stuff) / number_conc
        ! radius (m) calculated based on particle mass
        radius = ( ( true_conc(0,idx_stuff)      / DENSITY_stuff      + &
                     true_conc(0,idx_more_stuff) / DENSITY_more_stuff ) &
                   * 3.0 / 4.0 / const%pi )**(1.0/3.0)
      else if (scenario.eq.2) then
        ! radius (m)
        radius = mode_GMD / 2.0 * exp(5.0 * log(mode_GSD) * log(mode_GSD) / 2.0)
        ! number conc
        number_conc = 6.0 / (const%pi * (mode_GMD)**3.0 * &
                             exp(9.0/2.0 * log(mode_GSD) * log(mode_GSD) ))
        number_conc = number_conc * &
                        ( true_conc(0,idx_stuff)      / DENSITY_stuff      + &
                          true_conc(0,idx_more_stuff) / DENSITY_more_stuff )
      end if

      model_conc(0,:) = true_conc(0,:)

      ! Update the aerosol representation (single partile only)
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
        call update_data_GMD%set_GMD(i_sect_the_mode, mode_GMD)
        call update_data_GSD%set_GSD(i_sect_the_mode, mode_GSD)
        call camp_core%update_data(update_data_GMD)
        call camp_core%update_data(update_data_GSD)
      end if

      ! Calculate the surface reaction rate
      mean_speed = sqrt( 8.0_dp * const%univ_gas_const * temperature / &
                         ( const%pi * MW_foo ) )
      k_surf = number_conc * 4.0_dp * const%pi * radius * radius / &
               ( radius / Dg_foo + 4.0_dp / ( mean_speed * rxn_gamma ) )

      ! Set the initial state in the model
      call assert(453929652, size(camp_state%state_var) &
                             .eq. size(model_conc, dim=2))
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
        call assert_msg(236804703, solver_stats%Jac_eval_fails.eq.0, &
                        trim( to_string( solver_stats%Jac_eval_fails ) )// &
                        " Jacobian evaluation failures at time step "// &
                        trim( to_string( i_time ) ) )
#endif

        ! Get the analytic conc
        time = i_time * time_step
        true_conc(i_time,idx_foo) = true_conc(0,idx_foo) * exp(-k_surf*time)
        true_conc(i_time,idx_bar) = (true_conc(0,idx_foo) &
                                    - true_conc(i_time,idx_foo)) * bar_yield
        true_conc(i_time,idx_baz) = (true_conc(0,idx_foo) &
                                    - true_conc(i_time,idx_foo)) * baz_yield
        true_conc(i_time,idx_stuff)      = true_conc(0,idx_stuff)
        true_conc(i_time,idx_more_stuff) = true_conc(0,idx_more_stuff)
      end do

      ! Save the results
      if (scenario.eq.1) then
        open(unit=7, file="out/surface_results.txt", &
                status="replace", action="write")
      else if (scenario.eq.2) then
        open(unit=7, file="out/surface_results_2.txt", &
                status="replace", action="write")
      endif
      do i_time = 0, NUM_TIME_STEP
        write(7,*) i_time*time_step, &
          ' ', true_conc(i_time, idx_foo), &
          ' ', model_conc(i_time, idx_foo), &
          ' ', true_conc(i_time, idx_bar), &
          ' ', model_conc(i_time, idx_bar), &
          ' ', true_conc(i_time, idx_baz), &
          ' ', model_conc(i_time, idx_baz), &
          ' ', true_conc(i_time, idx_stuff), &
          ' ', model_conc(i_time, idx_stuff), &
          ' ', true_conc(i_time, idx_more_stuff), &
          ' ', model_conc(i_time, idx_more_stuff)
      end do
      close(7)

      ! Analyze the results
      do i_time = 1, NUM_TIME_STEP
        do i_spec = 1, size(model_conc, 2)
          call assert_msg(311433544, &
            almost_equal(model_conc(i_time, i_spec), &
            true_conc(i_time, i_spec), real(1.0e-5, kind=dp)).or. &
            (model_conc(i_time, i_spec).lt.1e-5*model_conc(1, i_spec).and. &
            true_conc(i_time, i_spec).lt.1e-5*true_conc(1, i_spec)), &
            "time: "//trim(to_string(i_time))//"; species: "// &
            trim(to_string(i_spec))//"; mod: "// &
            trim(to_string(model_conc(i_time, i_spec)))//"; true: "// &
            trim(to_string(true_conc(i_time, i_spec))))
        end do
      end do
    !if assert_msg does not exit, then the run is valid
#ifdef CAMP_USE_MPI
    end if
#endif
  end function
end program
