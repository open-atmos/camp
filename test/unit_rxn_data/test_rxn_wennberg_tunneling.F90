! Copyright (C) 2021 Barcelona Supercomputing Center,
!  University of Illinois at Urbana-Champaign, and
!  National Center for Atmospheric Research
! SPDX-License-Identifier: MIT

!> \file
!> The camp_test_wennberg_tunneling program

!> Test of wennberg_tunneling reaction module
program camp_test_wennberg_tunneling

  use camp_util,                         only: i_kind, dp, assert, &
                                              almost_equal, string_t, &
                                              warn_msg
  use camp_camp_core
  use camp_camp_state
  use camp_chem_spec_data
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

  if (run_wennberg_tunneling_tests()) then
    if (camp_mpi_rank().eq.0) write(*,*) "Wennberg tunneling reaction tests - PASS"
  else
    if (camp_mpi_rank().eq.0) write(*,*) "Wennberg tunneling reaction tests - FAIL"
    stop 3
  end if

  ! finalize mpi
  call camp_mpi_finalize()

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Run all camp_chem_mech_solver tests
  logical function run_wennberg_tunneling_tests() result(passed)

    use camp_camp_solver_data

    type(camp_solver_data_t), pointer :: camp_solver_data

    camp_solver_data => camp_solver_data_t()

    if (camp_solver_data%is_solver_available()) then
      passed = run_wennberg_tunneling_test()
    else
      call warn_msg(383750890, "No solver available")
      passed = .true.
    end if

    deallocate(camp_solver_data)

  end function run_wennberg_tunneling_tests

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Solve a mechanism of consecutive reactions
  !!
  !! The mechanism is of the form:
  !!
  !!   A -k1-> B -k2-> C
  !!
  !! where k1 and k2 are Wennberg tunneling reaction rate constants.
  logical function run_wennberg_tunneling_test()

    use camp_constants

    type(camp_core_t), pointer :: camp_core
    type(camp_state_t), pointer :: camp_state
    character(len=:), allocatable :: input_file_path, key
    type(string_t), allocatable, dimension(:) :: output_file_path

    type(chem_spec_data_t), pointer :: chem_spec_data
    real(kind=dp), dimension(0:NUM_TIME_STEP, 3) :: model_conc, true_conc
    integer(kind=i_kind) :: idx_A, idx_B, idx_C, i_time, i_spec
    real(kind=dp) :: time_step, time, k1, k2, temp, pressure, conv
#ifdef CAMP_USE_MPI
    character, allocatable :: buffer(:), buffer_copy(:)
    integer(kind=i_kind) :: pack_size, pos, i_elem, results, rank_1_results
#endif

    type(solver_stats_t), target :: solver_stats

    run_wennberg_tunneling_test = .true.

    ! Set the rate constants (for calculating the true value)
    temp = 272.5d0
    pressure = 101253.3d0
    conv = const%avagadro / const%univ_gas_const * 10.0d0**(-12.0d0) * &
            pressure / temp
    k1 = 4.0d-3
    k2 = 1.2d-4 * exp( -167d0 / temp ) * exp( 1.0d8 / temp**3 ) / 60.0d0

    ! Set output time step (s)
    time_step = 1.0

#ifdef CAMP_USE_MPI
    ! Load the model data on the root process and pass it to process 1 for solving
    if (camp_mpi_rank().eq.0) then
#endif

      ! Get the wennberg_tunneling reaction mechanism json file
      input_file_path = 'test_wennberg_tunneling_config.json'

      ! Construct a camp_core variable
      camp_core => camp_core_t(input_file_path)

      ! Initialize the model
      call camp_core%initialize()

      ! Get the chemical species data
      call assert(526585315, camp_core%get_chem_spec_data(chem_spec_data))

      ! Get species indices
      key = "A"
      idx_A = chem_spec_data%gas_state_id(key);
      key = "B"
      idx_B = chem_spec_data%gas_state_id(key);
      key = "C"
      idx_C = chem_spec_data%gas_state_id(key);

      ! Make sure the expected species are in the model
      call assert(251279907, idx_A.gt.0)
      call assert(981123002, idx_B.gt.0)
      call assert(528490849, idx_C.gt.0)

#ifdef CAMP_USE_MPI
      ! pack the camp core
      pack_size = camp_core%pack_size()
      allocate(buffer(pack_size))
      pos = 0
      call camp_core%bin_pack(buffer, pos)
      call assert(640809194, pos.eq.pack_size)
    end if

    ! broadcast the species ids
    call camp_mpi_bcast_integer(idx_A)
    call camp_mpi_bcast_integer(idx_B)
    call camp_mpi_bcast_integer(idx_C)

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
      call assert(418078038, pos.eq.pack_size)
      allocate(buffer_copy(pack_size))
      pos = 0
      call camp_core%bin_pack(buffer_copy, pos)
      call assert(865445884, pos.eq.pack_size)
      do i_elem = 1, pack_size
        call assert_msg(977764229, buffer(i_elem).eq.buffer_copy(i_elem), &
                "Mismatch in element: "//trim(to_string(i_elem)))
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
      true_conc(0,idx_A) = 1.0
      true_conc(0,idx_B) = 0.0
      true_conc(0,idx_C) = 0.0
      model_conc(0,:) = true_conc(0,:)

      ! Set the initial concentrations in the model
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
        call assert_msg(519867769, solver_stats%Jac_eval_fails.eq.0, &
                        trim( to_string( solver_stats%Jac_eval_fails ) )// &
                        " Jacobian evaluation failures at time step "// &
                        trim( to_string( i_time ) ) )
#endif

        ! Get the analytic conc
        time = i_time * time_step
        true_conc(i_time,idx_A) = true_conc(0,idx_A) * exp(-(k1)*time)
        true_conc(i_time,idx_B) = true_conc(0,idx_A) * (k1/(k2-k1)) * &
                (exp(-k1*time) - exp(-k2*time))
        true_conc(i_time,idx_C) = true_conc(0,idx_A) * &
               (1.0 + (k1*exp(-k2*time) - k2*exp(-k1*time))/(k2-k1))

      end do

      ! Save the results
      open(unit=7, file="out/wennberg_tunneling_results.txt", status="replace", &
            action="write")
      do i_time = 0, NUM_TIME_STEP
        write(7,*) i_time*time_step, &
              ' ', true_conc(i_time, idx_A),' ', model_conc(i_time, idx_A), &
              ' ', true_conc(i_time, idx_B),' ', model_conc(i_time, idx_B), &
              ' ', true_conc(i_time, idx_C),' ', model_conc(i_time, idx_C)
      end do
      close(7)

      ! Analyze the results
      do i_time = 1, NUM_TIME_STEP
        do i_spec = 1, size(model_conc, 2)
          call assert_msg(579611862, &
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

      deallocate(camp_state)

#ifdef CAMP_USE_MPI
      ! convert the results to an integer
      if (run_wennberg_tunneling_test) then
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
        run_wennberg_tunneling_test = .true.
      else
        run_wennberg_tunneling_test = .false.
      end if
    end if

    deallocate(buffer)
#endif

    deallocate(camp_core)

  end function run_wennberg_tunneling_test

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end program camp_test_wennberg_tunneling
