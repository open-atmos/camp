! Copyright (C) 2021 Barcelona Supercomputing Center and University of
! Illinois at Urbana-Champaign
! SPDX-License-Identifier: MIT

!> \file
!> The camp_test_arrhenius program

!> Test of arrhenius reaction module
program camp_test_arrhenius

  use camp_util, only: i_kind, dp, assert, &
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
  integer(kind=i_kind) :: NUM_TIME_STEP = 1440

  ! initialize mpi
  call camp_mpi_init()

  if (run_arrhenius_tests()) then
    if (camp_mpi_rank() .eq. 0) write (*, *) &
      "Arrhenius reaction tests - PASS"
  else
    if (camp_mpi_rank() .eq. 0) write (*, *) &
      "Arrhenius reaction tests - FAIL"
    stop 3
  end if

  ! finalize mpi
  call camp_mpi_finalize()

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Run all camp_chem_mech_solver tests
  logical function run_arrhenius_tests() result(passed)

    use camp_camp_solver_data

    type(camp_solver_data_t), pointer :: camp_solver_data

    camp_solver_data => camp_solver_data_t()

    if (camp_solver_data%is_solver_available()) then
      passed = run_test()
    else
      call warn_msg(713064651, "No solver available")
      passed = .true.
    end if

    deallocate (camp_solver_data)

  end function run_arrhenius_tests

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Solve a mechanism consisting of one arrhenius reaction
  !!
  !! The mechanism is of the form:
  !!
  !!    A + B -k1-> C
  !!
  !! where k1 is an Arrhenius rate constant
  logical function run_test()

    use camp_constants

    type(camp_core_t), pointer :: camp_core
    type(camp_state_t), pointer :: camp_state
    character(len=:), allocatable :: input_file_path, key, idx_prefix
    type(string_t), allocatable, dimension(:) :: output_file_path

    type(chem_spec_data_t), pointer :: chem_spec_data
    real(kind=dp), allocatable, dimension(:, :) :: model_conc, true_conc
    integer(kind=i_kind) :: idx_A, idx_B, idx_C, i_time, i_spec
    real(kind=dp) :: time_step, time
#ifdef CAMP_USE_MPI
    character, allocatable :: buffer(:), buffer_copy(:)
    integer(kind=i_kind) :: pack_size, pos, i_elem, results, rank_solve
#endif

    type(solver_stats_t), target :: solver_stats

    ! Parameters for calculating true concentrations
    real(kind=dp) :: k1, temp, pressure, conv

    run_test = .true.

    ! Allocate space for the results
    allocate (model_conc(0:NUM_TIME_STEP, 3))
    allocate (true_conc(0:NUM_TIME_STEP, 3))

    ! Set the rate constants (for calculating the true value)
    temp = 272.5d0              ! temperature (K)
    pressure = 101253.3d0       ! pressure (Pa)
    conv = pressure*const%avagadro*1e-12/const%univ_gas_const/temp
    k1 = 1.312e-18*conv

    ! Set output time step (s)
    time_step = 60.0d0

#ifdef CAMP_USE_MPI
    ! Load the model data on the root process and pass it to process 1 for solving
    if (camp_mpi_rank() .eq. 0) then
#endif

      ! Get the arrhenius reaction mechanism json file
      input_file_path = 'test_arrhenius_config.json'

      ! Construct a camp_core variable
      camp_core => camp_core_t(input_file_path)

      deallocate (input_file_path)

      ! Initialize the model
      call camp_core%initialize()

      ! Get the chemical species data
      call assert(106978557, camp_core%get_chem_spec_data(chem_spec_data))

      key = "A"
      idx_A = chem_spec_data%gas_state_id(key); 
      key = "B"
      idx_B = chem_spec_data%gas_state_id(key); 
      key = "C"
      idx_C = chem_spec_data%gas_state_id(key); 
      ! Make sure the expected species are in the model
      call assert(224107570, idx_A .gt. 0)
      call assert(367832024, idx_B .gt. 0)
      call assert(178368990, idx_C .gt. 0)

#ifdef CAMP_USE_MPI
      ! pack the camp core
      pack_size = camp_core%pack_size()
      allocate (buffer(pack_size))
      pos = 0
      call camp_core%bin_pack(buffer, pos)
      call assert(145814867, pos .eq. pack_size)
    end if

    ! broadcast the species ids
    call camp_mpi_bcast_integer(idx_A)
    call camp_mpi_bcast_integer(idx_B)
    call camp_mpi_bcast_integer(idx_C)

    ! broadcast the buffer size
    call camp_mpi_bcast_integer(pack_size)

    if (camp_mpi_rank() .eq. 1) then
      ! allocate the buffer to receive data
      allocate (buffer(pack_size))
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
      call assert(274920180, pos .eq. pack_size)
      allocate (buffer_copy(pack_size))
      pos = 0
      call camp_core%bin_pack(buffer_copy, pos)
      call assert(314541292, pos .eq. pack_size)
      do i_elem = 1, pack_size
        call assert_msg(423752476, buffer(i_elem) .eq. buffer_copy(i_elem), &
                        "Mismatch in element: "//trim(to_string(i_elem)))
      end do

      ! solve and evaluate results on process 1
#endif

      ! Initialize the solver
      call camp_core%solver_initialize()

      ! Get a model state variable
      camp_state => camp_core%new_state()

      ! Set the environmental conditions
      call camp_state%env_states(1)%set_temperature_K(temp)
      call camp_state%env_states(1)%set_pressure_Pa(pressure)

      ! Save the initial concentrations
      true_conc(:, :) = 0.0
      true_conc(0, idx_A) = 10.0
      true_conc(0, idx_B) = 10.0
      true_conc(0, idx_C) = 0.0
      model_conc(0, :) = true_conc(0, :)

      ! Set the initial concentrations in the model
      camp_state%state_var(:) = model_conc(0, :)

#ifdef CAMP_DEBUG
      ! Evaluate the Jacobian during solving
      solver_stats%eval_Jac = .true.
#endif

      ! Integrate the mechanism
      do i_time = 1, NUM_TIME_STEP

        ! Get the modeled conc
        call camp_core%solve(camp_state, time_step, &
                             solver_stats=solver_stats)
        model_conc(i_time, :) = camp_state%state_var(:)

#ifdef CAMP_DEBUG
        ! Check the Jacobian evaluations
        call assert_msg(348199734, solver_stats%Jac_eval_fails .eq. 0, &
                        trim(to_string(solver_stats%Jac_eval_fails))// &
                        " Jacobian evaluation failures at time step "// &
                        trim(to_string(i_time)))
#endif

        ! Get the analytic concentrations
        time = i_time*time_step
        true_conc(i_time, idx_A) = true_conc(0, idx_A)/(1 + true_conc(0, idx_A)*k1*time)
        true_conc(i_time, idx_B) = true_conc(i_time, idx_A)
        true_conc(i_time, idx_C) = true_conc(0, idx_A)*true_conc(0, idx_A)*k1*time/ &
                                   (1 + true_conc(0, idx_A)*k1*time)

      end do

      ! Save the results
      open (unit=7, file="out/arrhenius_results.txt")
      do i_time = 0, NUM_TIME_STEP
        write (7, '(6(E16.9, A1), E16.9)') i_time*time_step, ',', true_conc(i_time, idx_A), ',', model_conc(i_time, idx_A), &
          ',', true_conc(i_time, idx_B), ',', model_conc(i_time, idx_B), ',', true_conc(i_time, idx_C), &
          ',', model_conc(i_time, idx_C)
      end do
      close (7)

      ! Analyze the results (single-particle only)
      do i_time = 1, NUM_TIME_STEP
        do i_spec = 1, size(model_conc, 2)
          call assert_msg(703162515, &
                          almost_equal(model_conc(i_time, i_spec), &
                                       true_conc(i_time, i_spec), real(1.0e-3, kind=dp)) .or. &
                          (model_conc(i_time, i_spec) .lt. 1e-5*model_conc(1, i_spec) .and. &
                           true_conc(i_time, i_spec) .lt. 1e-5*true_conc(1, i_spec)), &
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
