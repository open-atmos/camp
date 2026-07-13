! Copyright (C) 2021 Barcelona Supercomputing Center and University of
! Illinois at Urbana-Champaign
! SPDX-License-Identifier: MIT

!> \file
!> The camp_test_vbs_nox_dependent program

!> Test of vbs_nox_dependent reaction module

program camp_test_vbs_nox_dependent
  use camp_util,                         only: i_kind, dp, assert, &
                                              almost_equal, string_t, &
                                              warn_msg                                              
  use camp_camp_core
  use camp_camp_state
  use camp_chem_spec_data
#ifdef CAMP_USE_JSON
  use json_module
#endif
  use camp_mpi

  implicit none

  ! Number of timesteps to output in mechanisms
  integer(kind=i_kind) :: NUM_TIME_STEP = 100

  ! initialize mpi
  call camp_mpi_init()

  if (run_vbs_nox_dependent_tests()) then
      if (camp_mpi_rank().eq.0) write(*,*) "vbs_nox_dependent reaction tests - PASS"
      else
      if (camp_mpi_rank().eq.0) write(*,*) "vbs_nox_dependent reaction tests - FAIL"
      stop 3
  end if
  ! finalize mpi
  call camp_mpi_finalize()

contains
    
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Run all camp_chem_mech_solver tests
  logical function run_vbs_nox_dependent_tests() result(passed)

    use camp_camp_solver_data

    type(camp_solver_data_t), pointer :: camp_solver_data

    camp_solver_data => camp_solver_data_t()

    if (camp_solver_data%is_solver_available()) then
      passed = run_vbs_nox_dependent_test()
    else
      call warn_msg(405400222, "No solver available")
      passed = .false.
    end if

    deallocate(camp_solver_data)

  end function run_vbs_nox_dependent_tests

  !> Solve a mechanism with vbs nox dependent yields
  !!
  !! The mechanisms is of the form
  !!    A -k1-> b B + c C
  !! where k1 is an arrhenius reaction rate constante
  !! b is the nox dependent yield of B
  !! b = b_low * (1-beta) + b_high * beta
  !! c is the nox dependent yield of C
  !! c = c_low * (1-beta) + c_high * beta
  !! beta is the ro2 nox reactivity ratio 
  !! beta = kro2+no * NO / (kro2+no * NO + kro2+ho2 * HO2)
  logical function run_vbs_nox_dependent_test()
    use camp_constants


    type(camp_core_t), pointer :: camp_core
    type(camp_state_t), pointer :: camp_state
    character(len=:), allocatable :: input_file_path
    type(string_t), allocatable, dimension(:) :: output_file_path
    type(chem_spec_data_t), pointer :: chem_spec_data
    real(kind=dp), dimension(0:NUM_TIME_STEP, 5) :: model_conc, true_conc
    integer(kind=i_kind) :: idx_A, idx_B, idx_C, idx_NO, idx_HO2
    character(len=:), allocatable :: key
    integer(kind=i_kind) :: i_time, i_spec
    real(kind=dp) :: time_step, time
#ifdef CAMP_USE_MPI
    character, allocatable :: buffer(:), buffer_copy(:)
    integer(kind=i_kind) :: pack_size, pos, i_elem, results, rank_solve
#endif

  ! parameters for calculating true concentrations
  real(kind=dp) :: k1, b_low, b_high, c_low, c_high, air_conc, temp, pressure, conv, b, c, beta
  real(kind=dp), parameter :: kro2_ho2 = 2.28e-11, kro2_no = 9.04e-12
  real(kind = dp) :: NO, HO2

  run_vbs_nox_dependent_test = .true.

  temp = 298.0
  pressure = 101253.3d0
  air_conc = 1.0e6
  conv = const%avagadro / const%univ_gas_const * 10.0d0**(-12.0d0) * &
          pressure / temp
  k1 = 1e-4
  
  NO = 1e-5 ! 0.01 ppb
  HO2 = 1e-5 ! 0.01 ppb

  beta = kro2_no * NO / (kro2_no * NO + kro2_ho2 * HO2)
  b_low = 0.9
  b_high = 0.2
  c_low = 0.1
  c_high = 0.8
  b = b_high * beta + b_low * (1-beta)
  c = c_high * beta + c_low * (1-beta)


  time_step = 1.0


#ifdef CAMP_USE_MPI
  ! Load the model data on the root process and pass it to process 1 for solving
  if (camp_mpi_rank().eq.0) then
#endif
    ! Get the CMAQ_H2O2 reaction mechanism json file
    input_file_path = 'test_vbs_nox_dependent_config.json'
    
    ! Construct a camp_core variable
    camp_core => camp_core_t(input_file_path)

    deallocate(input_file_path)

    ! Initialize the model
    call camp_core%initialize()

    ! Get the chemical species data
    call assert(159292767, camp_core%get_chem_spec_data(chem_spec_data))
    key = "A"
    idx_A = chem_spec_data%gas_state_id(key);
    key = "B"
    idx_B = chem_spec_data%gas_state_id(key);
    key = "C"
    idx_C = chem_spec_data%gas_state_id(key);
    key = "NO"
    idx_NO = chem_spec_data%gas_state_id(key);
    key = "HO2"
    idx_HO2 = chem_spec_data%gas_state_id(key);
    ! Make sure the expected species are in the model
    call assert(715485073, idx_A.gt.0)
    call assert(545328169, idx_B.gt.0)
    call assert(375171265, idx_C.gt.0)
    call assert(173486375, idx_NO.gt.0)
    call assert(705184114, idx_HO2.gt.0)

#ifdef CAMP_USE_MPI
      ! pack the camp core
      pack_size = camp_core%pack_size()
      allocate(buffer(pack_size))
      pos = 0
      call camp_core%bin_pack(buffer, pos)
      call assert(296902147, pos.eq.pack_size)
    end if
    
    ! broadcast the species ids
    call camp_mpi_bcast_integer(idx_A)
    call camp_mpi_bcast_integer(idx_B)
    call camp_mpi_bcast_integer(idx_C)
    call camp_mpi_bcast_integer(idx_NO)
    call camp_mpi_bcast_integer(idx_HO2)

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
      call assert(109951378, pos.eq.pack_size)
      allocate(buffer_copy(pack_size))
      pos = 0
      call camp_core%bin_pack(buffer_copy, pos)
      call assert(422968304, pos.eq.pack_size)
      do i_elem = 1, pack_size
        call assert_msg(159166857, buffer(i_elem).eq.buffer_copy(i_elem), &
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
      true_conc(0,idx_NO) = 1e-5
      true_conc(0,idx_HO2) = 1e-5
      model_conc(0,:) = true_conc(0,:)

      ! Set the initial concentrations in the model
      camp_state%state_var(:) = model_conc(0,:)

      ! Integrate the mechanism
      do i_time = 1, NUM_TIME_STEP
        ! Get the modeled conc
        call camp_core%solve(camp_state, time_step)
        model_conc(i_time,:) = camp_state%state_var(:)

        ! Get the analytic conc
        time = i_time * time_step
        true_conc(i_time,idx_A) = true_conc(0,idx_A) * exp(-(k1)*time)
        true_conc(i_time,idx_B) = true_conc(0,idx_A) * b * (1.0 - exp(-(k1)*time))
        true_conc(i_time,idx_C) = true_conc(0,idx_A) * c * (1.0 - exp(-(k1)*time))
        true_conc(i_time,idx_NO) = true_conc(0, idx_NO)
        true_conc(i_time,idx_HO2) = true_conc(0, idx_HO2)

      end do

      ! Save the results
      open(unit=7, file="out/vbs_nox_dependent_results.txt", status="replace", &
              action="write")
      do i_time = 0, NUM_TIME_STEP
        write(7,'(14(ES13.6, 1X))') i_time*time_step, &
              true_conc(i_time, idx_A), model_conc(i_time, idx_A), &
              true_conc(i_time, idx_B), model_conc(i_time, idx_B), &
              true_conc(i_time, idx_C), model_conc(i_time, idx_C), &
              true_conc(i_time, idx_NO), model_conc(i_time, idx_NO), &
              true_conc(i_time, idx_HO2), model_conc(i_time, idx_HO2), &
              beta, b, c
      end do
      close(7)

      ! Analyze the results
      do i_time = 1, NUM_TIME_STEP
        do i_spec = 1, size(model_conc, 2)
          call assert_msg(427627073, &
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
    !if assert_msg does not exit, then the run is valid
#ifdef CAMP_USE_MPI
    end if
#endif
  end function
end program
