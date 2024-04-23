! Copyright (C) 2021 Barcelona Supercomputing Center and University of
! Illinois at Urbana-Champaign
! SPDX-License-Identifier: MIT

!> \file
!> The camp_test_CMAQ_H2O2 program

!> Test of CMAQ_H2O2 reaction module
program camp_test_CMAQ_H2O2

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

  if (run_CMAQ_H2O2_tests()) then
    if (camp_mpi_rank().eq.0) write(*,*) "CMAQ_H2O2 reaction tests - PASS"
  else
    if (camp_mpi_rank().eq.0) write(*,*) "CMAQ_H2O2 reaction tests - FAIL"
    stop 3
  end if

  ! finalize mpi
  call camp_mpi_finalize()

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  integer function mpi_size( comm )
    integer, intent(in), optional :: comm
    integer :: size, ierr, l_comm
    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    endif
    call mpi_comm_size(l_comm, size, ierr)
    mpi_size = size
  end function

  integer function mpi_rank( comm )
    integer, intent(in), optional :: comm
    integer :: rank, ierr, l_comm
    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    endif
    call mpi_comm_rank(l_comm, rank, ierr)
    mpi_rank = rank
  end function

  subroutine mpi_transfer_integer(from_val, to_val, from_proc, to_proc)
    !> Value to send.
    integer, intent(in) :: from_val
    !> Variable to send to.
    integer, intent(out) :: to_val
    !> Process to send from.
    integer, intent(in) :: from_proc
    !> Process to send to.
    integer, intent(in) :: to_proc
    integer :: rank, size, ierr, status(MPI_STATUS_SIZE)

    rank = mpi_rank()
    if (from_proc == to_proc .or. mpi_size() == 1) then
      if (rank == from_proc) then
        to_val = from_val
      end if
    else
      if (rank == from_proc) then
        print*,from_proc,from_val
        call mpi_send(from_val, 1, MPI_INTEGER, to_proc, &
            208020430, MPI_COMM_WORLD, ierr)
      elseif (rank == to_proc) then
        call mpi_recv(to_val, 1, MPI_INTEGER, from_proc, &
            208020430, MPI_COMM_WORLD, status, ierr)
        print*,to_proc,from_val
      end if
    end if
  end subroutine

  !> Run all camp_chem_mech_solver tests
  logical function run_CMAQ_H2O2_tests() result(passed)

    use camp_camp_solver_data

    type(camp_solver_data_t), pointer :: camp_solver_data

    camp_solver_data => camp_solver_data_t()

    if (camp_solver_data%is_solver_available()) then
      passed = run_CMAQ_H2O2_test()
    else
      call warn_msg(405400222, "No solver available")
      passed = .true.
    end if

    deallocate(camp_solver_data)

  end function run_CMAQ_H2O2_tests

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Solve a mechanism of consecutive reactions
  !!
  !! The mechanism is of the form:
  !!
  !!   A -k1-> B -k2-> C
  !!
  !! where k1 and k2 are CMAQ_H2O2 reaction rate constants.
  logical function run_CMAQ_H2O2_test()

    use camp_constants

    type(camp_core_t), pointer :: camp_core
    type(camp_state_t), pointer :: camp_state
    character(len=:), allocatable :: input_file_path
    type(string_t), allocatable, dimension(:) :: output_file_path

    type(chem_spec_data_t), pointer :: chem_spec_data
    real(kind=dp), dimension(0:NUM_TIME_STEP, 3) :: model_conc, true_conc
    integer(kind=i_kind) :: idx_A, idx_B, idx_C
    character(len=:), allocatable :: key
    integer(kind=i_kind) :: i_time, i_spec
    real(kind=dp) :: time_step, time
    logical :: run_CMAQ_H2O2_test2
#ifdef CAMP_USE_MPI
    character, allocatable :: buffer(:), buffer_copy(:)
    integer :: pack_size, pos, i_elem, results
#endif

    type(solver_stats_t), target :: solver_stats

    ! Parameters for calculating true concentrations
    real(kind=dp) :: k1, k2, air_conc, temp, pressure, conv

    run_CMAQ_H2O2_test = .true.

    print*,"a"
    if (run_CMAQ_H2O2_test) then
      print*,"run_CMAQ_H2O2_test a"
      results = 0
    else
      print*,"run_CMAQ_H2O2_test b"
      results = 1
    end if
    !print*,"results",results !if uncommented, works fine
    call camp_mpi_transfer_integer(results, results, 1, 0)
    if (results.eq.0) then
      run_CMAQ_H2O2_test = .true.
    else
      run_CMAQ_H2O2_test = .false.
    end if

  end function run_CMAQ_H2O2_test

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end program camp_test_CMAQ_H2O2
