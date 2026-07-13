! Copyright (C) 2021 Barcelona Supercomputing Center and University of
! Illinois at Urbana-Champaign
! SPDX-License-Identifier: MIT

!> \file
!> The camp_solver_stats module

!> The solver_stats_t type and associated subroutines
module camp_solver_stats

  use camp_constants, only: i_kind, dp

  implicit none
  private

  public :: solver_stats_t

  !> Solver statistics
  !!
  !! Holds information related to a solver run
  type :: solver_stats_t
    !> Status code
    integer(kind=i_kind), allocatable :: status_code(:)
    !> Integration start time [s]
    real(kind=dp) :: start_time__s
    !> Integration end time [s]
    real(kind=dp) :: end_time__s
    !> Last flag returned by the solver
    integer(kind=i_kind), allocatable :: solver_flag(:)
    !> Number of steps
    integer(kind=i_kind), allocatable :: num_steps(:)
#ifdef CAMP_DEBUG
    !> Flag to output debugging info during solving
    !! THIS PRINTS A LOT OF TEXT TO THE STANDARD OUTPUT
    logical :: debug_out = .false.
    !> Evalute the Jacobian during solving
    logical :: eval_Jac = .false.
    !> Jacobian evaluation failures
    integer(kind=i_kind) :: Jac_eval_fails
#endif
  end type

  interface solver_stats_t
    procedure :: constructor
  end interface solver_stats_t

contains
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  function constructor(n_cells) result(this)
    !> A new set of model parameters
    type(solver_stats_t), pointer :: this
    integer(kind=i_kind), optional :: n_cells
    integer :: n_cells1

    n_cells1 = 1
    if (present(n_cells)) then
      n_cells1 = n_cells
    end if

    allocate (this)
    allocate (this%status_code(n_cells1))
    allocate (this%solver_flag(n_cells1))
    allocate (this%num_steps(n_cells1))
    this%status_code(:) = -1
    this%solver_flag(:) = -1
    this%num_steps(:) = -1
  end function constructor

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end module camp_solver_stats
