! Copyright (C) 2021 Barcelona Supercomputing Center and University of
! Illinois at Urbana-Champaign
! SPDX-License-Identifier: MIT

!> \file
!> The camp_debug_diff_check module

!> Interfaces to the debug_diff_check c functions - NOT THREAD SAFE!
module camp_debug_diff_check

  use iso_c_binding

  implicit none
  private

  public :: diff_check, diff_check_update_only

  !> Interface to c functions for diff checker
  interface

    !> Do a difference check on the int, float, and env model data
    subroutine diff_check( message ) bind (c)
      use iso_c_binding
      !> Message to print along with difference check
      character(kind=c_char) :: message(*)
    end subroutine diff_check

    !> Check the dimensions of the data and update the saved
    !! values, but do not compare data
    subroutine diff_check_update_only( message ) bind (c)
      use iso_c_binding
      !> Message to print along with the update
      character(kind=c_char) :: message(*)
    end subroutine diff_check_update_only

  end interface

contains

end module camp_debug_diff_check
