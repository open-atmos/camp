! Copyright (C) 2021 Barcelona Supercomputing Center and University of
! Illinois at Urbana-Champaign
! SPDX-License-Identifier: MIT

!> \file
!> The monarch_original module

!> Functions to run original MONARCH modules for testing integration with PartMC
module monarch_original

  implicit none
  private

  public :: monarch_original_t

  !> MONARCH original wrapper type
  !!
  !! Allows mock MONARCH model to initialize and run original MONARCH modules
  !! for testing integration with PartMC
  type :: monarch_original_t
    private
  contains
    !> Initialize the original MONARCH modules
    procedure :: initialize
    !> Solve for chemistry using original MONARCH modules
    procedure :: solve
  end type :: monarch_original_t

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Initialize the original MONARCH modules
  subroutine initialize( GLOBAL                 &
                        ,CHEM                   &
                        ,CAT_AERO               &
                        ,NUM_AERO               &
                        ,NUM_GAS                &
                        ,NUM_GAS_TOTAL          &
                        ,NUM_TRACERS_TOTAL      &
                        ,NUM_TRACERS_CHEM       &
                        ,NUM_WATER              &
                        ,NUM_TRACERS_MET        &
                        ,IDS, IDE, JDS, JDE, LM &
                        ,IMS, IME, JMS, JME     &
                        ,ITS, ITE, JTS, JTE)



  end subroutine initialize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end module monarch_original
