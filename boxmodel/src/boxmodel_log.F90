module boxmodel_log
  use flogger
  use camp_mpi, only:  camp_mpi_rank

  use camp_util, only: to_string

  implicit none
  private

  !> logger initialization
  type(floggerunit), public :: thread_log

  public :: init_log

contains

  subroutine init_log()
    thread_log = floggerunit("process "//to_string(camp_mpi_rank()))

    call flogger_set_options(Level = FLOGS_SET_DEBUG, UseEncoding = .FALSE., &
      ConsolePrint = .TRUE., FileOutput = .FALSE.)

  end subroutine init_log


end module boxmodel_log
