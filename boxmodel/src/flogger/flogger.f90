! "Flogger" is simple and fast logging library for Modern Fortran applications.
! https://github.com/arifyunando/flogger
!
! MIT License
!
! Copyright (c) 2023 Arif Y. Sunanhadikusuma
!
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
!
! The above copyright notice and this permission notice shall be included in all
! copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
! SOFTWARE.


module Flogger
  implicit none
  private

  ! Compile Mode ID
  integer, parameter, public :: FLOGS_SET_DEBUG   = 0
  integer, parameter, public :: FLOGS_SET_RELEASE = 1
  integer, parameter, public :: FLOGS_SET_NOWARN  = 2
  integer, parameter, public :: FLOGS_SET_SILENT  = 3

  ! Level-ID
  integer, parameter, private :: FLOGS_ID_DEBUG    = 1
  integer, parameter, private :: FLOGS_ID_INFO     = 2
  integer, parameter, private :: FLOGS_ID_NOTICE   = 3
  integer, parameter, private :: FLOGS_ID_WARNING  = 4
  integer, parameter, private :: FLOGS_ID_ERROR    = 5
  integer, parameter, private :: FLOGS_ID_FATAL    = 6

  ! Class-wide Variable
  integer, private :: FLOGS_LEVEL_GLOBAL = FLOGS_SET_DEBUG
  logical, private :: FLOGS_USE_ENCODING = .true.
  logical, private :: FLOGS_CONSOLE_PRINT = .true.
  logical, private :: FLOGS_FILEOUT_PRINT = .false.

  ! File Creation Variable
  integer, private                  :: FLOGS_FILE_UNIT
  character(11), private, parameter :: FLOGS_FILE_NAME = 'logfile.txt'
  character(7), private, parameter  :: FLOGS_FILE_STAT = 'REPLACE'
  character(5), private, parameter  :: FLOGS_FILE_ACTS = 'WRITE'

  ! Abstract Data Type : Flogger Unit
  type, public :: FloggerUnit
    character(64) :: id
  contains
    procedure, public :: debug => debug_print
    procedure, public :: info => info_print
    procedure, public :: notice => notice_print
    procedure, public :: warning => warning_print
    procedure, public :: error => error_print
    procedure, public :: fatal => fatal_print
  end type FloggerUnit

  ! define procedure visibility
  private  :: debug_print, info_print, notice_print,                          &
    warning_print, error_print, fatal_print
  public   :: flogger_set_options

contains

!--- OBJECT METHODS / PROCEDURES

  subroutine debug_print(this, message, output)
    implicit none
    class(FloggerUnit) :: this
    character(:), allocatable, optional, intent(out) :: output
    character(*), intent(in) :: message

    !--- processes
    if ( FLOGS_LEVEL_GLOBAL < 1 ) then
      call BASE_PRINT(message, this%id, FLOGS_ID_DEBUG, output)
    end if
  end subroutine debug_print


  subroutine info_print(this, message, output)
    implicit none
    class(FloggerUnit) :: this
    character(:), allocatable, optional, intent(out) :: output
    character(*), intent(in) :: message

    !--- processes
    if ( FLOGS_LEVEL_GLOBAL < 2 ) then
      call BASE_PRINT(message, this%id, FLOGS_ID_INFO, output)
    end if
  end subroutine info_print


  subroutine notice_print(this, message, output)
    implicit none
    class(FloggerUnit) :: this
    character(:), allocatable, optional, intent(out) :: output
    character(*), intent(in) :: message

    !--- processes
    if ( FLOGS_LEVEL_GLOBAL < 2 ) then
      call BASE_PRINT(message, this%id, FLOGS_ID_NOTICE, output)
    end if
  end subroutine notice_print


  subroutine warning_print(this, message, output)
    implicit none
    class(FloggerUnit) :: this
    character(:), allocatable, optional, intent(out) :: output
    character(*), intent(in) :: message

    !--- processes
    if ( FLOGS_LEVEL_GLOBAL < 2 ) then
      call BASE_PRINT(message, this%id, FLOGS_ID_WARNING, output)
    end if
  end subroutine warning_print


  subroutine error_print(this, message, output)
    implicit none
    class(FloggerUnit) :: this
    character(:), allocatable, optional, intent(out) :: output
    character(*), intent(in) :: message

    !--- processes
    if ( FLOGS_LEVEL_GLOBAL < 3 ) then
      call BASE_PRINT(message, this%id, FLOGS_ID_ERROR, output)
    end if
  end subroutine error_print


  subroutine fatal_print(this, message, output)
    implicit none
    class(FloggerUnit) :: this
    character(:), allocatable, optional, intent(out) :: output
    character(*), intent(in) :: message

    !--- processes
    if ( FLOGS_LEVEL_GLOBAL < 4 ) then
      call BASE_PRINT(message, this%id, FLOGS_ID_FATAL, output)
    end if
  end subroutine fatal_print

!--- PRIVATE FUNCTIONS / SUBROUTINES

  subroutine base_print(message, label, level, output)
    use FloggerFormatter
    implicit none
    character(:), allocatable, optional, intent(out) :: output
    character(*), intent(in) :: message, label
    integer, intent(in) :: level
    character(2048) :: tmp

    !--- processes
    tmp = printFormatted(message, label, level, inConsole=.false.)
    if ( FLOGS_FILEOUT_PRINT ) write(FLOGS_FILE_UNIT, *) trim(tmp)
    if ( FLOGS_USE_ENCODING ) tmp = printFormatted(message, label, level, inConsole=.true.)
    if ( FLOGS_CONSOLE_PRINT ) write(*, '(A)') trim(tmp)
    if ( present(output) ) output = trim(tmp)
  end subroutine base_print


  subroutine flogger_open_file()
    use FloggerFormatter, only: PRINT_FILE_HEADER
    implicit none

    !--- processes
    if ( FLOGS_FILEOUT_PRINT .eqv. .true. ) return
    open(newunit=FLOGS_FILE_UNIT, File=FLOGS_FILE_NAME,                            &
      Status=FLOGS_FILE_STAT, Action=FLOGS_FILE_ACTS)
    call print_file_header(FLOGS_FILE_UNIT)
    FLOGS_FILEOUT_PRINT = .true.
  end subroutine flogger_open_file


  subroutine flogger_close_file()
    implicit none

    !--- processes
    if ( FLOGS_FILEOUT_PRINT .eqv. .false. ) return
    close(unit=FLOGS_FILE_UNIT)
    FLOGS_FILEOUT_PRINT = .false.
  end subroutine flogger_close_file

!--- PUBLIC FUNCTIONS / SUBROUTINES

  subroutine flogger_set_options(Level, UseEncoding, ConsolePrint, FileOutput)
    use FloggerFormatter
    implicit none
    integer, optional, intent(in) :: Level
    logical, optional, intent(in) :: UseEncoding
    logical, optional, intent(in) :: ConsolePrint
    logical, optional, intent(in) :: FileOutput

    !--- processes
    if ( present(Level) ) then
      FLOGS_LEVEL_GLOBAL = Level
    end if

    if ( present(UseEncoding) ) then
      FLOGS_USE_ENCODING = UseEncoding
    end if

    if ( present(ConsolePrint) ) then
      FLOGS_CONSOLE_PRINT = ConsolePrint
    end if

    if ( present(FileOutput) ) then
      if ( FileOutput ) then
        call FLOGGER_OPEN_FILE()
      else
        call FLOGGER_CLOSE_FILE()
      end if
    end if
  end subroutine flogger_set_options

end module Flogger
