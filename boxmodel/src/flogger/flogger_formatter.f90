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


module FloggerFormatter
  use FloggerAnsiStyling
  implicit none
  private

  ! Default Section Styles
  character(64), private :: FLOGS_STYLE_LABEL = FAS_K_CLEAR
  character(64), private :: FLOGS_STYLE_TEXT  = FAS_K_CLEAR
  character(64), private :: FLOGS_STYLE_DATETIME                              &
    = FAS_K_START // trim(FGB_CYAN) // FAS_K_END

  ! Default Category Styles
  character(15), private :: STY_DEBUG, STY_INFO, STY_NOTE
  character(15), private :: STY_WARN, STY_ERROR, STY_FATAL
  parameter(STY_DEBUG = FAS_K_START // trim(FGD_WHITE)   // FAS_K_END)
  parameter(STY_INFO  = FAS_K_START // trim(FGD_GREEN)   // FAS_K_END)
  parameter(STY_NOTE  = FAS_K_START // trim(FGD_MAGENTA) // FAS_K_END)
  parameter(STY_WARN  = FAS_K_START // trim(FGB_YELLOW)  // FAS_K_END)
  parameter(STY_ERROR = FAS_K_START // trim(FAS_BOLD)    // ';'               &
    // trim(FGB_GREEN)   // FAS_K_END)
  parameter(STY_FATAL = FAS_K_START // trim(FAS_BOLD)    // ';'               &
    // trim(FGB_RED)     // FAS_K_END)

  ! Section Options (Default)
  logical, private :: FLOGS_SECTION_DATETIME = .true.
  logical, private :: FLOGS_SECTION_LABEL = .true.

  ! Category Definition
  type :: FloggerCategory
    character(7) :: name
    character(15) :: style
  end type FloggerCategory

  type(FloggerCategory), private :: flogDebug, flogInfo, flogNotice
  type(FloggerCategory), private :: flogWarning, flogError, flogFatal
  parameter(flogDebug   = FloggerCategory('debug  ', STY_DEBUG))
  parameter(flogInfo    = FloggerCategory('info   ', STY_INFO))
  parameter(flogNotice  = FloggerCategory('notice ', STY_NOTE))
  parameter(flogWarning = FloggerCategory('warning', STY_WARN))
  parameter(flogError   = FloggerCategory('ERROR!!', STY_ERROR))
  parameter(flogFatal   = FloggerCategory('FATAL!!', STY_FATAL))

  ! define procedures visibility
  public :: flogger_set_style, flogger_set_section, print_file_header
  public :: printFormatted
contains

!--- PRIVATE FUNCTIONS / SUBROUTINES

  function getDateTime(useEncoding) result(out)
    implicit none
    character(:), allocatable :: out
    logical, optional :: useEncoding

    !--- local variables
    integer :: date_time(8)
    logical :: useEncodingLocal = .true.
    character(100) :: tmp

    !--- processes
    call date_and_time(values=date_time)

    if ( present(useEncoding) ) useEncodingLocal = useEncoding
    if ( useEncodingLocal ) then
      write(tmp, 200) trim(FLOGS_STYLE_DATETIME),                             &
        date_time(1), date_time(2), date_time(3),               &
        date_time(5), date_time(6), date_time(7), date_time(8), &
        FAS_K_CLEAR
    else
      write(tmp, 210) date_time(1), date_time(2), date_time(3),               &
        date_time(5), date_time(6), date_time(7), date_time(8)
    end if

    out = trim(tmp)

    !--- formatters
200 format (A, '[', I4, '-', I2.2, '-', I2.2, ' ',                          &
      I2.2, ':', I2.2, ':', I2.2, '.', I3.3, ']', A)
210 format ('[', I4, '-', I2.2, '-', I2.2, ' ',                             &
      I2.2, ':', I2.2, ':', I2.2, '.', I3.3, ']')
  end function


  function getLabel(label, useEncoding) result(out)
    implicit none
    character(:), allocatable :: out
    character(*), intent(in) :: label
    logical, optional :: useEncoding

    !--- local variables
    character(100) :: tmp
    logical :: useEncodingLocal = .true.

    !--- processes
    if ( present(useEncoding) ) useEncodingLocal = useEncoding
    if ( useEncodingLocal ) then
      write(tmp, 200) trim(FLOGS_STYLE_LABEL), trim(label), FAS_K_CLEAR
    else
      write(tmp, 210) trim(label)
    end if

    out = trim(tmp)

    !--- formatters
200 format (A, '[', A, ']', A)
210 format ('[', A, ']')
  end function


  function getLevel(level, useEncoding) result(out)
    implicit none
    character(:), allocatable :: out
    integer, intent(in) :: level
    logical, optional :: useEncoding

    !--- local variables
    type(FloggerCategory) :: lv(6) = [                                          &
      flogDebug, flogInfo, flogNotice, flogWarning, flogError, flogFatal      &
      ]
    character(100) :: tmp
    logical :: useEncodingLocal = .true.

    !--- processes
    if ( present(useEncoding) ) useEncodingLocal = useEncoding
    if ( useEncodingLocal ) then
      write(tmp, 200) trim(lv(level)%style), trim(lv(level)%name), FAS_K_CLEAR
    else
      write(tmp, 210) trim(lv(level)%name)
    end if

    out = trim(tmp)

    !--- formatters
200 format ('[', A, A, A, ']')
210 format ('[', A, ']')
  end function

!--- PUBLIC FUNCTIONS / SUBROUTINES

  function printFormatted(message, label, level, inConsole) result(out)
    implicit none
    character(:), allocatable :: out
    character(*), intent(in) :: message
    character(*), intent(in) :: label
    integer, intent(in) :: level
    logical, intent(in) :: inConsole

    !--- local variables
    out = ''

    !--- processes
    if ( FLOGS_SECTION_DATETIME )                                               &
      out = out // trim(getDateTime(useEncoding=inConsole)) // ' '
    if ( FLOGS_SECTION_LABEL )                                                  &
      out = out // trim(getLabel(label, useEncoding=inConsole)) // ' '

    out = out // getLevel(level, useEncoding=inConsole) // ' ' // message
  end function printFormatted


  subroutine flogger_set_style(LabelOptions,  DateOptions, TextOptions)
    implicit none
    character(*), optional, intent(in) :: LabelOptions(:)
    character(*), optional, intent(in) :: DateOptions(:)
    character(*), optional, intent(in) :: TextOptions(:)

    if ( present(LabelOptions) ) &
      FLOGS_STYLE_LABEL = getStyleEncoding(LabelOptions)

    if ( present(DateOptions) ) &
      FLOGS_STYLE_DATETIME = getStyleEncoding(DateOptions)

    if ( present(TextOptions) ) &
      FLOGS_STYLE_TEXT = getStyleEncoding(TextOptions)
  end subroutine flogger_set_style


  subroutine flogger_set_section(addDateTime,  addLabel)
    implicit none
    logical, optional, intent(in) :: addDateTime, addLabel

    if ( present(addDateTime) ) FLOGS_SECTION_DATETIME = addDateTime
    if ( present(addLabel) ) FLOGS_SECTION_LABEL = addLabel
  end subroutine flogger_set_section


  subroutine print_file_header(unit)
    implicit none
    integer, intent(in) :: unit

    write(unit, 200)
    write(unit, 210) "LOGFILES GENERATED BY FLOGGER"
    write(unit, 220) getDateTime(UseEncoding=.false.)
    write(unit, 200)
    write(unit, *)

    !--- formatters
210 format (34X, A29, 34X)
220 format (36X, A25, 36X)
200 format ("=================================================",            &
      "================================================")
  end subroutine print_file_header

end module FloggerFormatter
