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
! furnished to do so, subject to t  he following conditions:
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


module FloggerAnsiStyling
  implicit none

  !--- escape keys
  character(*), parameter :: FAS_K_ESC   = achar(27)
  character(*), parameter :: FAS_K_END   = 'm'
  character(*), parameter :: FAS_K_START = FAS_K_ESC // '['
  character(*), parameter :: FAS_K_CLEAR = FAS_K_START // '0m'

  !--- font style
  character(3), parameter :: FAS_RESET   = '0'
  character(3), parameter :: FAS_BOLD    = '1'
  character(3), parameter :: FAS_ITALIC  = '3'
  character(3), parameter :: FAS_ULINE   = '4'

  !--- text colors
  character(3), parameter :: FGD_BLACK   = '30'
  character(3), parameter :: FGD_RED     = '31'
  character(3), parameter :: FGD_GREEN   = '32'
  character(3), parameter :: FGD_YELLOW  = '33'
  character(3), parameter :: FGD_BLUE    = '34'
  character(3), parameter :: FGD_MAGENTA = '35'
  character(3), parameter :: FGD_CYAN    = '36'
  character(3), parameter :: FGD_WHITE   = '37'
  character(3), parameter :: FGB_BLACK   = '90'
  character(3), parameter :: FGB_RED     = '91'
  character(3), parameter :: FGB_GREEN   = '92'
  character(3), parameter :: FGB_YELLOW  = '93'
  character(3), parameter :: FGB_BLUE    = '94'
  character(3), parameter :: FGB_MAGENTA = '95'
  character(3), parameter :: FGB_CYAN    = '96'
  character(3), parameter :: FGB_WHITE   = '97'

  !--- background colors
  character(3), parameter :: BGD_BLACK   = '40'
  character(3), parameter :: BGD_RED     = '41'
  character(3), parameter :: BGD_GREEN   = '42'
  character(3), parameter :: BGD_YELLOW  = '43'
  character(3), parameter :: BGD_BLUE    = '44'
  character(3), parameter :: BGD_MAGENTA = '45'
  character(3), parameter :: BGD_CYAN    = '46'
  character(3), parameter :: BGD_WHITE   = '47'
  character(3), parameter :: BGB_BLACK   = '100'
  character(3), parameter :: BGB_RED     = '101'
  character(3), parameter :: BGB_GREEN   = '102'
  character(3), parameter :: BGB_YELLOW  = '103'
  character(3), parameter :: BGB_BLUE    = '104'
  character(3), parameter :: BGB_MAGENTA = '105'
  character(3), parameter :: BGB_CYAN    = '106'
  character(3), parameter :: BGB_WHITE   = '107'


contains

  function getStyleEncoding(options) result(out)
    implicit none

    character(len=*), optional, intent(in) :: options(:)
    character(len=:), allocatable :: out
    integer :: option_count, i

    out = FAS_K_START
    option_count = size(options)

    if ( (.not. present(options)) .or. (option_count == 0) ) then
      out = FAS_K_CLEAR
      return
    end if

    do i = 1, option_count
      if ( i == 1 ) then
        out = out // trim(options(i))
      else
        out = out // ';' // trim(options(i))
      end if
    end do

    out = out // FAS_K_END
    out = trim(out)
  end function getStyleEncoding

end module FloggerAnsiStyling
