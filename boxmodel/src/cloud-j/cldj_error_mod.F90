! INTERFACE:

      MODULE CLDJ_ERROR_MOD

      IMPLICIT NONE

      PUBLIC  :: CLOUDJ_ERROR_STOP
      PUBLIC  :: SAFE_DIV

      CONTAINS

!-----------------------------------------------------------------------
      subroutine CLOUDJ_ERROR_STOP( msg, loc )
      ! This subroutine...

      ! This subroutine is based on the equivalent function in GEOS-Chem
      ! (https://github.com/geoschem/geos-chem).
!-----------------------------------------------------------------------
#if defined( ESMF_ )
      ! if using GCHP or GEOS
      USE MAPL_Mod
#     include "MAPL_Generic.h"
#elif defined( MODEL_CESM )
      ! if using cesm
      USE CAM_ABORTUTILS,     ONLY : ENDRUN
#endif

      character(len=*), intent(in)           :: msg
      character(len=*), intent(in), optional :: loc

      character(len=512) :: tmpmsg

      if ( present(loc) ) then
         tmpmsg = 'Cloud-J error at '//trim(loc)//': '//trim(msg)
      else
         tmpmsg = 'Cloud-J error: '//trim(msg)
      endif
      write(6, '(a)' ) tmpmsg
      call flush(6)

#if defined( ESMF_ )
      __Iam__('CLOUDJ_ERROR_STOP')
#elif defined( MODEL_CESM )
      call endrun('Cloud-J failure!')
#elif defined( MODEL_GCCLASSIC )
      call exit( 99999 )
#else
      stop
#endif

      end subroutine CLOUDJ_ERROR_STOP

!-----------------------------------------------------------------------
      function SAFE_DIV( numer, denom, alt_nan, alt_overflow, alt_underflow ) &
       result( quot )
      ! This funtion performs "safe division", that is to prevent overflow,
      ! underlow, NaN, or infinity errors.  An alternate value is returned
      ! if the division cannot be performed.

      ! This function is based on the equivalent function in GEOS-Chem
      ! (https://github.com/geoschem/geos-chem). 
!-----------------------------------------------------------------------

      ! Numerator and denominator
      real*8, intent(in) :: numer
      real*8, intent(in) :: denom

      ! Alternate value to be returned if the division is either NAN (0/0) or
      ! leads to overflow (i.e., a too large number)
      real*8, intent(in) :: alt_nan

      ! Alternate value to be returned if the division leads to overflow
      ! (default is ALT_NAN)
      real*8, optional, intent(in) :: alt_overflow
 
      ! Alternate value to be returned if the division leads to underflow
      ! (default is 0, but you could use TINY() if you want a non-zero result).
      real*8, optional, intent(in) :: alt_underflow

      ! Return value is output from division
      real*8 :: quot

      ! NaN
      if ( numer == 0 .and. denom == 0 ) THEN
         quot = alt_nan

      ! Overflow
      else if ( EXPONENT(numer) - EXPONENT(denom) >= MAXEXPONENT(numer) &
                .OR. Denom == 0 ) then
         quot = alt_nan
         if ( PRESENT(alt_overflow) ) quot = alt_overflow

      ! Underflow
      else if ( EXPONENT(numer) - EXPONENT(denom) <= MINEXPONENT(numer) ) then

         quot = 0D0
         if ( PRESENT(alt_underflow) ) quot = alt_underflow
  
      else
  
         ! No problem
         quot = numer / denom
  
      endif

      end function SAFE_DIV

      END MODULE CLDJ_ERROR_MOD
