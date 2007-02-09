! -*- mode: f90; -*-
! Copyright (C) 2005-2007 Nicole Riemer and Matthew West
! Licensed under the GNU General Public License version 2 or (at your
! option) any later version. See the file COPYING for details.

module mod_environ
  
  type environ
     real*8 :: T    ! temperature (K)
     real*8 :: RH   ! relative humidity (1)
     real*8 :: V_comp ! computational volume (m^3)
     real*8 :: p     ! ambient pressure (Pa)
     real*8 :: rho_a ! air density (kg m^{-3})
     integer :: n_temps ! number of temperature set-points
     real*8, dimension(:), pointer :: temp_times ! times at temp set-points
     real*8, dimension(:), pointer :: temps      ! temps at temp set-points
  end type environ
  
contains
  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine allocate_environ_temps(env, n_temps)

    type(environ), intent(inout) :: env   ! environment
    integer, intent(in) :: n_temps        ! number of temperature set-points

    env%n_temps = n_temps
    allocate(env%temp_times(n_temps))
    allocate(env%temps(n_temps))

  end subroutine allocate_environ_temps

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  subroutine change_water_volume(env, mat, dv)
    
    ! Adds the given water volume to the water vapor and updates all
    ! environment quantities.
    
    use mod_constants
    use mod_material
    
    type(environ), intent(inout) :: env ! environment state to update
    type(material), intent(in)   :: mat ! material constants
    real*8, intent(in) :: dv            ! volume of water added (m^3)
    
    real*8 pmv     ! ambient water vapor pressure (Pa)
    real*8 mv      ! ambient water vapor density (kg m^{-3})
                   ! pmv and mv are related by the factor M_w/(R*T)
    real*8 dmv     ! change of water density (kg m^{-3})
    
    dmv = dv * mat%rho(mat%i_water) / env%V_comp
    pmv = sat_vapor_pressure(env) * env%RH
    mv = mat%M_w(mat%i_water)/(const%R*env%T) * pmv
    mv = mv - dmv    
    env%RH = const%R * env%T / mat%M_w(mat%i_water) * mv &
         / sat_vapor_pressure(env)
    
  end subroutine change_water_volume
  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  subroutine update_environ(env, time)
    
    ! Update time-dependent contents of the environment (currently
    ! just temperature).

    use mod_util

    type(environ), intent(inout) :: env ! environment state to update
    real*8, intent(in) :: time          ! current time (s)
    
    real*8 pmv      ! ambient water vapor pressure (Pa)

    env%T = interp_1d(env%n_temps, env%temp_times, env%temps, time)
    pmv = sat_vapor_pressure(env) * env%RH
    env%RH = pmv / sat_vapor_pressure(env)

    
  end subroutine update_environ
  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  real*8 function sat_vapor_pressure(env) ! Pa
    
    use mod_constants
    
    type(environ), intent(in) :: env ! environment state
    
    sat_vapor_pressure = const%p00 * 10d0**(7.45d0 * (env%T - const%T0) &
         / (env%T - 38d0)) ! Pa
    
  end function sat_vapor_pressure
  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
end module mod_environ
