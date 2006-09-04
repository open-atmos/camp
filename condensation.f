! -*- mode: f90;-*-
! Condensation
!

module mod_condensation
contains
  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine condense_particles(n_bin, TDV, n_spec, MH, VH, &
       del_t, bin_v, bin_r, bin_g, bin_gs, bin_n, dlnr, env, mat)

    use mod_array
    use mod_array_hybrid
    use mod_bin
    use mod_environ
    use mod_material

    integer, intent(in) :: n_bin ! number of bins
    integer, intent(in) :: TDV   ! second dimension of VH
    integer, intent(in) :: n_spec       ! number of species
    integer, intent(inout) :: MH(n_bin) ! number of particles per bin
    real*8, intent(inout) :: VH(n_bin,TDV,n_spec) ! particle volumes (m^3)
    real*8, intent(in) :: del_t         ! total time to integrate
    real*8, intent(in) :: bin_v(n_bin) ! volume of particles in bins (m^3)
    real*8, intent(in) ::  bin_r(n_bin) ! radius of particles in bins (m)
    real*8, intent(inout) :: bin_g(n_bin) ! mass in bins  
    real*8, intent(inout) :: bin_gs(n_bin,n_spec) ! species mass in bins
    integer, intent(inout) :: bin_n(n_bin)      ! number in bins
    real*8, intent(in) :: dlnr                  ! bin scale factor
    type(environ), intent(inout) :: env  ! environment state
    type(material), intent(in) :: mat    ! material properties
    
    ! local variables
    integer bin, j, new_bin, k
    real*8 pv

    do bin = 1,n_bin
       do j = 1,MH(bin)
          call condense_particle(n_spec, VH(bin,j,:), del_t, env, mat)
       end do
    end do

    ! We resort the particles in the bins after all particles are
    ! advanced, otherwise we will lose track of which ones have been
    ! advanced and which have not.
    call resort_array_hybrid(n_bin, TDV, n_spec, MH, VH, bin_v, &
         bin_r, dlnr)

    ! update the bin arrays
    call moments_hybrid(n_bin, TDV, n_spec, MH, VH, bin_v, &
         bin_r, bin_g, bin_gs, bin_n, dlnr)

  end subroutine condense_particles

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine condense_particle(n_spec, V, del_t, env, mat)

    ! integrates the condensation growth or decay ODE for total time
    ! del_t

    use mod_util
    use mod_environ
    use mod_material

    integer, intent(in) :: n_spec ! number of species
    real*8, intent(inout) :: V(n_spec) ! particle volumes (m^3)
    real*8, intent(in) :: del_t ! total time to integrate
    type(environ), intent(in) :: env     ! environment state
    type(material), intent(in) :: mat    ! material properties

    real*8 time_step, time
    logical done

    integer i
    real*8 dvdt

    time = 0d0
    done = .false.
    do while (.not. done)
       call condense_step_euler(n_spec, V, del_t - time, &
            time_step, done, env, mat)
       time = time + time_step
    end do
    
  end subroutine condense_particle

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine condense_step_euler(n_spec, V, max_dt, dt, &
       done, env, mat)

    ! Does one timestep (determined by this subroutine) of the
    ! condensation ODE. The timestep will not exceed max_dt, but might
    ! be less. If we in fact step all the way to max_dt then done will
    ! be true.
    
    use mod_environ
    use mod_material

    integer, intent(in) :: n_spec ! number of species
    real*8, intent(inout) :: V(n_spec) ! particle volumes (m^3)
    real*8, intent(in) :: max_dt ! maximum timestep to integrate
    real*8, intent(out) :: dt ! actual timestep used
    logical, intent(out) :: done ! whether we reached the maximum timestep
    type(environ), intent(in) :: env     ! environment state
    type(material), intent(in) :: mat    ! material properties

    real*8 dvdt

    done = .false.
    call find_condense_timestep_variable(n_spec, V, dt, env, mat)
    if (dt .ge. max_dt) then
       dt = max_dt
       done = .true.
    end if
!    write(*,*) 'dt = ', dt
!    stop

    call cond_newt(n_spec, V, dvdt, env, mat)
    V(mat%i_water) = V(mat%i_water) + dt * dvdt
    V(mat%i_water) = max(0d0, V(mat%i_water))
   
  end subroutine condense_step_euler

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine condense_step_rk_fixed(n_spec, V, max_dt, &
       dt, done, env, mat)

    ! Does one timestep (determined by this subroutine) of the
    ! condensation ODE. The timestep will not exceed max_dt, but might
    ! be less. If we in fact step all the way to max_dt then done will
    ! be true.
    
    use mod_environ
    use mod_material

    integer, intent(in) :: n_spec ! number of species
    real*8, intent(inout) :: V(n_spec) ! particle volumes (m^3)
    real*8, intent(in) :: max_dt ! maximum timestep to integrate
    real*8, intent(out) :: dt ! actual timestep used
    logical, intent(out) :: done ! whether we reached the maximum timestep
    type(environ), intent(in) :: env     ! environment state
    type(material), intent(in) :: mat    ! material properties

    done = .false.
    call find_condense_timestep_variable(n_spec, V, dt, env, mat)
    if (dt .ge. max_dt) then
       dt = max_dt
       done = .true.
    end if

    call condense_step_rk(n_spec, V, dt, env, mat)
   
  end subroutine condense_step_rk_fixed

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine condense_step_rk(n_spec, V, dt, env, mat)

    ! Does one fixed timestep of RK4.

    use mod_environ
    use mod_material

    integer, intent(in) :: n_spec ! number of species
    real*8, intent(inout) :: V(n_spec) ! particle volumes (m^3)
    real*8, intent(out) :: dt ! timestep
    type(environ), intent(in) :: env     ! environment state
    type(material), intent(in) :: mat    ! material properties

    ! local variables
    real*8 k1, k2, k3, k4
    real*8 V_tmp(n_spec)

    V_tmp = V

    ! step 1
    call cond_newt(n_spec, V, k1, env, mat)

    ! step 2
    V_tmp(mat%i_water) = V(mat%i_water) + dt * k1 / 2d0
    V_tmp(mat%i_water) = max(0d0, V_tmp(mat%i_water))
    call cond_newt(n_spec, V_tmp, k2, env, mat)

    ! step 3
    V_tmp(mat%i_water) = V(mat%i_water) + dt * k2 / 2d0
    V_tmp(mat%i_water) = max(0d0, V_tmp(mat%i_water))
    call cond_newt(n_spec, V_tmp, k3, env, mat)

    ! step 4
    V_tmp(mat%i_water) = V(mat%i_water) + dt * k3
    V_tmp(mat%i_water) = max(0d0, V_tmp(mat%i_water))
    call cond_newt(n_spec, V_tmp, k4, env, mat)

    V(mat%i_water) = V(mat%i_water) &
         + dt * (k1 / 6d0 + k2 / 3d0 + k3 / 3d0 + k4 / 6d0)

    V(mat%i_water) = max(0d0, V(mat%i_water))
   
  end subroutine condense_step_rk

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine find_condense_timestep_constant(n_spec, V, dt, env, mat)

    ! constant timestep

    use mod_array
    use mod_environ
    use mod_material

    integer, intent(in) :: n_spec ! number of species
    real*8, intent(in) :: V(n_spec) ! particle volumes (m^3)
    real*8, intent(out) :: dt ! timestep to use
    type(environ), intent(in) :: env     ! environment state
    type(material), intent(in) :: mat    ! material properties

    dt = 5d-3

  end subroutine find_condense_timestep_constant

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine find_condense_timestep_variable(n_spec, V, dt, env, mat)

    ! timestep is proportional to V / (dV/dt)

    use mod_array
    use mod_environ
    use mod_material

    integer, intent(in) :: n_spec ! number of species
    real*8, intent(in) :: V(n_spec) ! particle volumes (m^3)
    real*8, intent(out) :: dt ! timestep to use
    type(environ), intent(in) :: env     ! environment state
    type(material), intent(in) :: mat    ! material properties

    ! parameters
    real*8 scale
    parameter (scale = 0.1d0) ! scale factor for timestep

    real*8 pv, dvdt

    call particle_vol_base(n_spec, V, pv)
    call cond_newt(n_spec, V, dvdt, env, mat)
!    write(*,*) 'pv = ', pv, ' dvdt = ', dvdt
    dt = abs(scale * pv / dvdt)

  end subroutine find_condense_timestep_variable
  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      
  subroutine cond_newt(n_spec, V, dvdt, env, mat)
    
    ! Newton's method to solve the error equation, determining the
    ! growth rate dm/dt. The extra variable T_a is the local
    ! temperature, which is also implicitly determined, but not
    ! returned at this point.

    use mod_util
    use mod_array
    use mod_environ
    use mod_material

    integer, intent(in) :: n_spec ! number of species
    real*8, intent(in) :: V(n_spec) ! particle volumes (m^3)
    real*8, intent(out) :: dvdt  ! dv/dt (m^3 s^{-1})
    type(environ), intent(in) :: env     ! environment state
    type(material), intent(in) :: mat    ! material properties

    ! parameters
    integer iter_max
    real*8 dmdt_min, dmdt_max, dmdt_tol, f_tol

    parameter (dmdt_min = -1d0)      ! minimum value of dm/dt (kg s^{-1})
    parameter (dmdt_max = 1d0)     ! maximum value of dm/dt (kg s^{-1})
    parameter (dmdt_tol = 1d-15) ! dm/dt tolerance for convergence
    parameter (f_tol = 1d-15) ! function tolerance for convergence
    parameter (iter_max = 400)   ! maximum number of iterations

    ! local variables
    integer iter, k
    real*8 g_water, g_solute, pv
    real*8 dmdt, T_a, delta_f, delta_dmdt, f, old_f, df, d

    g_water = V(mat%i_water) * mat%rho(mat%i_water)
    g_solute = 0d0
    do k = 1,n_spec
       if (k .ne. mat%i_water) then
          g_solute = g_solute + V(k) * mat%rho(k)
       end if
    end do

    call particle_vol_base(n_spec, V, pv)
    d = vol2diam(pv)

    dmdt = (dmdt_min + dmdt_max) / 2d0
    call cond_func(n_spec, V, dmdt, d, f, df, T_a, env, mat)
    old_f = f

    iter = 0
    do
       iter = iter + 1

       delta_dmdt = f / df
       dmdt = dmdt - delta_dmdt
       call cond_func(n_spec, V, dmdt, d, f, df, T_a, env, mat)
       delta_f = f - old_f
       old_f = f
       
       if ((dmdt .lt. dmdt_min) .or. (dmdt .gt. dmdt_max)) then
          write(0,*) 'ERROR: Newton iteration exceeded bounds'
          write(0,'(a15,a15,a15,a15)') 'pv', 'dmdt', 'lower bound', 'upper bound'
          write(0,'(g15.4,g15.4,g15.4,g15.4)') pv, dmdt, dmdt_min, dmdt_max
          call exit(1)
       endif

       if (iter .ge. iter_max) then
          write(0,*) 'ERROR: Newton iteration had too many iterations'
          write(0,'(a15,a15,a15)') 'pv', 'dmdt', 'iter_max'
          write(0,'(g15.4,g15.4,i15)') pv, dmdt, iter_max
          call exit(2)
       end if
       
       if ((abs(delta_dmdt) .lt. dmdt_tol) &
            .and. (abs(delta_f) .lt. f_tol)) exit
    enddo

    dvdt = dmdt / mat%rho(mat%i_water)

  end subroutine cond_newt

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine cond_func(n_spec, V, x, d_p, f, df, T_a, env, mat)

    ! Return the error function value and its derivative.

    use mod_environ
    use mod_material
    use mod_constants

    integer, intent(in) :: n_spec ! number of species
    real*8, intent(in) :: V(n_spec) ! particle volumes (m^3)
    real*8, intent(in) :: x   ! mass growth rate dm/dt (kg s^{-1})
    real*8, intent(in) :: d_p ! diameter (m)
    real*8, intent(out) :: f  ! error
    real*8, intent(out) :: df ! derivative of error with respect to x
    real*8, intent(out) :: T_a ! droplet temperature (K)
    type(environ), intent(in) :: env     ! environment state
    type(material), intent(in) :: mat    ! material properties
    
    ! local variables
    real*8 k_a, k_ap, k_ap_div, D_v, D_vp
    real*8 rat, fact1, fact2, c1, c2, c3, c4, c5
    real*8 M_water, M_solute, rho_water, rho_solute
    real*8 eps, nu, g_water, g_solute

    M_water = average_water_quantity(V, mat, mat%M_w)
    M_solute = average_solute_quantity(V, mat, mat%M_w)
    nu = average_solute_quantity(V, mat, dble(mat%nu))
    eps = average_solute_quantity(V, mat, mat%eps)
    rho_water = average_water_quantity(V, mat, mat%rho)
    rho_solute = average_solute_quantity(V, mat, mat%rho)
    g_water = total_water_quantity(V, mat, mat%rho)
    g_solute = total_solute_quantity(V, mat, mat%rho)

!    write(*,*) 'x = ', x

    ! molecular diffusion coefficient uncorrected
    D_v = 0.211d-4 / (env%p / const%atm) * (env%T / 273d0)**1.94d0 ! m^2 s^{-1}

!    write(*,*) 'D_v = ', D_v

    ! molecular diffusion coefficient corrected for non-continuum effects
    ! D_v_div = 1d0 + (2d0 * D_v * 1d-4 / (const%alpha * d_p)) &
    !      * (2 * const%pi * M_water / (const%R * env%T))**0.5d0
    ! D_vp = D_v / D_v_div

    ! TEST: use the basic expression for D_vp
    D_vp = D_v                ! m^2 s^{-1}
    ! FIXME: check whether we can reinstate the correction

    ! thermal conductivity uncorrected
    k_a = 1d-3 * (4.39d0 + 0.071d0 * env%T) ! J m^{-1} s^{-1} K^{-1}
    k_ap_div = 1d0 + 2d0 &
         * k_a / (const%alpha * d_p * const%rho_a * const%cp) &
         * (2d0 * const%pi * const%M_a / (const%R * env%T))**0.5d0 ! dim-less
    ! thermal conductivity corrected
    k_ap = k_a / k_ap_div     ! J m^{-1} s^{-1} K^{-1}
      
!    write(*,*) 'k_ap = ', k_ap

!    write(*,*) 'M_water = ', M_water

    rat = sat_vapor_pressure(env) / (const%R * env%T)
    fact1 = const%L_v * M_water / (const%R * env%T)
    fact2 = const%L_v / (2d0 * const%pi * d_p * k_ap * env%T)
    
!    write(*,*) 'rat = ', rat
!    write(*,*) 'fact1 = ', fact1
!    write(*,*) 'fact2 = ', fact2

!    write(*,*) 'nu = ', nu
!    write(*,*) 'eps = ', eps
!    write(*,*) 'M_water = ', M_water
!    write(*,*) 'M_solute = ', M_solute
!    write(*,*) 'g_solute = ', g_solute
!    write(*,*) 'g_water = ', g_water

    c1 = 2d0 * const%pi * d_p * D_vp * M_water * rat
    c2 = 4d0 * M_water &
         * const%sig / (const%R * rho_water * d_p)
    c3 = c1 * fact1 * fact2
    c4 = const%L_v / (2d0 * const%pi * d_p * k_ap)
    ! incorrect expression from Majeed and Wexler:
!     c5 = nu * eps * M_water * rho_solute * r_n**3d0 &
!         / (M_solute * rho_water * ((d_p / 2)**3d0 - r_n**3))
    c5 = nu * eps * M_water / M_solute * g_solute / g_water
    ! corrected according to Jim's note:
!    c5 = nu * eps * M_water / M_solute * g_solute / &
!         (g_water + (rho_water / rho_solute) * eps * g_solute)
    
!    write(*,*) 'c1 = ', c1
!    write(*,*) 'c2 = ', c2
!    write(*,*) 'c3 = ', c3
!    write(*,*) 'c4 = ', c4
!    write(*,*) 'c5 = ', c5

    T_a = env%T + c4 * x ! K
    
!    write(*,*) 'T_a = ', T_a
    
    f = x - c1 * (env%RH - exp(c2 / T_a - c5)) &
         / (1d0 + c3 * exp(c2 / T_a - c5))
    
    df = 1d0 + c1 * env%RH * (1d0 + c3 * exp(c2 / T_a -c5))**(-2d0) * c3 * &
         exp(c2 / T_a - c5) * (-1d0) * c2 * c4 / T_a**2d0 + c1 * &
         (exp(c2 / T_a - c5) * (-1d0) * c2 * c4 / T_a**2d0 * (1d0 + c3 &
         * exp(c2 / T_a -c5))**(-1d0) + exp(c2 / T_a - c5) * (-1d0) * &
         (1d0 + c3 * exp(c2 / T_a -c5))**(-2d0) * c3 * exp(c2 / T_a - &
         c5) * (-1d0) * c2 * c4 / T_a**2d0)

!    write(*,*) 'f = ', f, ' df = ', df
    
  end subroutine cond_func
  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine equilibriate_particle(n_spec, V, rho, i_water, &
       nu, eps, M_s, RH, env, mat)

    ! add water to the particle until it is in equilibrium

    use mod_util
    use mod_array
    use mod_environ
    use mod_material

    integer, intent(in) :: n_spec ! number of species
    real*8, intent(inout) :: V(n_spec) ! particle volumes (m^3)
    real*8, intent(in) :: rho(n_spec) ! density of species (kg m^{-3})  
    integer, intent(in) :: i_water ! water species number
    integer, intent(in) :: nu(n_spec)      ! number of ions in the solute
    real*8, intent(in) :: eps(n_spec)      ! solubility of aerosol material (1)
    real*8, intent(in) :: M_s(n_spec)      ! molecular weight of solute (kg mole^{-1})
    real*8, intent(in) :: RH               ! relative humidity for equilibrium (1)
    type(environ), intent(in) :: env     ! environment state
    type(material), intent(in) :: mat    ! material properties

    ! FIXME: Preliminarily set i_spec = 1. 

    ! paramters
    real*8 x_min, x_max, x_tol, x1, x2, xacc
    real*8 c0, c1, c3, c4, dc0, dc2, dc3
    real*8 rw, x 
    real*8 A, B, T, T0

    real*8 pv
    real*8 sig_w, RR 
    parameter (sig_w = 0.073d0) ! surface energy (J m^{-2})
    parameter (RR = 8.314d0)   ! universal gas constant (J mole^{-1} K^{-1})
    parameter (T0 = 298d0)     ! temperature of gas medium (K)

    integer i_spec

!    write(6,*)'in equilibriate_particle ', V(1), V(2), V(3)
    i_spec = 1
 
    call particle_vol_base(n_spec, V, pv)

    T = T0
    A = 4d0 * M_s(i_water) * sig_w / (RR * T * rho(i_water))
    
    B = dble(nu(i_spec)) * eps(i_spec) * M_s(i_water) * rho(i_spec) &
             * vol2rad(pv)**3.d0 / (M_s(i_spec) * rho(i_water))
    
    c4 = log(RH) / 8d0
    c3 = A / 8d0
    
    dc3 = log(RH) / 2d0
    dc2 = 3d0 * A / 8d0
    
    x1 = 0d0
    x2 = 10d0
    xacc = 1d-15
    
    c1 = B - log(RH) * vol2rad(pv)**3d0
    c0 = A * vol2rad(pv)**3d0
    dc0 = c1
    
    call equilibriate_newt(x1, x2, xacc, x, c4, c3, c1, c0, dc3, dc2, dc0)
    
    rw = x / 2d0

    V(i_water) = rad2vol(rw) - pv

!    write(6,*)'out equilibriate_particle ', V(1), V(2), V(3)

  end subroutine equilibriate_particle

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine equilibriate_newt(x1, x2, xacc, x, c4, c3, c1, c0, dc3, &
       dc2, dc0)

    real*8, intent(out) :: x
    real*8, intent(in) :: x1
    real*8, intent(in) :: x2
    real*8, intent(in) :: xacc
    real*8, intent(in) :: c4
    real*8, intent(in) :: c3
    real*8, intent(in) :: c1
    real*8, intent(in) :: c0
    real*8, intent(in) :: dc3
    real*8, intent(in) :: dc2
    real*8, intent(in) :: dc0

    integer jmax
    parameter (jmax=400)
    
    integer j
    real*8 df, dx, f, d 

    x = 0.5d0 * (x1 + x2)
    
    do j = 1,jmax
       call equilibriate_func(x,f,df,d,c4,c3,c1,c0,dc3,dc2,dc0)
       dx = f / df
       x = x - dx
       if((x .lt. x1) .or. (x .gt. x2)) then
          write(6,*)'x1,x2,x ',x1,x2,x
          write(*,*) 'rtnewt jumped out of brackets'
          call exit(2)
       endif
       if(abs(dx) .lt. xacc) then
          return
       endif
    enddo
    
    write(*,*) 'rtnewt exceeded maximum iteration '
    call exit(2)

  end subroutine equilibriate_newt

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine equilibriate_func(x, f, df, d_p, c4, c3, c1, c0, dc3, dc2, dc0)

    real*8 x, f, df, d_p
    real*8 c4, c3, c1, c0, dc3, dc2, dc0

    f = c4 * x**4d0 - c3 * x**3d0 + c1 * x + c0
    df = dc3 * x**3d0 -dc2 * x**2d0 + dc0

  end subroutine equilibriate_func

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
end module mod_condensation
