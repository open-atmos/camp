! Copyright (C) 2005-2021 Barcelona Supercomputing Center and University of
! Illinois at Urbana-Champaign
! SPDX-License-Identifier: MIT

!> \file
!> The camp_env_state module.

!> The env_state_t structure and associated subroutines.
module camp_env_state

  use camp_constants
  use camp_util
  use camp_mpi
#ifdef CAMP_USE_MPI
  use mpi
#endif

  !> Current environment state.
  !!
  !! All quantities are instantaneous, describing the state at a
  !! particular instant of time. Constant data and other data not
  !! associated with the current environment state is stored in
  !! scenario_t.
  type env_state_t
     !> Temperature (K).
     real(kind=dp) :: temp
     !> Relative humidity (1).
     real(kind=dp) :: rel_humid
     !> Ambient pressure (Pa).
     real(kind=dp) :: pressure
     !> Longitude (degrees).
     real(kind=dp) :: longitude
     !> Latitude (degrees).
     real(kind=dp) :: latitude
     !> Altitude (m).
     real(kind=dp) :: altitude
     !> Start time (s since 00:00 UTC on \c start_day).
     real(kind=dp) :: start_time
     !> Start day of year (UTC).
     integer :: start_day
     !> Time since \c start_time (s).
     real(kind=dp) :: elapsed_time
     !> Solar zenith angle (radians from zenith).
     real(kind=dp) :: solar_zenith_angle
     !> Box height (m).
     real(kind=dp) :: height
  contains
     !> Determine the number of bytes required to pack the given value
     procedure, pass(val) :: pack_size => camp_mpi_pack_size_env_state
     !> Pack the given value to a buffer, advancing position
     procedure, pass(val) :: bin_pack => camp_mpi_pack_env_state
     !> Unpack the given value from a buffer, advancing position
     procedure, pass(val) :: bin_unpack => camp_mpi_unpack_env_state
  end type env_state_t

  !> Pointer for env_state_t
  type env_state_ptr
    type(env_state_t), pointer :: val => null()
  contains
    !> Set the temperature (K)
    procedure :: set_temperature_K
    !> Set the pressure (Pa)
    procedure :: set_pressure_Pa
  end type env_state_ptr

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> env_state += env_state_delta
  subroutine env_state_add(env_state, env_state_delta)

    !> Environment.
    type(env_state_t), intent(inout) :: env_state
    !> Increment.
    type(env_state_t), intent(in) :: env_state_delta

    env_state%temp = env_state%temp + env_state_delta%temp
    env_state%rel_humid = env_state%rel_humid + env_state_delta%rel_humid
    env_state%pressure = env_state%pressure + env_state_delta%pressure
    env_state%longitude = env_state%longitude + env_state_delta%longitude
    env_state%latitude = env_state%latitude + env_state_delta%latitude
    env_state%altitude = env_state%altitude + env_state_delta%altitude
    env_state%start_time = env_state%start_time + env_state_delta%start_time
    env_state%start_day = env_state%start_day + env_state_delta%start_day
    env_state%elapsed_time = env_state%elapsed_time &
         + env_state_delta%elapsed_time
    env_state%solar_zenith_angle = env_state%solar_zenith_angle &
         + env_state_delta%solar_zenith_angle
    env_state%height = env_state%height + env_state_delta%height

  end subroutine env_state_add

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> env_state *= alpha
  subroutine env_state_scale(env_state, alpha)

    !> Environment.
    type(env_state_t), intent(inout) :: env_state
    !> Scale factor.
    real(kind=dp), intent(in) :: alpha

    env_state%temp = env_state%temp * alpha
    env_state%rel_humid = env_state%rel_humid * alpha
    env_state%pressure = env_state%pressure * alpha
    env_state%longitude = env_state%longitude * alpha
    env_state%latitude = env_state%latitude * alpha
    env_state%altitude = env_state%altitude * alpha
    env_state%start_time = env_state%start_time * alpha
    env_state%start_day = nint(real(env_state%start_day, kind=dp) * alpha)
    env_state%elapsed_time = env_state%elapsed_time * alpha
    env_state%solar_zenith_angle = env_state%solar_zenith_angle * alpha
    env_state%height = env_state%height * alpha

  end subroutine env_state_scale

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Adds the given water volume to the water vapor and updates all
  !> environment quantities.
  subroutine env_state_change_water_volume(env_state, dv)

    !> Environment state to update.
    type(env_state_t), intent(inout) :: env_state
    !> Volume concentration of water added (m^3/m^3).
    real(kind=dp), intent(in) :: dv

    real(kind=dp) pmv     ! ambient water vapor pressure (Pa)
    real(kind=dp) mv      ! ambient water vapor density (kg m^{-3})
                   ! pmv and mv are related by the factor molec_weight/(R*T)
    real(kind=dp) dmv     ! change of water density (kg m^{-3})

    dmv = dv * const%water_density
    pmv = env_state_sat_vapor_pressure(env_state) * env_state%rel_humid
    mv = const%water_molec_weight / (const%univ_gas_const*env_state%temp) * pmv
    mv = mv - dmv
    if (mv < 0d0) then
       call warn_msg(980320483, "relative humidity tried to go negative")
       mv = 0d0
    end if
    env_state%rel_humid = const%univ_gas_const * env_state%temp &
         / const%water_molec_weight * mv &
         / env_state_sat_vapor_pressure(env_state)

  end subroutine env_state_change_water_volume

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Computes the current saturation vapor pressure (Pa).
  real(kind=dp) function env_state_sat_vapor_pressure(env_state)

    !> Environment state.
    type(env_state_t), intent(in) :: env_state

    env_state_sat_vapor_pressure = const%water_eq_vap_press &
         * 10d0**(7.45d0 * (env_state%temp - const%water_freeze_temp) &
         / (env_state%temp - 38d0))

  end function env_state_sat_vapor_pressure

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Air density (kg m^{-3}).
  real(kind=dp) function env_state_air_den(env_state)

    !> Environment state.
    type(env_state_t), intent(in) :: env_state

    env_state_air_den = const%air_molec_weight &
         * env_state_air_molar_den(env_state)

  end function env_state_air_den

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Air molar density (mol m^{-3}).
  real(kind=dp) function env_state_air_molar_den(env_state)

    !> Environment state.
    type(env_state_t), intent(in) :: env_state

    env_state_air_molar_den = env_state%pressure &
         / (const%univ_gas_const * env_state%temp)

  end function env_state_air_molar_den

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Condensation \f$A\f$ parameter.
  real(kind=dp) function env_state_A(env_state)

    !> Environment state.
    type(env_state_t), intent(in) :: env_state

    env_state_A = 4d0 * const%water_surf_eng * const%water_molec_weight &
         / (const%univ_gas_const * env_state%temp * const%water_density)

  end function env_state_A

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Convert (ppb) to (molecules m^{-3}).
  real(kind=dp) function env_state_ppb_to_conc(env_state, ppb)

    !> Environment state.
    type(env_state_t), intent(in) :: env_state
    !> Mixing ratio (ppb).
    real(kind=dp), intent(in) :: ppb

    env_state_ppb_to_conc = ppb / 1d9 * env_state_air_molar_den(env_state) &
         * const%avagadro

  end function env_state_ppb_to_conc

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Convert (molecules m^{-3}) to (ppb).
  real(kind=dp) function env_state_conc_to_ppb(env_state, conc)

    !> Environment state.
    type(env_state_t), intent(in) :: env_state
    !> Concentration (molecules m^{-3}).
    real(kind=dp), intent(in) :: conc

    env_state_conc_to_ppb = conc * 1d9 / env_state_air_molar_den(env_state) &
         / const%avagadro

  end function env_state_conc_to_ppb

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Average val over all processes.
  subroutine env_state_mix(val)

    !> Value to average.
    type(env_state_t), intent(inout) :: val

#ifdef CAMP_USE_MPI
    type(env_state_t) :: val_avg

    call camp_mpi_allreduce_average_real(val%temp, val_avg%temp)
    call camp_mpi_allreduce_average_real(val%rel_humid, val_avg%rel_humid)
    call camp_mpi_allreduce_average_real(val%pressure, val_avg%pressure)
    val%temp = val_avg%temp
    val%rel_humid = val_avg%rel_humid
    val%pressure = val_avg%pressure
#endif

  end subroutine env_state_mix

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Average val over all processes, with the result only on the root
  !> process.
  subroutine env_state_reduce_avg(val)

    !> Value to average.
    type(env_state_t), intent(inout) :: val

#ifdef CAMP_USE_MPI
    type(env_state_t) :: val_avg

    call camp_mpi_reduce_avg_real(val%temp, val_avg%temp)
    call camp_mpi_reduce_avg_real(val%rel_humid, val_avg%rel_humid)
    call camp_mpi_reduce_avg_real(val%pressure, val_avg%pressure)
    if (camp_mpi_rank() == 0) then
       val%temp = val_avg%temp
       val%rel_humid = val_avg%rel_humid
       val%pressure = val_avg%pressure
    end if
#endif

  end subroutine env_state_reduce_avg

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Determines the number of bytes required to pack the given value.
  integer function camp_mpi_pack_size_env_state(val)

    !> Value to pack.
    class(env_state_t), intent(in) :: val

    camp_mpi_pack_size_env_state = &
         camp_mpi_pack_size_real(val%temp) &
         + camp_mpi_pack_size_real(val%rel_humid) &
         + camp_mpi_pack_size_real(val%pressure) &
         + camp_mpi_pack_size_real(val%longitude) &
         + camp_mpi_pack_size_real(val%latitude) &
         + camp_mpi_pack_size_real(val%altitude) &
         + camp_mpi_pack_size_real(val%start_time) &
         + camp_mpi_pack_size_integer(val%start_day) &
         + camp_mpi_pack_size_real(val%elapsed_time) &
         + camp_mpi_pack_size_real(val%solar_zenith_angle) &
         + camp_mpi_pack_size_real(val%height)

  end function camp_mpi_pack_size_env_state

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Packs the given value into the buffer, advancing position.
  subroutine camp_mpi_pack_env_state(buffer, position, val)

    !> Memory buffer.
    character, intent(inout) :: buffer(:)
    !> Current buffer position.
    integer, intent(inout) :: position
    !> Value to pack.
    class(env_state_t), intent(in) :: val

#ifdef CAMP_USE_MPI
    integer :: prev_position

    prev_position = position
    call camp_mpi_pack_real(buffer, position, val%temp)
    call camp_mpi_pack_real(buffer, position, val%rel_humid)
    call camp_mpi_pack_real(buffer, position, val%pressure)
    call camp_mpi_pack_real(buffer, position, val%longitude)
    call camp_mpi_pack_real(buffer, position, val%latitude)
    call camp_mpi_pack_real(buffer, position, val%altitude)
    call camp_mpi_pack_real(buffer, position, val%start_time)
    call camp_mpi_pack_integer(buffer, position, val%start_day)
    call camp_mpi_pack_real(buffer, position, val%elapsed_time)
    call camp_mpi_pack_real(buffer, position, val%solar_zenith_angle)
    call camp_mpi_pack_real(buffer, position, val%height)
    call assert(464101191, &
         position - prev_position <= camp_mpi_pack_size_env_state(val))
#endif

  end subroutine camp_mpi_pack_env_state

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Unpacks the given value from the buffer, advancing position.
  subroutine camp_mpi_unpack_env_state(buffer, position, val)

    !> Memory buffer.
    character, intent(inout) :: buffer(:)
    !> Current buffer position.
    integer, intent(inout) :: position
    !> Value to pack.
    class(env_state_t), intent(inout) :: val

#ifdef CAMP_USE_MPI
    integer :: prev_position

    prev_position = position
    call camp_mpi_unpack_real(buffer, position, val%temp)
    call camp_mpi_unpack_real(buffer, position, val%rel_humid)
    call camp_mpi_unpack_real(buffer, position, val%pressure)
    call camp_mpi_unpack_real(buffer, position, val%longitude)
    call camp_mpi_unpack_real(buffer, position, val%latitude)
    call camp_mpi_unpack_real(buffer, position, val%altitude)
    call camp_mpi_unpack_real(buffer, position, val%start_time)
    call camp_mpi_unpack_integer(buffer, position, val%start_day)
    call camp_mpi_unpack_real(buffer, position, val%elapsed_time)
    call camp_mpi_unpack_real(buffer, position, val%solar_zenith_angle)
    call camp_mpi_unpack_real(buffer, position, val%height)
    call assert(205696745, &
         position - prev_position <= camp_mpi_pack_size_env_state(val))
#endif

  end subroutine camp_mpi_unpack_env_state

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Computes the average of val across all processes, storing the
  !> result in val_avg on the root process.
  subroutine camp_mpi_reduce_avg_env_state(val, val_avg)

    !> Value to average.
    type(env_state_t), intent(in) :: val
    !> Result.
    type(env_state_t), intent(inout) :: val_avg

    val_avg = val
    call camp_mpi_reduce_avg_real(val%temp, val_avg%temp)
    call camp_mpi_reduce_avg_real(val%rel_humid, val_avg%rel_humid)
    call camp_mpi_reduce_avg_real(val%pressure, val_avg%pressure)

  end subroutine camp_mpi_reduce_avg_env_state

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Set the temperature (K)
  subroutine set_temperature_K( this, temperature )

    !> Environmental state pointer
    class(env_state_ptr), intent(inout) :: this
    !> New temperature (K)
    real(kind=dp), intent(in) :: temperature

    this%val%temp = temperature

  end subroutine set_temperature_K

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Set the pressure (Pa)
  subroutine set_pressure_Pa( this, pressure )

    !> Environmental state pointer
    class(env_state_ptr), intent(inout) :: this
    !> New pressure (Pa)
    real(kind=dp), intent(in) :: pressure

    this%val%pressure = pressure

  end subroutine set_pressure_Pa

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end module camp_env_state
