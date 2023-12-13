!! [Modules to use]
program box_model

  use camp_camp_core
  use camp_camp_state
  use camp_constants
  !! [Modules to use]

  !! [Chem-spec data module]
  use camp_chem_spec_data

#ifdef CAMP_USE_MPI
  use mpi
  use camp_mpi
#endif


  !! [Chem-spec data module]

  !! [Core and state]
  type(camp_core_t), pointer :: camp_core
  type(camp_state_t), pointer :: camp_state

  !! [Core and state]

  !! [Species ids]
  integer(kind=i_kind) :: idx_O3, idx_NO, idx_NO2, idx_O2
  type(chem_spec_data_t), pointer :: chem_spec_data

  !! [Species ids]

  !! [Output variables]
  character(len=*), parameter :: fmt_hdr = "(A10,',',A10,',',A10,',',A10,',',A10)"
  character(len=*), parameter :: fmt_dat = "(ES10.4,',',ES10.4,',',ES10.4,',',ES10.4,',',ES10.4)"

  !! [Output variables]

  !! [Time id]
  integer(kind=i_kind) :: i_time

  !! [Time id]

  ! allows example to run with MPI supprt compiled in
#ifdef CAMP_USE_MPI
  call camp_mpi_init( )
#endif

  !! [Initialize core]
  camp_core => camp_core_t( "my_config_file.json" )
  call camp_core%initialize( )
  !! [Initialize core]

  !! [Get species ids]
  if( .not.camp_core%get_chem_spec_data( chem_spec_data ) ) then
    write(*,*) "Something's gone wrong!"
    stop 3
  end if

  idx_O3  = chem_spec_data%gas_state_id( "O3"  )
  idx_NO  = chem_spec_data%gas_state_id( "NO"  )
  idx_NO2 = chem_spec_data%gas_state_id( "NO2" )
  idx_O2  = chem_spec_data%gas_state_id( "O2"  )
  if( idx_O3.eq.0 .or. idx_NO2.eq.0 .or.idx_O2.eq.0 ) then
    write(*,*) "Missing species!"
    stop 3
  end if
  !! [Get species ids]

  !! [Set initial conditions]
  call camp_core%solver_initialize( )
  camp_state => camp_core%new_state( )

  call camp_state%env_states(1)%set_temperature_K(  275.4_dp )
  call camp_state%env_states(1)%set_pressure_Pa( 101532.2_dp )

  camp_state%state_var( idx_O3   ) = 0.13  ! [O3] in ppm
  camp_state%state_var( idx_NO   ) = 0.02  ! [NO] in ppm
  camp_state%state_var( idx_NO2  ) = 0.053 ! [NO2] in ppm
  camp_state%state_var( idx_O2   ) = 2.1e5 ! [O2] in ppm
  !! [Set initial conditions]

  !! [Solve and output]
  write(*,fmt_hdr) "time", "O3", "NO", "NO2", "O2"
  do i_time = 1, 100
    call camp_core%solve( camp_state, 1.0d-15 ) ! time step in s
    write(*,fmt_dat) i_time*1.0e-15, &
                     camp_state%state_var( idx_O3  ), &
                     camp_state%state_var( idx_NO  ), &
                     camp_state%state_var( idx_NO2 ), &
                     camp_state%state_var( idx_O2  )
  end do

#ifdef CAMP_USE_MPI
  call camp_mpi_finalize( )
#endif

  deallocate( camp_core )
  deallocate( camp_state )

end program box_model
!! [Solve and output]
