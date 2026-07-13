!> The monarch constraint type and associated functions
!! A value constrained by interpolating
!! over monarch values
module boxmodel_monarch_constraint
  use netcdf
  use mpi, only: MPI_COMM_WORLD

  use ntergeo

  use camp_constants, only: i_kind, dp, const
  use camp_util, only: assert_msg, to_string, interp_1d, die_msg, find_1d
  use camp_mpi

  use boxmodel_time_control
  use boxmodel_constraints, only: constraint_t, constraint_ptr, MONARCH_CONSTRAINT
  use boxmodel_io, only: handle_err

  use boxmodel_log

  implicit none
  private

  integer(kind=i_kind), parameter :: MARGIN = 2

  public :: monarch_constraint_from_trajectory, empty_monarch_constraint

  type, extends(constraint_t) :: monarch_constraint_t
    integer(kind=i_kind) :: ntimes
    real(kind=dp), dimension(:), allocatable :: time
    real(kind=dp), dimension(:), allocatable :: value
    !> a time offset: the data before that time point is not considered and we assume t=0 starts at time=time_offset
    !! in the netcdf file. Useful for removing spinoff time for instance
    real(kind=dp) :: time_offset

    type(time_control_t), pointer :: time_control
    class(constraint_ptr), pointer :: latitude_constraint, longitude_constraint, altitude_constraint
  contains

    !> constraint initialization
    procedure :: initializer
    !> get the constraint value at given time and place
    procedure :: get_at

    !> randomize the constraint based on its montecarlo object
    procedure :: randomize

    !> capture min and max of each dimension during the boxmodel run
    procedure :: get_3D_dimensions_boundaries
    procedure :: get_2D_dimensions_boundaries
    !> interpolate the variable of intereste on the constrained trajectory
    procedure :: interpolate_3D_variable_on_traj
    procedure :: interpolate_2D_variable_on_traj

    !> Determine the number of bytes required to pack the variable
    procedure :: internal_pack_size
    !> Pack the given variable into a buffer, advancing position
    procedure :: internal_bin_pack
    !> Unpack the given variable from a buffer, advancing position
    procedure :: internal_bin_unpack
    !> print informations for debugging
    procedure :: internal_print

    procedure :: internal_type_id

  end type monarch_constraint_t

  !> Constructor for monarch_constraint_t
  interface monarch_constraint_t
    procedure :: monarch_constraint_from_trajectory, empty_monarch_constraint
  end interface monarch_constraint_t

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  function empty_monarch_constraint() result(new_obj)
    !> a new constraint instance
    type(monarch_constraint_t), pointer :: new_obj

    allocate (new_obj)
  end function empty_monarch_constraint

  !> constructors for the monarch constraint
  function monarch_constraint_from_trajectory( &
    time_control, &
    latitude_constraint, &
    longitude_constraint, &
    altitude_constraint) result(new_obj)
    !> a new constraint instance
    type(monarch_constraint_t), pointer :: new_obj
    !> time control and constraints needed to interpolate the monarch data
    !> needs to be a copy of the time_control that will be used
    !> by the boxmodel
    type(time_control_t), target, intent(in) :: time_control
    class(constraint_ptr), target, intent(in)  :: latitude_constraint, longitude_constraint, altitude_constraint

    call assert_msg(165900407, &
      latitude_constraint%is_initialized(), &
      "initializing monarch constraint requires initialized latitude constraint")
    call assert_msg(165900407, &
      longitude_constraint%is_initialized(), &
      "initializing monarch constraint requires initialized longitude constraint")
    call assert_msg(165900407, &
      altitude_constraint%is_initialized(), &
      "initializing monarch constraint requires initialized altitude constraint")

    allocate (new_obj)
    new_obj%time_control => time_control_t(time_control)
    new_obj%latitude_constraint => latitude_constraint
    new_obj%longitude_constraint => longitude_constraint
    new_obj%altitude_constraint => altitude_constraint
  end function monarch_constraint_from_trajectory

  !> initialize monarch constraint with provided information
  !! here we interpolate in advance and only store the interpolated time
  !! evolution of the variable.
  subroutine initializer(this)
    !> constraint data
    class(monarch_constraint_t), intent(inout) :: this
    !> experiment to read
    character(len=:), allocatable :: exp_name
    character(len=:), allocatable :: path
    character(len=:), allocatable :: exp_full_path
    character(len=:), allocatable :: ncfile_name

    !> name of the variables to read in file
    character(len=:), allocatable :: variable_name
    character(len=:), allocatable :: latitude_name
    character(len=:), allocatable :: longitude_name
    character(len=:), allocatable :: altitude_name
    character(len=:), allocatable :: time_name

    !> local variables
    character(len=:), allocatable :: key_name
    integer(kind=i_kind)          :: status, ncid
    integer(kind=i_kind)          :: lat_varid, lon_varid, alt_varid, time_varid, var_varid, rotpol_varid
    integer(kind=i_kind)          :: lat_dimid, lon_dimid, alt_dimid, time_dimid
    real(kind=dp)                 :: rotpol_lon, rotpol_lat
    logical                       :: flag_2D

    !> subsetting the netcdf data
    real(kind=dp), dimension(:), allocatable    :: latitude, longitude
    real(kind=dp), dimension(:, :, :), allocatable    :: altitude
    real(kind=dp), dimension(:, :, :, :), allocatable :: variable_3d
    real(kind=dp), dimension(:, :, :), allocatable :: variable_2d
    real(kind=dp), dimension(:), allocatable    :: time
    integer(kind=i_kind) :: numlats, numlons, numalts, numtimes
    integer(kind=i_kind), dimension(2)   ::  lat_dimbounds, lon_dimbounds, alt_dimbounds, time_dimbounds

    call assert_msg(94262593, &
      associated(this%property_set), &
      "Missing property set needed to initialize constraint")

    ! \todo: get units from netcdf file attributes
    key_name = "value_unit"
    call assert_msg(535716172, &
      this%property_set%get_string(key_name, this%value_unit), &
      "Missing unit for monarch constraint")

    key_name = "time_unit"
    call assert_msg(872674534, &
      this%property_set%get_string(key_name, this%time_unit), &
      "Missing unit for monarch constraint")

    ! get the required info
    key_name = "exp_name"
    call assert_msg(917030769, &
      this%property_set%get_string(key_name, exp_name), &
      "Missing 'exp_name' needed for monarch constraint")

    key_name = "ncfile_name"
    call assert_msg(948319891, &
      this%property_set%get_string(key_name, ncfile_name), &
      "Missing 'ncfile_name' needed for monarch constraint")

    key_name = "variable_name"
    call assert_msg(310873369, &
      this%property_set%get_string(key_name, variable_name), &
      "Missing 'variable_name' needed for monarch constraint")

    key_name = "latitude_name"
    call assert_msg(501094502, &
      this%property_set%get_string(key_name, latitude_name), &
      "Missing 'latitude_name' needed for monarch constraint")

    key_name = "longitude_name"
    call assert_msg(836153090, &
      this%property_set%get_string(key_name, longitude_name), &
      "Missing 'longitude_name' needed for monarch constraint")

    key_name = "time_name"
    call assert_msg(797298585, &
      this%property_set%get_string(key_name, time_name), &
      "Missing 'time_name' needed for monarch constraint")

    key_name = "altitude_name"
    ! altitude name optional, if not present assume interpolation of 2d field (lon, lat)
    flag_2D = .not. this%property_set%get_string(key_name, altitude_name)
    ! get optional info
    key_name = "path"
    if (.not. this%property_set%get_string(key_name, path)) then
      path = "/esarchive/exp/monarch/"
    end if

    key_name = "time_offset"
    ! time offset optional. assume same unit as time_unit
    if (.not. this%property_set%get_real(key_name, this%time_offset)) then
      this%time_offset = 0.
    end if

    ! \todo find a way to detect in the exp directory wich netcdf files
    ! to read
    exp_full_path = path//exp_name//"/original_files/test_data/"//ncfile_name

    status = nf90_open(path=exp_full_path, mode=0, ncid=ncid)
    call handle_err(24257949, status)

    ! longitude dimension
    status = nf90_inq_dimid(ncid, "rlon", lon_dimid)
    call handle_err(131261340, status)
    status = nf90_inquire_dimension(ncid, lon_dimid, len=numlons)
    call handle_err(898098838, status)

    ! altitude dimension
    if (.not. flag_2D) then
      status = nf90_inq_dimid(ncid, "lm", alt_dimid)
      call handle_err(157009930, status)
      status = nf90_inquire_dimension(ncid, alt_dimid, len=numalts)
      call handle_err(514889380, status)
    end if
    ! time dimension
    status = nf90_inq_dimid(ncid, "time", time_dimid)
    call handle_err(444795355, status)
    status = nf90_inquire_dimension(ncid, time_dimid, len=numtimes)
    call handle_err(44823838, status)

    ! latitude dimension
    status = nf90_inq_dimid(ncid, "rlat", lat_dimid)
    call handle_err(942531570, status)
    status = nf90_inquire_dimension(ncid, lat_dimid, len=numlats)
    call handle_err(333723971, status)

    ! rotated latitude variable
    status = nf90_inq_varid(ncid, latitude_name, lat_varid)
    call handle_err(778758091, status)
    allocate (latitude(numlats))
    status = nf90_get_var(ncid, lat_varid, latitude)
    call handle_err(772387692, status)

    ! longitude variable 2d
    status = nf90_inq_varid(ncid, longitude_name, lon_varid)
    call handle_err(303945343, status)
    allocate (longitude(numlons))
    status = nf90_get_var(ncid, lon_varid, longitude)
    call handle_err(772387692, status)

    if (.not. flag_2D) then
      ! altitude variable 4d, only take a slice in the time dimension, assuming it doesn´t vary with time
      status = nf90_inq_varid(ncid, altitude_name, alt_varid)
      call handle_err(905117104, status)
      allocate (altitude(numlons, numlats, numalts))
      status = nf90_get_var(ncid, alt_varid, altitude, &
        start=(/1, 1, 1, 2/), count=(/numlons, numlats, numalts, 1/))
      call handle_err(481418157, status)
    end if

    ! time variable 1d
    status = nf90_inq_varid(ncid, time_name, time_varid)
    call handle_err(905117104, status)
    allocate (time(numtimes))
    status = nf90_get_var(ncid, time_varid, time)
    call handle_err(481418157, status)

    ! convert monarch time unit to seconds
    select case (this%time_unit)
     case ("h", "hours")
      time = time*3600.
      this%time_offset = this%time_offset*3600.
     case ("s", "seconds")
     case default
      call die_msg(785849517, &
        "Error in monarch, unknown time unit: "// &
        this%time_unit)
    end select

    status = nf90_inq_varid(ncid, variable_name, var_varid)
    call handle_err(905117104, status)

    status = nf90_inq_varid(ncid, "rotated_pole", rotpol_varid)
    call handle_err(115535902, status)
    status = nf90_get_att(ncid, rotpol_varid, "grid_north_pole_latitude", rotpol_lat)
    call handle_err(984485748, status)
    status = nf90_get_att(ncid, rotpol_varid, "grid_north_pole_longitude", rotpol_lon)
    call handle_err(7832950, status)
    if (flag_2D) then
      call this%get_2D_dimensions_boundaries(latitude, longitude, rotpol_lat, rotpol_lon, time, &
        lat_dimbounds, lon_dimbounds, time_dimbounds)
    else
      call this%get_3d_dimensions_boundaries(latitude, longitude, rotpol_lat, rotpol_lon, altitude, time, &
        lat_dimbounds, lon_dimbounds, alt_dimbounds, time_dimbounds)
    end if

    !! offset time because of time_offset
    time = time - time(time_dimbounds(1))
    time_dimbounds(2) = time_dimbounds(2) + time_dimbounds(1) - 1

    ! read the variable within these boundaries
    numlons = lon_dimbounds(2) - lon_dimbounds(1) + 1
    numlats = lat_dimbounds(2) - lat_dimbounds(1) + 1
    numtimes = time_dimbounds(2) - time_dimbounds(1) + 1
    if (.not. flag_2D) then
      numalts = alt_dimbounds(2) - alt_dimbounds(1) + 1
    end if
    if (flag_2D) then
      allocate (variable_2d(numlons, numlats, numtimes))
      status = nf90_get_var(ncid, var_varid, variable_2d, &
        start=(/lon_dimbounds(1), lat_dimbounds(1), time_dimbounds(1)/), &
        count=(/numlons, numlats, numtimes/))
      call handle_err(199784546, status)
      call this%interpolate_2D_variable_on_traj(variable_2d, &
        longitude(lon_dimbounds(1):lon_dimbounds(2)), &
        latitude(lat_dimbounds(1):lat_dimbounds(2)), &
        time(time_dimbounds(1):time_dimbounds(2)), &
        rotpol_lat, &
        rotpol_lon)
    else
      allocate (variable_3d(numlons, numlats, numalts, numtimes))
      status = nf90_get_var(ncid, var_varid, variable_3d, &
        start=(/lon_dimbounds(1), lat_dimbounds(1), alt_dimbounds(1), time_dimbounds(1)/), &
        count=(/numlons, numlats, numalts, numtimes/))
      call handle_err(178453157, status)
      !> interpolate the variables on the restricted domain defined above
      call this%interpolate_3D_variable_on_traj(variable_3d, &
        longitude(lon_dimbounds(1):lon_dimbounds(2)), &
        latitude(lat_dimbounds(1):lat_dimbounds(2)), &
        altitude(lon_dimbounds(1):lon_dimbounds(2), &
        lat_dimbounds(1):lat_dimbounds(2), &
        alt_dimbounds(1):alt_dimbounds(2)), &
        time(time_dimbounds(1):time_dimbounds(2)), &
        rotpol_lat, &
        rotpol_lon)
    end if

  end subroutine initializer

  !> capture min and max of each dimension during the boxmodel run
  subroutine get_3D_dimensions_boundaries(this, &
    latitude, longitude, rotpol_lat, rotpol_lon, altitude, time, &
    lat_dimbounds, lon_dimbounds, alt_dimbounds, time_dimbounds)
    class(monarch_constraint_t), intent(inout)          :: this
    real(kind=dp), dimension(:), intent(in)          :: time
    real(kind=dp), dimension(:), intent(in)       :: latitude, longitude
    real(kind=dp) :: rotpol_lat, rotpol_lon
    real(kind=dp), dimension(:, :, :), intent(in) :: altitude
    integer(kind=i_kind), dimension(2), intent(out)  :: lat_dimbounds, lon_dimbounds, alt_dimbounds, time_dimbounds
    real(kind=dp), dimension(2)                      :: lat_bounds, lon_bounds, alt_bounds, time_bounds

    type(time_control_t) :: time_control
    real(kind=dp) :: curr_time, curr_latitude, curr_longitude, curr_altitude
    real(kind=dp) :: curr_rotated_latitude, curr_rotated_longitude
    integer(kind=i_kind) :: i_lon, i_lat, i_alt, i_time

    time_control = this%time_control

    lat_bounds = (/huge(real(0.0, kind=dp)), -huge(real(0.0, kind=dp))/)
    lon_bounds = (/huge(real(0.0, kind=dp)), -huge(real(0.0, kind=dp))/)
    alt_bounds = (/huge(real(0.0, kind=dp)), -huge(real(0.0, kind=dp))/)
    time_bounds = (/huge(real(0.0, kind=dp)), -huge(real(0.0, kind=dp))/)

    ! loop over the whole simulation time to capture simulation bounds

    this%ntimes = 0
    do
      this%ntimes = this%ntimes + 1
      curr_time = time_control%get_current_time()
      if (curr_time < time_bounds(1)) then
        time_bounds(1) = curr_time
      end if
      if (curr_time > time_bounds(2)) then
        time_bounds(2) = curr_time
      end if

      ! update lon and lat (degrees)
      curr_latitude = this%latitude_constraint%get(curr_time)
      select case (this%latitude_constraint%value_unit())
       case ("deg", "degree", "degrees", "°", "d") ! do nothing
       case default
        call die_msg(753619967, &
          "Error in latitude constraint, unknown unit: "// &
          this%latitude_constraint%value_unit())
      end select

      curr_longitude = this%longitude_constraint%get(curr_time)
      select case (this%longitude_constraint%value_unit())
       case ("deg", "degree", "degrees", "°", "d") ! do nothing
       case default
        call die_msg(623275069, &
          "Error in longitude constraint, unknown unit: "// &
          this%longitude_constraint%value_unit())
      end select

      ! transform to rotated frame
      call lltorll(curr_latitude, curr_longitude, rotpol_lat, rotpol_lon, curr_rotated_latitude, curr_rotated_longitude)

      if (curr_rotated_latitude < lat_bounds(1)) then
        lat_bounds(1) = curr_rotated_latitude
      end if
      if (curr_rotated_latitude > lat_bounds(2)) then
        lat_bounds(2) = curr_rotated_latitude
      end if

      if (curr_rotated_longitude < lon_bounds(1)) then
        lon_bounds(1) = curr_rotated_longitude
      end if
      if (curr_rotated_longitude > lon_bounds(2)) then
        lon_bounds(2) = curr_rotated_longitude
      end if

      ! update altitude (m)
      curr_altitude = this%altitude_constraint%get(curr_time)
      select case (this%altitude_constraint%value_unit())
       case ("m", "meter", "meters") ! do nothing
       case ("cm", "centimeter", "centimeters")
        curr_altitude = curr_altitude/100.0
       case ("km", "kilometer", "kilometers")
        curr_altitude = curr_altitude*1000.0
       case default
        call die_msg(524070400, &
          "Error in altitude constraint, unknown unit: "// &
          this%altitude_constraint%value_unit())
      end select
      if (curr_altitude < alt_bounds(1)) then
        alt_bounds(1) = curr_altitude
      end if
      if (curr_altitude > alt_bounds(2)) then
        alt_bounds(2) = curr_altitude
      end if

      if (.not. time_control%increment()) exit

    end do

    allocate (this%time(this%ntimes))
    allocate (this%value(this%ntimes))

    ! find dimension boundaries encompassing these bounds, with a margin
    if (time_bounds(1) < this%time_offset) then
      time_bounds(1) = this%time_offset
    end if

    time_dimbounds(1) = min(max(1, find_1d(size(time), time, time_bounds(1))), size(time))
    time_dimbounds(2) = min(max(1, find_1d(size(time), time, time_bounds(2))), size(time))

    lat_dimbounds(1) = min(max(1, find_1d(size(latitude), latitude, lat_bounds(1)) - MARGIN), size(latitude))
    lat_dimbounds(2) = min(max(1, find_1d(size(latitude), latitude, lat_bounds(2)) + MARGIN), size(latitude))

    lon_dimbounds(1) = min(max(1, find_1d(size(longitude), longitude, lon_bounds(1)) - MARGIN), size(longitude))
    lon_dimbounds(2) = min(max(1, find_1d(size(longitude), longitude, lon_bounds(2)) + MARGIN), size(longitude))

    alt_dimbounds = (/huge(int(0, kind=i_kind)), -huge(int(0, kind=i_kind))/)

    do i_lat = lat_dimbounds(1), lat_dimbounds(2)
      do i_lon = lon_dimbounds(1), lon_dimbounds(2)
        do i_alt = 1, size(altitude, 3) - 1
          ! reminder, vertical levels are reversed
          if (altitude(i_lon, i_lat, i_alt) > alt_bounds(1) .and. &
            altitude(i_lon, i_lat, i_alt + 1) < alt_bounds(1) .and. &
            alt_dimbounds(2) < (i_alt + 1 + MARGIN)) then
            alt_dimbounds(2) = i_alt + 1
          end if
          if (altitude(i_lon, i_lat, i_alt) > alt_bounds(2) .and. &
            altitude(i_lon, i_lat, i_alt + 1) < alt_bounds(2) .and. &
            alt_dimbounds(1) > i_alt) then
            alt_dimbounds(1) = i_alt
          end if
        end do
      end do
    end do
    alt_dimbounds(1) = max(1, alt_dimbounds(1) - MARGIN)
    alt_dimbounds(2) = min(size(altitude, 3), alt_dimbounds(2) + MARGIN)

  end subroutine get_3d_dimensions_boundaries

  !> capture min and max of each dimension during the boxmodel run
  subroutine get_2D_dimensions_boundaries(this, &
    latitude, longitude, rotpol_lat, rotpol_lon, time, &
    lat_dimbounds, lon_dimbounds, time_dimbounds)
    class(monarch_constraint_t), intent(inout)          :: this
    real(kind=dp), dimension(:), intent(in)          :: time
    real(kind=dp), dimension(:), intent(in)       :: latitude, longitude
    real(kind=dp) :: rotpol_lat, rotpol_lon
    integer(kind=i_kind), dimension(2), intent(out)  :: lat_dimbounds, lon_dimbounds, time_dimbounds
    real(kind=dp), dimension(2)                      :: lat_bounds, lon_bounds, alt_bounds, time_bounds

    type(time_control_t) :: time_control
    real(kind=dp) :: curr_time, curr_latitude, curr_longitude
    real(kind=dp) :: curr_rotated_latitude, curr_rotated_longitude
    integer(kind=i_kind) :: i_lon, i_lat, i_alt, i_time

    time_control = this%time_control

    lat_bounds = (/huge(real(0.0, kind=dp)), -huge(real(0.0, kind=dp))/)
    lon_bounds = (/huge(real(0.0, kind=dp)), -huge(real(0.0, kind=dp))/)
    time_bounds = (/huge(real(0.0, kind=dp)), -huge(real(0.0, kind=dp))/)

    ! loop over the whole simulation time to capture simulation bounds

    this%ntimes = 0
    do
      this%ntimes = this%ntimes + 1
      curr_time = time_control%get_current_time()
      if (curr_time < time_bounds(1)) then
        time_bounds(1) = curr_time
      end if
      if (curr_time > time_bounds(2)) then
        time_bounds(2) = curr_time
      end if

      ! update lon and lat (degrees)
      curr_latitude = this%latitude_constraint%get(curr_time)
      select case (this%latitude_constraint%value_unit())
       case ("deg", "degree", "degrees", "���", "d") ! do nothing
       case default
        call die_msg(442390620, &
          "Error in latitude constraint, unknown unit: "// &
          this%latitude_constraint%value_unit())
      end select

      curr_longitude = this%longitude_constraint%get(curr_time)
      select case (this%longitude_constraint%value_unit())
       case ("deg", "degree", "degrees", "°", "d") ! do nothing
       case default
        call die_msg(269588554, &
          "Error in longitude constraint, unknown unit: "// &
          this%longitude_constraint%value_unit())
      end select

      ! transform to rotated frame
      call lltorll(curr_latitude, curr_longitude, rotpol_lat, rotpol_lon, curr_rotated_latitude, curr_rotated_longitude)

      if (curr_rotated_latitude < lat_bounds(1)) then
        lat_bounds(1) = curr_rotated_latitude
      end if
      if (curr_rotated_latitude > lat_bounds(2)) then
        lat_bounds(2) = curr_rotated_latitude
      end if

      if (curr_rotated_longitude < lon_bounds(1)) then
        lon_bounds(1) = curr_rotated_longitude
      end if
      if (curr_rotated_longitude > lon_bounds(2)) then
        lon_bounds(2) = curr_rotated_longitude
      end if

      if (.not. time_control%increment()) exit

    end do

    allocate (this%time(this%ntimes))
    allocate (this%value(this%ntimes))

    ! find dimension boundaries encompassing these bounds, with a margin for lon and lat
    if (time_bounds(1) < this%time_offset) then
      time_bounds(1) = this%time_offset
    end if

    time_dimbounds(1) = min(max(1, find_1d(size(time), time, time_bounds(1))), size(time))
    time_dimbounds(2) = min(max(1, find_1d(size(time), time, time_bounds(2))), size(time))

    lat_dimbounds(1) = min(max(1, find_1d(size(latitude), latitude, lat_bounds(1)) - MARGIN), size(latitude))
    lat_dimbounds(2) = min(max(1, find_1d(size(latitude), latitude, lat_bounds(2)) + MARGIN), size(latitude))

    lon_dimbounds(1) = min(max(1, find_1d(size(longitude), longitude, lon_bounds(1)) - MARGIN), size(longitude))
    lon_dimbounds(2) = min(max(1, find_1d(size(longitude), longitude, lon_bounds(2)) + MARGIN), size(longitude))

  end subroutine get_2d_dimensions_boundaries

  !> interpolate the variable on the trajectory
  !! populates this%time and this%value
  subroutine interpolate_3d_variable_on_traj(this, variable, longitude, latitude, altitude, time, rotpol_lat, rotpol_lon)
    class(monarch_constraint_t), intent(inout)    :: this
    real(kind=dp), dimension(:, :, :, :), intent(in) :: variable
    real(kind=dp), dimension(:, :, :), intent(in)   :: altitude
    real(kind=dp), dimension(:), intent(in)     :: latitude, longitude
    real(kind=dp) :: rotpol_lat, rotpol_lon
    real(kind=dp), dimension(:), intent(in)       :: time

    integer(kind=i_kind) :: nlat, nlon, nalt, ntime, ndim, maxdim, np
    integer(kind=i_kind) :: i_lon, i_lat, i_alt, i_time, i_dim
    integer(kind=i_kind), dimension(:), allocatable :: kdim

    real(kind=dp), dimension(:, :, :), allocatable :: rev_altitude
    real(kind=dp), dimension(:), allocatable :: table1d
    real(kind=dp), dimension(:, :), allocatable :: vect1d
    real(kind=dp), dimension(:, :, :, :, :), allocatable :: vect
    real(kind=dp), dimension(:), allocatable :: avedelta, maxdelta
    real(kind=dp), dimension(:), allocatable :: vtable

    real(kind=dp) :: curr_longitude, curr_latitude, curr_altitude, curr_time
    real(kind=dp) :: curr_rotated_longitude, curr_rotated_latitude

    real(kind=dp) :: resu
    integer(kind=i_kind) :: inei
    integer(kind=i_kind), dimension(:, :), allocatable :: neighbours
    real(kind=dp), dimension(:), allocatable :: weights
    logical :: found

    type(time_control_t) :: time_control

    nlat = size(latitude)
    nlon = size(longitude)
    nalt = size(altitude, 3)
    ntime = size(time)

    ndim = 4
    allocate (kdim(0:ndim))
    kdim(0) = 1
    kdim(1) = nlon
    kdim(2) = nlat
    kdim(3) = nalt
    kdim(4) = ntime

    allocate (vect(ndim, kdim(1), kdim(2), kdim(3), kdim(4)))
    maxdim = kdim(1)*kdim(2)*kdim(3)*kdim(4)

    vect = 0.0

    do i_lon = 1, kdim(1)
      vect(1, i_lon, :, :, :) = longitude(i_lon)
    end do
    do i_lat = 1, kdim(2)
      vect(2, :, i_lat, :, :) = latitude(i_lat)
    end do
    do i_lon = 1, kdim(1)
      do i_lat = 1, kdim(2)
        do i_alt = 1, kdim(3)
          vect(3, i_lon, i_lat, i_alt, :) = altitude(i_lon, i_lat, kdim(3) - i_alt + 1) ! reverse vertical index
        end do
      end do
    end do
    do i_time = 1, kdim(4)
      vect(4, :, :, :, i_time) = time(i_time)
    end do

    ! transform n-dimensional variable to 1d
    allocate (table1d(maxdim))
    allocate (vect1d(ndim, maxdim))
    np = 1
    do i_time = 1, kdim(4)
      do i_alt = 1, kdim(3)
        do i_lat = 1, kdim(2)
          do i_lon = 1, kdim(1)
            table1d(np) = variable(i_lon, i_lat, kdim(3) - i_alt + 1, i_time) ! need to reverse vertical index
            do i_dim = 1, ndim
              vect1d(i_dim, np) = vect(i_dim, i_lon, i_lat, i_alt, i_time)
            end do
            np = np + 1
          end do
        end do
      end do
    end do

    allocate (avedelta(ndim))
    allocate (maxdelta(ndim))
    allocate (vtable(ndim))
    allocate (neighbours(2**ndim, ndim))
    allocate (weights(2**ndim))
    avedelta = 0.
    maxdelta = 0.
    call findmaxave( &
      ndim, &  ! Input : Integer                   : Dimension of the look-up table : 1, 2, 3, 4, ....
      maxdim, &  ! Input : Integer                   : kdim(1)*kdim(2)*...*kdim(ndim)
      kdim, &  ! Input : Array 1D, Integer         : Array where are stored the number of data in each direction
      vect1D, &  ! Input : Array 2D, Real            : Array where are stored all the interval edges on each dimension
      avedelta, &  ! Output: Array 1D, Real            : Inverse average interval on each dimension
      maxdelta &  ! Output: Array 1D, Real            : Maximum interval on each dimension
      )

    time_control = this%time_control
    neighbours = 0
    i_time = 0
    do
      i_time = i_time + 1
      curr_time = time_control%get_current_time()
      this%time(i_time) = curr_time

      ! update lon and lat (degrees)
      curr_latitude = this%latitude_constraint%get(curr_time)
      select case (this%latitude_constraint%value_unit())
       case ("deg", "degree", "degrees", "°", "d") ! do nothing
       case default
        call die_msg(545631472, &
          "Error in latitude constraint, unknown unit: "// &
          this%latitude_constraint%value_unit())
      end select

      curr_longitude = this%longitude_constraint%get(curr_time)
      select case (this%longitude_constraint%value_unit())
       case ("deg", "degree", "degrees", "��", "d") ! do nothing
       case default
        call die_msg(763542747, &
          "Error in longitude constraint, unknown unit: "// &
          this%longitude_constraint%value_unit())
      end select

      ! transform to rotated frame
      call lltorll(curr_latitude, curr_longitude, rotpol_lat, rotpol_lon, curr_rotated_latitude, curr_rotated_longitude)

      ! update altitude (m)
      curr_altitude = this%altitude_constraint%get(curr_time)
      select case (this%altitude_constraint%value_unit())
       case ("m", "meter", "meters") ! do nothing
       case ("cm", "centimeter", "centimeters")
        curr_altitude = curr_altitude/100.0
       case ("km", "kilometer", "kilometers")
        curr_altitude = curr_altitude*1000.0
       case default
        call die_msg(333490832, &
          "Error in altitude constraint, unknown unit: "// &
          this%altitude_constraint%value_unit())
      end select

      ! need to convert time to hour for monarch
      curr_time = curr_time

      ! longitude of interpolated point
      vtable(1) = curr_rotated_longitude
      ! latitude of interpolated point
      vtable(2) = curr_rotated_latitude
      ! altitude of interpolated point
      vtable(3) = curr_altitude
      ! time of interpolated point
      vtable(4) = curr_time

      resu = 0.0
      call interpolation_general( &
        ndim, &  ! Scalar    : Integer   :I  : Dimension of the problem N > 1
        maxdim, &  ! Scalar    : Integer   :I  : Product of the number of elements of the input grid data on each dimension
        kdim, &  ! Array 1D  : Integer   :I  : Vector where are stored the number of data in each dimension in each direction(0:ndim)
        vect1D, &  ! Array 2D  : Real      :I  : Array where are stored all the interva edges for each dimension
        vtable, &  ! Array 1D  : Real      :I  : Coordinate values for the point you want to interpolate
        table1D, &  ! Array 1D  : Real      :I  : Coordinate values of the input grid data
        avedelta, &  ! Array 1D  : Real      :I  : Average intervals of the input grid data on each dimension
        maxdelta, &  ! Array 1D  : Real      :I  : Maximum intervals of the input grid on each dimension
        resu, &  ! Scalar    : Real      :O  : Result of the interpolation
        inei, &  ! Scalar    : Integer   :I/O: Number of neighbours (reusable as Input for next point)
        neighbours, &  ! Array 2D  : Integer   :I/O: Coordinates of neighbours in all directions ndim (reusable as Input for next point)
        weights, &  ! Array 1D  : Real      :O  : Weight factors
        found      &  ! Scalar    : Logical   :O  : True if found, False if not found
      &)

      ! call assert_msg(3730961567, &
      !    found,&
      !    "Could not interpolate point on monarch grid")

      this%value(i_time) = resu

      if (.not. time_control%increment()) exit
    end do

  end subroutine interpolate_3D_variable_on_traj

  subroutine interpolate_2d_variable_on_traj(this, variable, longitude, latitude, time, rotpol_lat, rotpol_lon)
    class(monarch_constraint_t), intent(inout)    :: this
    real(kind=dp), dimension(:, :, :), intent(in) :: variable
    real(kind=dp), dimension(:), intent(in)     :: latitude, longitude
    real(kind=dp) :: rotpol_lat, rotpol_lon
    real(kind=dp), dimension(:), intent(in)       :: time

    integer(kind=i_kind) :: nlat, nlon, nalt, ntime, ndim, maxdim, np
    integer(kind=i_kind) :: i_lon, i_lat, i_alt, i_time, i_dim
    integer(kind=i_kind), dimension(:), allocatable :: kdim

    real(kind=dp), dimension(:, :, :), allocatable :: rev_altitude
    real(kind=dp), dimension(:), allocatable :: table1d
    real(kind=dp), dimension(:, :), allocatable :: vect1d
    real(kind=dp), dimension(:, :, :, :), allocatable :: vect
    real(kind=dp), dimension(:), allocatable :: avedelta, maxdelta
    real(kind=dp), dimension(:), allocatable :: vtable

    real(kind=dp) :: curr_longitude, curr_latitude, curr_time
    real(kind=dp) :: curr_rotated_longitude, curr_rotated_latitude

    real(kind=dp) :: resu
    integer(kind=i_kind) :: inei
    integer(kind=i_kind), dimension(:, :), allocatable :: neighbours
    real(kind=dp), dimension(:), allocatable :: weights
    logical :: found

    type(time_control_t) :: time_control

    nlat = size(latitude)
    nlon = size(longitude)
    ntime = size(time)

    ndim = 3
    allocate (kdim(0:ndim))
    kdim(0) = 1
    kdim(1) = nlon
    kdim(2) = nlat
    kdim(3) = ntime

    allocate (vect(ndim, kdim(1), kdim(2), kdim(3)))
    maxdim = kdim(1)*kdim(2)*kdim(3)

    vect = 0.0

    do i_lon = 1, kdim(1)
      vect(1, i_lon, :, :) = longitude(i_lon)
    end do
    do i_lat = 1, kdim(2)
      vect(2, :, i_lat, :) = latitude(i_lat)
    end do
    do i_time = 1, kdim(3)
      vect(3, :, :, i_time) = time(i_time)
    end do

    ! transform n-dimensional variable to 1d
    allocate (table1d(maxdim))
    allocate (vect1d(ndim, maxdim))
    np = 1
    do i_time = 1, kdim(3)
      do i_lat = 1, kdim(2)
        do i_lon = 1, kdim(1)
          table1d(np) = variable(i_lon, i_lat, i_time) ! need to reverse vertical index
          do i_dim = 1, ndim
            vect1d(i_dim, np) = vect(i_dim, i_lon, i_lat, i_time)
          end do
          np = np + 1
        end do
      end do
    end do

    allocate (avedelta(ndim))
    allocate (maxdelta(ndim))
    allocate (vtable(ndim))
    allocate (neighbours(2**ndim, ndim))
    allocate (weights(2**ndim))
    avedelta = 0.
    maxdelta = 0.
    call findmaxave( &
      ndim, &  ! Input : Integer                   : Dimension of the look-up table : 1, 2, 3, 4, ....
      maxdim, &  ! Input : Integer                   : kdim(1)*kdim(2)*...*kdim(ndim)
      kdim, &  ! Input : Array 1D, Integer         : Array where are stored the number of data in each direction
      vect1D, &  ! Input : Array 2D, Real            : Array where are stored all the interval edges on each dimension
      avedelta, &  ! Output: Array 1D, Real            : Inverse average interval on each dimension
      maxdelta &  ! Output: Array 1D, Real            : Maximum interval on each dimension
      )

    time_control = this%time_control
    neighbours = 0
    i_time = 0
    do
      i_time = i_time + 1
      curr_time = time_control%get_current_time()
      this%time(i_time) = curr_time

      ! update lon and lat (degrees)
      curr_latitude = this%latitude_constraint%get(curr_time)
      select case (this%latitude_constraint%value_unit())
       case ("deg", "degree", "degrees", "°", "d") ! do nothing
       case default
        call die_msg(993919793, &
          "Error in latitude constraint, unknown unit: "// &
          this%latitude_constraint%value_unit())
      end select

      curr_longitude = this%longitude_constraint%get(curr_time)
      select case (this%longitude_constraint%value_unit())
       case ("deg", "degree", "degrees", "°", "d") ! do nothing
       case default
        call die_msg(387650754, &
          "Error in longitude constraint, unknown unit: "// &
          this%longitude_constraint%value_unit())
      end select

      ! transform to rotated frame
      call lltorll(curr_latitude, curr_longitude, rotpol_lat, rotpol_lon, curr_rotated_latitude, curr_rotated_longitude)

      ! need to convert time to hour for monarch
      curr_time = curr_time

      ! longitude of interpolated point
      vtable(1) = curr_rotated_longitude
      ! latitude of interpolated point
      vtable(2) = curr_rotated_latitude
      ! time of interpolated point
      vtable(3) = curr_time

      resu = 0.0
      call interpolation_general( &
        ndim, &  ! Scalar    : Integer   :I  : Dimension of the problem N > 1
        maxdim, &  ! Scalar    : Integer   :I  : Product of the number of elements of the input grid data on each dimension
        kdim, &  ! Array 1D  : Integer   :I  : Vector where are stored the number of data in each dimension in each direction(0:ndim)
        vect1D, &  ! Array 2D  : Real      :I  : Array where are stored all the interva edges for each dimension
        vtable, &  ! Array 1D  : Real      :I  : Coordinate values for the point you want to interpolate
        table1D, &  ! Array 1D  : Real      :I  : Coordinate values of the input grid data
        avedelta, &  ! Array 1D  : Real      :I  : Average intervals of the input grid data on each dimension
        maxdelta, &  ! Array 1D  : Real      :I  : Maximum intervals of the input grid on each dimension
        resu, &  ! Scalar    : Real      :O  : Result of the interpolation
        inei, &  ! Scalar    : Integer   :I/O: Number of neighbours (reusable as Input for next point)
        neighbours, &  ! Array 2D  : Integer   :I/O: Coordinates of neighbours in all directions ndim (reusable as Input for next point)
        weights, &  ! Array 1D  : Real      :O  : Weight factors
        found      &  ! Scalar    : Logical   :O  : True if found, False if not found
      &)

      ! call assert_msg(3730961567, &
      !    found,&
      !    "Could not interpolate point on monarch grid")

      this%value(i_time) = resu

      if (.not. time_control%increment()) exit
    end do

  end subroutine interpolate_2d_variable_on_traj

  !> interpolate constrained value at given time
  real(kind=dp) function get_at(this, time)
    class(monarch_constraint_t), intent(in) :: this
    real(kind=dp), intent(in) :: time
    real(kind=dp) :: current_time

    current_time = time

    call assert_msg(240819926, &
      this%time(1) <= current_time .and. &
      this%time(size(this%time)) >= current_time, &
      "Constraint error: current time:"//trim(to_string(current_time))// &
      " is not encompassed by monarch constraint")

    get_at = interp_1d(this%time, this%value, current_time)

  end function get_at

  subroutine randomize(this)
    class(monarch_constraint_t), intent(inout) :: this

    if (this%montecarlo_fg) then
      call this%montecarlo%val%randomize_array(this%value)
    else
      call warn_msg(881433699, " trying to randomize a monarch constraint without a montecarlo object")
    end if
  end subroutine randomize

  !> transformation from lat-lon to rotated lat-lon coordinates
  pure elemental subroutine lltorll(old_lat, old_lon, ref_lat, ref_lon, new_lat, new_lon)

    real(kind=dp), intent(in)  :: old_lat, old_lon, ref_lat, ref_lon
    real(kind=dp), intent(out) :: new_lat, new_lon
    real(kind=dp)              :: lon, lat, SP_lon, SP_lat, theta, phi, x, y, z, x_new, y_new, z_new, dtr

    ! dtr = degrees to radians PI / 180.0
    dtr = const%pi/180.0

    lon = old_lon*dtr
    lat = old_lat*dtr
    if (ref_lon > 0) then
      SP_lon = ref_lon - 180.0
    else
      SP_lon = ref_lon + 180.0
    end if
    SP_lat = -ref_lat
    theta = (90 + SP_lat)*dtr
    phi = SP_lon*dtr

    x = cos(lon)*cos(lat)
    y = sin(lon)*cos(lat)
    z = sin(lat)

    x_new = cos(theta)*cos(phi)*x + cos(theta)*sin(phi)*y + sin(theta)*z
    y_new = -sin(phi)*x + cos(phi)*y;
    z_new = -sin(theta)*cos(phi)*x - sin(theta)*sin(phi)*y + cos(theta)*z

    new_lon = atan2(y_new, x_new)/dtr
    new_lat = asin(z_new)/dtr

  end subroutine lltorll

  subroutine internal_print(this, f_unit)
    class(monarch_constraint_t), intent(in) :: this
    integer(kind=i_kind), intent(in), optional :: f_unit
    integer(kind=i_kind) :: i_time

    if (present(f_unit)) then
      write (f_unit, *) "**************************"
      write (f_unit, *) "*** MONARCH constraint ***"
      write (f_unit, *) "**************************"

      write (f_unit, *) " *** number of points = ", this%ntimes
      write (f_unit, *) "*** points values ****"
      write (f_unit, *) "time, value"
      do i_time = 1, this%ntimes
        write (f_unit, *) this%time(i_time), this%value(i_time)
      end do
      write (f_unit, *) "******** end ********"
    else
      call thread_log%debug("**************************")
      call thread_log%debug("*** MONARCH constraint ***")
      call thread_log%debug("**************************")

      call thread_log%debug("number of points = "//to_string(this%ntimes))
      call thread_log%debug("*** points values ****")
      call thread_log%debug("time, value")
      do i_time = 1, this%ntimes
        call thread_log%debug(to_string(this%time(i_time))//","//to_string(this%value(i_time)))
      end do
      call thread_log%debug("******** end ********")
    endif

  end subroutine internal_print

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  integer(kind=i_kind) function internal_pack_size(this, comm)
    !> Reaction update data
    class(monarch_constraint_t), intent(in) :: this
    !> MPI communicator
    integer, intent(in), optional :: comm

#ifdef CAMP_USE_MPI
    integer :: l_comm

    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if

    internal_pack_size = camp_mpi_pack_size_integer(this%ntimes, l_comm) + &
      camp_mpi_pack_size_real_array(this%time, l_comm) + &
      camp_mpi_pack_size_real_array(this%value, l_comm)
#else
    internal_pack_size = 0
#endif
  end function internal_pack_size

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine internal_bin_pack(this, buffer, pos, comm)
    class(monarch_constraint_t), intent(in) :: this
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos
    !> MPI communicator
    integer, intent(in), optional :: comm

#ifdef CAMP_USE_MPI
    integer(kind=i_kind) :: prev_position, l_comm

    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if

    prev_position = pos
    call camp_mpi_pack_integer(buffer, pos, this%ntimes, l_comm)
    call camp_mpi_pack_real_array(buffer, pos, this%time, l_comm)
    call camp_mpi_pack_real_array(buffer, pos, this%value, l_comm)

    call assert(77314480, &
      pos - prev_position <= this%pack_size(l_comm))
#endif
  end subroutine internal_bin_pack

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!> Unpack the given variable from a buffer, advancing position
  subroutine internal_bin_unpack(this, buffer, pos, comm)
    class(monarch_constraint_t), intent(inout) :: this
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos
    !> MPI communicator
    integer, intent(in), optional :: comm

#ifdef CAMP_USE_MPI
    integer(kind=i_kind) :: prev_position, l_comm

    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if

    prev_position = pos
    call camp_mpi_unpack_integer(buffer, pos, this%ntimes, l_comm)
    call camp_mpi_unpack_real_array(buffer, pos, this%time, l_comm)
    call camp_mpi_unpack_real_array(buffer, pos, this%value, l_comm)
    call assert(160146216, &
      pos - prev_position <= this%pack_size(l_comm))
#endif

  end subroutine internal_bin_unpack

  integer(kind=i_kind) function internal_type_id(this)
    class(monarch_constraint_t), intent(in) :: this

    internal_type_id = MONARCH_CONSTRAINT

  end function internal_type_id

end module boxmodel_monarch_constraint
