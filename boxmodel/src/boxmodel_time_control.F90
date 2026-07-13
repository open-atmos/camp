module boxmodel_time_control

#ifdef CAMP_USE_JSON
  use json_module
#endif
#ifdef CAMP_USE_MPI
  use mpi
#endif

  use camp_constants, only: dp, i_kind, sp
  use camp_util, only: assert_msg
  use camp_property
  use camp_mpi

  use boxmodel_log
  implicit none
  private

  public :: time_control_t

  integer(kind=i_kind), dimension(12), parameter :: months = (/31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31/)

  type :: time_control_t
    type(property_t), public, pointer :: property_set => null()
    integer(kind=i_kind) :: year
    integer(kind=i_kind) :: month
    integer(kind=i_kind) :: day
    real(kind=dp)        :: hour
    real(kind=dp)        :: previous_hour
    real(kind=dp)        :: start_time ! starting time (s)
    real(kind=dp)        :: end_time ! finishing time
    real(kind=dp)        :: tz ! difference with gmt timezone
    real(kind=dp)        :: lon, lat
    !> time step length (s)
    !! remains constant for the whole simulation, may be reduced for the last time step
    !! to exactly reach the end time
    real(kind=dp)         :: current_time_step_length
    real(kind=dp)         :: current_time
    integer(kind=i_kind), public  :: output_time_step_index = 1

    !> time since last photolysis update (s)
    real(kind=dp)         :: last_photolysis_update
    !> photolysis update timestep (s)
    real(kind=dp)         :: photolysis_update_timestep

    !> time since last emissions update (s)
    real(kind=dp)         :: last_emissions_update
    !> emissions update timestep
    real(kind=dp)         :: emissions_update_timestep

    !> time since last deposition update (s)
    real(kind=dp)         :: last_deposition_update
    !> deposition update timestep
    real(kind=dp)         :: deposition_update_timestep

    !> time since last microphysics update (s)
    real(kind=dp)         :: last_microphysics_update
    !> microphysics update timestep
    real(kind=dp)         :: microphysics_update_timestep

    !> time since last forced species update (s)
    real(kind=dp)         :: last_forcing_species_update
    !> forced species update timestep
    real(kind=dp)         :: forcing_species_timestep

    !> output results interval
    real(kind=dp)         :: output_timestep

    logical               :: is_initialized = .false.
  contains
    private
    !> load time control info from input file
    procedure, public :: load

    !> time control initialization=
    procedure, public :: initialize

    !> get current time (s)
    procedure, public :: get_current_time

    !> increment time step and check if the simulation is over
    procedure, public :: increment

    !> check if the output should be written at the current timestep
    procedure, public :: is_output_time

    !> get current time step
    procedure, public :: get_time_step

    !> print information
    procedure, public :: do_print

    !> check if current year is a leap year
    procedure, private :: is_leap_year

    !> Determine the number of bytes required to pack the variable
    procedure, public :: pack_size
    !> Pack the given variable into a buffer, advancing position
    procedure, public :: bin_pack
    !> Unpack the given variable from a buffer, advancing position
    procedure, public :: bin_unpack

  end type time_control_t
  !> Constructor for time_control_t
  interface time_control_t
    procedure :: constructor
    procedure :: copy_constructor
  end interface time_control_t

contains
!> constructor for the file constraint
  function constructor() result(new_obj)
    !> a new time_control_t instance
    type(time_control_t), pointer :: new_obj

    allocate (new_obj)
  end function constructor

  !> copy constructor
  function copy_constructor(other) result(new_obj)
    !> a time_control_t instance to copy
    type(time_control_t), pointer, intent(in) :: other
    !> a new time_control_t instance
    type(time_control_t), pointer :: new_obj

    allocate (new_obj)
    new_obj%property_set => null()
    new_obj%year = other%year
    new_obj%month = other%month
    new_obj%day = other%day
    new_obj%hour = other%hour
    new_obj%previous_hour = other%previous_hour
    new_obj%start_time = other%start_time
    new_obj%end_time = other%end_time
    new_obj%tz = other%tz
    new_obj%lon = other%lon
    new_obj%lat = other%lat
    new_obj%current_time_step_length = other%current_time_step_length
    new_obj%current_time = other%current_time
    new_obj%output_time_step_index = other%output_time_step_index
    new_obj%is_initialized = other%is_initialized
    new_obj%last_photolysis_update = other%last_photolysis_update
    new_obj%photolysis_update_timestep = other%photolysis_update_timestep
    new_obj%last_emissions_update = other%last_emissions_update
    new_obj%emissions_update_timestep = other%emissions_update_timestep
    new_obj%microphysics_update_timestep = other%microphysics_update_timestep
    new_obj%last_forcing_species_update = other%last_forcing_species_update
    new_obj%forcing_species_timestep = other%forcing_species_timestep
    new_obj%last_deposition_update = other%last_deposition_update
    new_obj%deposition_update_timestep = other%deposition_update_timestep

  end function copy_constructor

  !> load time control from input file
#ifdef CAMP_USE_JSON
  subroutine load(this, json, j_obj)
    !> time control data
    class(time_control_t), intent(inout) :: this
    !> JSON core
    type(json_core), pointer, intent(in) :: json
    !> JSON object
    type(json_value), pointer, intent(in) :: j_obj

    type(json_value), pointer :: child, next
    character(kind=json_ck, len=:), allocatable :: key
    character(len=:), allocatable :: owner_name

    ! allocate space for the constraint property set
    this%property_set => property_t()

    ! No names currently for constraints, so use generic label
    owner_name = "time control"

    ! cycle through the time control properties, loading them into the reaction
    ! property set
    next => null()
    call json%get_child(j_obj, child)
    do while (associated(child))
      call json%info(child, name=key)
      call this%property_set%load(json, child, &
        .false., owner_name)

      call json%get_next(child, next)
      child => next
    end do
#else
  subroutine load(this)

    !> constraint data
    class(time_control_t), intent(inout) :: this

    call warn_msg(3631752331, "No support for input files")
#endif
  end subroutine load

  subroutine initialize(this)
    class(time_control_t), intent(inout) :: this
    !> times in seconds
    real(kind=dp):: simulation_length, timestep
    type(property_t), pointer :: subset
    character(len=:), allocatable :: key_name

    call assert_msg(487686867, &
      associated(this%property_set), &
      "Missing property set needed to initialize time control")

    ! get the required info
    key_name = "start_time"
    call assert_msg(917030769, &
      this%property_set%get_property_t(key_name, subset), &
      "Missing 'start_time' needed for time control")

    key_name = "year"
    call assert_msg(291994605, &
      subset%get_int(key_name, this%year), &
      "Missing 'year' in 'start_time', for time control")

    key_name = "month"
    call assert_msg(1406159707, &
      subset%get_int(key_name, this%month), &
      "Missing 'month' in 'start_time', for time control")

    key_name = "day"
    call assert_msg(1406159707, &
      subset%get_int(key_name, this%day), &
      "Missing 'day' in 'start_time', for time control")

    key_name = "hour"
    call assert_msg(1406159707, &
      subset%get_real(key_name, this%hour), &
      "Missing 'hour' in 'start_time', for time control")
    this%start_time = this%hour*3600

    key_name = "tz"
    if (.not. subset%get_real(key_name, this%tz)) then
      this%tz = 0.0
    end if

    deallocate (subset)
    key_name = "simulation_length"
    call assert_msg(1981375939, &
      this%property_set%get_property_t(key_name, subset), &
      "Missing 'simulation_length' needed for time control")
    simulation_length = seconds_from_time_property_set(subset)

    deallocate (subset)
    key_name = "timestep"
    call assert_msg(1273710175, &
      this%property_set%get_property_t(key_name, subset), &
      "Missing 'timestep' needed for time control")
    timestep = seconds_from_time_property_set(subset)

    deallocate (subset)
    key_name = "photolysis_update_timestep"
    if (this%property_set%get_property_t(key_name, subset)) then
      this%photolysis_update_timestep = seconds_from_time_property_set(subset)
    else
      this%photolysis_update_timestep = 15*60 ! by default photolysis update every 15 minutes
    end if

    if (associated(subset)) then
      deallocate (subset)
    end if
    key_name = "emissions_update_timestep"
    if (this%property_set%get_property_t(key_name, subset)) then
      this%emissions_update_timestep = seconds_from_time_property_set(subset)
    else
      this%emissions_update_timestep = 15*60 ! by default emissions update every 15 minutes
    end if

    if (associated(subset)) then
      deallocate (subset)
    end if
    key_name = "microphysics_update_timestep"
    if (this%property_set%get_property_t(key_name, subset)) then
      this%microphysics_update_timestep = seconds_from_time_property_set(subset)
    else
      this%microphysics_update_timestep = 5*60 ! by default microphysics update every 5 minutes
    end if

    if (associated(subset)) then
      deallocate (subset)
    end if
    key_name = "forced_species_update_timestep"
    if (this%property_set%get_property_t(key_name, subset)) then
      this%forcing_species_timestep = seconds_from_time_property_set(subset)
    else
      this%forcing_species_timestep = 60 ! by default microphysics update every minutes
    end if


    if (associated(subset)) then
      deallocate (subset)
    end if
    key_name = "output_timestep"
    if (this%property_set%get_property_t(key_name, subset)) then
      this%output_timestep = seconds_from_time_property_set(subset)
    else
      this%output_timestep = timestep ! by default output file for each physical timestep
    end if

    call assert_msg(155661892, &
      simulation_length > 0. .and. this%start_time >= 0. .and. timestep > 0., &
      "simulation length, start_time and time_step must be positive.")

    call assert_msg(481227742, &
      timestep < simulation_length, &
      "the time step must be smaller than the simulation length")

    this%current_time = 0.
    this%end_time = simulation_length
    this%current_time_step_length = timestep
    this%last_photolysis_update = 0.
    this%last_emissions_update = 0.
    this%last_microphysics_update = 0.
    this%last_forcing_species_update = 0.
    this%last_deposition_update = 0.

    this%is_initialized = .true.

  end subroutine

  !> convert a json property set in the form
  !> \code{.json} {
  !> "duration": 2.0,
  !>  "unit": "min"
  !> }
  !> to seconds
  real function seconds_from_time_property_set(property_set)
    type(property_t), intent(in), pointer:: property_set

    character(len=:), allocatable :: key_name, unit_name
    real(kind=dp) :: val
    key_name = "time"
    call assert_msg(311040954, &
      property_set%get_real(key_name, val), &
      "Missing key 'time' needed for time value")

    key_name = "unit"
    call assert_msg(962707835, &
      property_set%get_string(key_name, unit_name), &
      "Missing key 'unit' needed for time value")

    select case (unit_name)
     case ("sec", "s", "seconds")
      ! nothing to do
     case ("min", "mn", "minutes", "MIN", "MINUTES", "minute")
      val = val*60.0
     case ("h", "hr", "hours", "H", "HOURS", "hour")
      val = val*3600.0
     case ("d", "days", "D", "DAYS", "day")
      val = val*86400.0
     case DEFAULT
      call die_msg(225664810, unit_name//" is an invalid time unit")
    end select

    seconds_from_time_property_set = real(val, kind=sp)

  end function seconds_from_time_property_set

  real(kind=dp) function get_current_time(this)
    class(time_control_t), intent(in) :: this

    get_current_time = this%current_time

  end function get_current_time

  logical function is_leap_year(this)
    class(time_control_t), intent(in) :: this

    if (mod(this%year, 4) == 0) then
      if (mod(this%year, 100) == 0) then
        if (mod(this%year, 400) == 0) then
          is_leap_year = .true.
          return
        else
          is_leap_year = .false.
          return
        end if
      else
        is_leap_year = .true.
        return
      end if
    else
      is_leap_year = .false.
    end if

  end function is_leap_year

  !> function to calculate new time step and increase current time
  !! returns .false. if the end of the simulation has been reached
  logical function increment(this)
    class(time_control_t), intent(inout) :: this
    integer(kind=i_kind), dimension(12) :: actual_months

    if (this%current_time >= this%end_time) then
      increment = .false.
      return
    end if

    if (this%is_output_time()) then
      this%output_time_step_index = this%output_time_step_index + 1
    endif

    ! adjust timestep length for next timestep if needed
    if ((this%current_time + this%current_time_step_length) > this%end_time) then
      this%current_time_step_length = this%end_time - this%current_time
    end if
    this%current_time = this%current_time + this%current_time_step_length
    this%last_photolysis_update = this%last_photolysis_update + this%current_time_step_length
    this%last_emissions_update = this%last_emissions_update + this%current_time_step_length
    this%last_microphysics_update = this%last_microphysics_update + this%current_time_step_length
    this%last_forcing_species_update = this%last_forcing_species_update + this%current_time_step_length
    this%last_deposition_update = this%last_deposition_update + this%current_time_step_length

    ! check if we need to increase the date and hour for sza calculations
    this%previous_hour = this%hour
    this%hour = mod(this%current_time/3600., 24.0)
    ! check if the day needs to advance
    if (this%previous_hour > this%hour) then
      this%day = this%day + 1

      actual_months = months
      if (this%is_leap_year()) then
        actual_months(2) = 29
      end if
      if (this%day > actual_months(this%month)) then
        this%day = 1
        this%month = this%month + 1
        if (this%month > 12) then
          this%month = 1
          this%year = this%year + 1
        end if
      end if
    end if

#ifdef CAMP_DEBUG
    print *, this%year, "-", this%month, "-", this%day, " ", this%hour
#endif
    increment = .true.

  end function increment

  logical function is_output_time(this)
    class(time_control_t), intent(in) :: this
    is_output_time = (mod(this%get_current_time(), this%output_timestep) == 0)
  end function is_output_time

  real(kind=dp) function get_time_step(this)
    class(time_control_t), intent(in) :: this

    get_time_step = this%current_time_step_length

  end function get_time_step

  subroutine do_print(this, file_unit)
    class(time_control_t), intent(in) :: this
    !> file unit for output
    integer(kind=i_kind), optional :: file_unit

    integer(kind=i_kind) :: f_unit

    f_unit = 6

    if (present(file_unit)) f_unit = file_unit
    write (f_unit, *) "*** Time Control ***"
    ! if (associated(this%property_set)) call this%property_set%print(f_unit)
    write (f_unit, *) "start_time=", this%start_time
    write (f_unit, *) "end_time=", this%end_time
    write (f_unit, *) "time_step=", this%current_time_step_length

  end subroutine do_print

  !> Determine the number of bytes required to pack the variable
  integer(kind=i_kind) function pack_size(this, comm)
    class(time_control_t), intent(in) :: this
    !> MPI communicator
    integer, intent(in), optional :: comm

#ifdef CAMP_USE_MPI
    integer(kind=i_kind) :: l_comm
    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if

    call assert_msg(83935016, this%is_initialized, &
      "Trying to get the buffer size of an uninitialized time_control.")

    pack_size = camp_mpi_pack_size_integer(this%year, l_comm) + &
      camp_mpi_pack_size_integer(this%month, l_comm) + &
      camp_mpi_pack_size_integer(this%day, l_comm) + &
      camp_mpi_pack_size_real(this%hour, l_comm) + &
      camp_mpi_pack_size_real(this%start_time, l_comm) + &
      camp_mpi_pack_size_real(this%end_time, l_comm) + &
      camp_mpi_pack_size_real(this%current_time_step_length, l_comm) + &
      camp_mpi_pack_size_real(this%output_timestep, l_comm) + &
      camp_mpi_pack_size_real(this%photolysis_update_timestep, l_comm) + &
      camp_mpi_pack_size_real(this%emissions_update_timestep, l_comm) + &
      camp_mpi_pack_size_real(this%microphysics_update_timestep, l_comm) + &
      camp_mpi_pack_size_real(this%forcing_species_timestep, l_comm) + &
      camp_mpi_pack_size_real(this%output_timestep, l_comm)
#else
    pack_size = 0
#endif

  end function pack_size

  !> Pack the given variable into a buffer, advancing position
  subroutine bin_pack(this, buffer, pos, comm)
    class(time_control_t), intent(in) :: this
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

    call assert_msg(231330219, this%is_initialized, &
      "Trying to pack an uninitialized time_control.")

    prev_position = pos
    call camp_mpi_pack_integer(buffer, pos, this%year, l_comm)
    call camp_mpi_pack_integer(buffer, pos, this%month, l_comm)
    call camp_mpi_pack_integer(buffer, pos, this%day, l_comm)
    call camp_mpi_pack_real(buffer, pos, this%hour, l_comm)
    call camp_mpi_pack_real(buffer, pos, this%start_time, l_comm)
    call camp_mpi_pack_real(buffer, pos, this%end_time, l_comm)
    call camp_mpi_pack_real(buffer, pos, this%current_time_step_length, l_comm)
    call camp_mpi_pack_real(buffer, pos, this%output_timestep, l_comm)
    call camp_mpi_pack_real(buffer, pos, this%photolysis_update_timestep, l_comm)
    call camp_mpi_pack_real(buffer, pos, this%emissions_update_timestep, l_comm)
    call camp_mpi_pack_real(buffer, pos, this%microphysics_update_timestep, l_comm)
    call camp_mpi_pack_real(buffer, pos, this%forcing_species_timestep, l_comm)
    call camp_mpi_pack_real(buffer, pos, this%output_timestep, l_comm)
    call assert(1111237251, &
      pos - prev_position == this%pack_size(l_comm))
#endif

  end subroutine bin_pack

  !> Unpack the given variable from a buffer, advancing position
  subroutine bin_unpack(this, buffer, pos, comm)
    class(time_control_t), intent(inout) :: this
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
    call camp_mpi_unpack_integer(buffer, pos, this%year, l_comm)
    call camp_mpi_unpack_integer(buffer, pos, this%month, l_comm)
    call camp_mpi_unpack_integer(buffer, pos, this%day, l_comm)
    call camp_mpi_unpack_real(buffer, pos, this%hour, l_comm)
    call camp_mpi_unpack_real(buffer, pos, this%start_time, l_comm)
    call camp_mpi_unpack_real(buffer, pos, this%end_time, l_comm)
    call camp_mpi_unpack_real(buffer, pos, this%current_time_step_length, l_comm)
    call camp_mpi_unpack_real(buffer, pos, this%output_timestep, l_comm)
    call camp_mpi_unpack_real(buffer, pos, this%photolysis_update_timestep, l_comm)
    call camp_mpi_unpack_real(buffer, pos, this%emissions_update_timestep, l_comm)
    call camp_mpi_unpack_real(buffer, pos, this%microphysics_update_timestep, l_comm)
    call camp_mpi_unpack_real(buffer, pos, this%forcing_species_timestep, l_comm)
    call camp_mpi_unpack_real(buffer, pos, this%output_timestep, l_comm)

    this%current_time = 0.
    this%last_photolysis_update = 0.
    this%last_emissions_update = 0.
    this%last_microphysics_update = 0.
    this%last_forcing_species_update = 0.
    this%last_deposition_update = 0.

    this%is_initialized = .true.

    call assert(311274588, &
      pos - prev_position == this%pack_size(l_comm))
#endif
  end subroutine bin_unpack

end module boxmodel_time_control
