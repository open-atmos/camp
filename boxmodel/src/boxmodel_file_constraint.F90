!> The file constraint type and associated functions
!! A value constrained with linear interpolation
!! over discrete points read in a file
module boxmodel_file_constraint
  use mpi, only: MPI_COMM_WORLD

  use camp_constants, only: i_kind, dp
  use camp_util, only: assert_msg, open_file_read, close_file, &
    interp_1d, to_string, &
    die_msg, almost_equal
  use camp_mpi

  use boxmodel_constraints, only: constraint_t, FILE_CONSTRAINT
  use boxmodel_montecarlo

  use boxmodel_log

  implicit none
  private

  public :: file_constraint_t
  real(kind=dp) :: zero_dp

  type, extends(constraint_t) :: file_constraint_t
    real(kind=dp), dimension(:), allocatable :: time
    real(kind=dp), dimension(:), allocatable :: value
    integer(kind=i_kind)                     :: n_points
    logical                                  :: diurnal_cycle
  contains

    !> constraint initialization
    procedure :: initializer
    !> get the constraint value at given time
    procedure :: get_at

    !> randomize the constraint based on its montecarlo object
    procedure :: randomize

    !> Determine the number of bytes required to pack the variable
    procedure :: internal_pack_size
    !> Pack the given variable into a buffer, advancing position
    procedure :: internal_bin_pack
    !> Unpack the given variable from a buffer, advancing position
    procedure :: internal_bin_unpack
    !> print information for debugging
    procedure :: internal_print

    procedure :: internal_type_id

  end type file_constraint_t

  !> Constructor for file_constraint_t
  interface file_constraint_t
    procedure :: constructor
  end interface file_constraint_t

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> constructor for the file constraint
  function constructor() result(new_obj)
    !> a new constraint instance
    type(file_constraint_t), pointer :: new_obj

    allocate (new_obj)
  end function constructor

  !> initialize file constraint with provided information
  subroutine initializer(this)
    !> constraint data
    class(file_constraint_t), intent(inout) :: this
    !>input file format info
    character(len=:), allocatable :: filename ! file to read
    character(len=:), allocatable :: format_arg ! format to read the input file
    character(len=:), allocatable :: separator ! separator used in input file
    integer(kind=i_kind) :: header ! header size of the input file (number of lines to ignore)
    character(len=:), allocatable :: current_line ! line read the input file
    integer(kind=i_kind) :: INPUT_FILE_UNIT ! input file unit
    !
    integer :: i_line, io, sep_index, i_time
    character(len=:), allocatable :: key_name
    character(len=1000) :: buffer

    call assert_msg(820361480, &
      associated(this%property_set), &
      "Missing property set needed to initialize constraint")

    ! get the required info
    key_name = "filename"
    call assert_msg(917030769, &
      this%property_set%get_string(key_name, filename), &
      "Missing 'filename' needed for file constraint")

    key_name = "time_unit"
    call assert_msg(105173810, &
      this%property_set%get_string(key_name, this%time_unit), &
      "Missing 'time_unit' needed for file constraint: "//filename)

    key_name = "value_unit"
    call assert_msg(792588190, &
      this%property_set%get_string(key_name, this%value_unit), &
      "Missing 'value_unit' needed for file constraint: "//filename)

    ! get optional info
    key_name = "header"
    if (.not. this%property_set%get_int(key_name, header)) then
      header = 0
    end if

    key_name = "separator"
    if (.not. this%property_set%get_string(key_name, separator)) then
      separator = " "
    end if

    key_name = "diurnal cycle"
    if (.not. this%property_set%get_logical(key_name, this%diurnal_cycle)) then
      this%diurnal_cycle = .TRUE.
    end if

    ! open file
    call open_file_read(filename, INPUT_FILE_UNIT)

    do i_line = 1, header
      read (INPUT_FILE_UNIT, *)
    end do

    ! get number of points and allocate data arrays
    this%n_points = 0
    do
      read (INPUT_FILE_UNIT, *, iostat=io)
      if (io /= 0) exit
      this%n_points = this%n_points + 1
    end do

    call assert_msg(135447180, &
      this%n_points > 1, &
      "File constraint requires at least 2 points: "//filename)

    allocate (this%time(this%n_points))
    allocate (this%value(this%n_points))

    ! read points
    rewind (INPUT_FILE_UNIT)

    do i_line = 1, header
      read (INPUT_FILE_UNIT, '(A)')
    end do

    do i_line = 1, this%n_points
      read (INPUT_FILE_UNIT, '(A)') buffer
      current_line = trim(buffer)

      sep_index = index(current_line, separator)
      call assert_msg(993151218, &
        sep_index > 0, &
        "Missing separator in constraint file:"//filename// &
        " at line "//to_string(i_line + header))

      read (current_line(1:(sep_index - 1)), *) this%time(i_line)
      read (current_line((sep_index + len(separator)):len(current_line)), *) this%value(i_line)
    end do

    ! check that times are sorted
    do i_time = 1, this%n_points - 1
      call assert_msg(316037403, &
        this%time(i_time) < this%time(i_time + 1), &
        "The time column of constraint file '"//filename//"' must be strictly increasing")
    end do

    ! convert time to seconds
    select case (this%time_unit)
     case ("sec", "s", "S", "seconds", "SECONDS")
      ! do nothing
     case ("min", "MIN", "minutes", "MINUTES")
      this%time = this%time*60.
     case ("h", "H", "hr", "hours", "HOURS")
      this%time = this%time*3600.
     case default
      call die_msg(928325480, &
        "Unidentified time_unit for constraint file: "//filename)
    end select

    if (this%diurnal_cycle) then
      call assert_msg(992185678, &
        (almost_equal(this%time(1), zero_dp)) .and. this%time(size(this%time)) == 86400., &
        "Error in constraint file '"//filename//"': "// &
        "to constrain a diurnal cycle, the first time point must be 0. and the last one must be 86400 s (24h)")
    end if
  end subroutine initializer

  !> interpolate constrained value at given time
  real(kind=dp) function get_at(this, time)
    class(file_constraint_t), intent(in) :: this
    real(kind=dp), intent(in) :: time
    real(kind=dp) :: current_time

    zero_dp = 0.0  ! is there NO zero in double precision definition in all CAMP? I do not think so, find it @amelli

    current_time = time
    if (this%diurnal_cycle) then
      current_time = modulo(time, 86400.)
    end if

    call assert_msg(240819925, &
      (almost_equal(this%time(1), zero_dp)) .and. &
      this%time(size(this%time)) >= current_time, &
      "Constraint error: current time:"//trim(to_string(current_time))// &
      " is not encompassed by file constraint")

    get_at = interp_1d(this%time, this%value, current_time)

  end function get_at

  subroutine internal_print(this, f_unit)
    class(file_constraint_t), intent(in) :: this
    integer(kind=i_kind), intent(in), optional :: f_unit
    integer(kind=i_kind) :: i_time




    if (present(f_unit)) then
      write (f_unit, *) "***********************"
      write (f_unit, *) "*** FILE constraint ***"
      write (f_unit, *) "***********************"

      write (f_unit, *) " *** number of points = ", this%n_points
      write (f_unit, *) " *** diurnal cycle = ", this%diurnal_cycle
      write (f_unit, *) "*** points values ****"
      write (f_unit, *) "time, value"
      do i_time = 1, this%n_points
        write (f_unit, *) this%time(i_time), this%value(i_time)
      end do
      write (f_unit, *) "******** end ********"
    else
      call thread_log%debug("***********************")
      call thread_log%debug("*** FILE constraint ***")
      call thread_log%debug("***********************")

      call thread_log%debug("number of points"//to_string(this%n_points))
      call thread_log%debug("diurnal cycle = "//to_string(this%diurnal_cycle))
      call thread_log%debug("*** points values ***")
      call thread_log%debug("time, value")
      do i_time = 1, this%n_points
        call thread_log%debug(to_string(this%time(i_time))//","//to_string(this%value(i_time)))
      end do
      call thread_log%debug("******** end ********")
    endif

  end subroutine internal_print

  subroutine randomize(this)
    class(file_constraint_t), intent(inout) :: this

    if (this%montecarlo_fg) then
      call this%montecarlo%val%randomize_array(this%value)
    else
      call warn_msg(881433698, " trying to randomize a file constraint without a montecarlo object")
    end if

  end subroutine randomize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  integer(kind=i_kind) function internal_pack_size(this, comm)
    !> Reaction update data
    class(file_constraint_t), intent(in) :: this
    !> MPI communicator
    integer, intent(in), optional :: comm

#ifdef CAMP_USE_MPI
    integer :: l_comm

    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if

    internal_pack_size = camp_mpi_pack_size_logical(this%diurnal_cycle, l_comm) + &
      camp_mpi_pack_size_integer(this%n_points, l_comm) + &
      camp_mpi_pack_size_real_array(this%time, l_comm) + &
      camp_mpi_pack_size_real_array(this%value, l_comm)
#else
    internal_pack_size = 0
#endif
  end function internal_pack_size

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine internal_bin_pack(this, buffer, pos, comm)
    class(file_constraint_t), intent(in) :: this
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
    call camp_mpi_pack_logical(buffer, pos, this%diurnal_cycle, l_comm)
    call camp_mpi_pack_integer(buffer, pos, this%n_points, l_comm)
    call camp_mpi_pack_real_array(buffer, pos, this%time, l_comm)
    call camp_mpi_pack_real_array(buffer, pos, this%value, l_comm)

    call assert(37034085, &
      pos - prev_position <= this%pack_size(l_comm))
#endif
  end subroutine internal_bin_pack

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!> Unpack the given variable from a buffer, advancing position
  subroutine internal_bin_unpack(this, buffer, pos, comm)
    class(file_constraint_t), intent(inout) :: this
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
    call camp_mpi_unpack_logical(buffer, pos, this%diurnal_cycle, l_comm)
    call camp_mpi_unpack_integer(buffer, pos, this%n_points, l_comm)
    call camp_mpi_unpack_real_array(buffer, pos, this%time, l_comm)
    call camp_mpi_unpack_real_array(buffer, pos, this%value, l_comm)
    call assert(391493910, &
      pos - prev_position <= this%pack_size(l_comm))
#endif

  end subroutine internal_bin_unpack

  integer(kind=i_kind) function internal_type_id(this)
    class(file_constraint_t), intent(in) :: this

    internal_type_id = FILE_CONSTRAINT

  end function internal_type_id

end module boxmodel_file_constraint
