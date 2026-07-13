!> The constant constraint type and associated functions
!! A simple constraint with a constant value over time
module boxmodel_constant_constraint
  use mpi, only: MPI_COMM_WORLD

  use camp_constants, only: i_kind, dp
  use camp_util, only: assert_msg, to_string
  use camp_mpi

  use boxmodel_constraints, only: constraint_t, CONSTANT_CONSTRAINT
  use boxmodel_montecarlo

  use boxmodel_log

  implicit none
  private

  public :: constant_constraint_t

  type, extends(constraint_t) :: constant_constraint_t
    real(kind=dp) :: value = 0.0
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
    !> print constraint informations
    procedure :: internal_print

    procedure :: internal_type_id

  end type constant_constraint_t

  !> Constructor for constant_constraint_t
  interface constant_constraint_t
    procedure :: constructor
  end interface constant_constraint_t

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> constructor for constant constraint
  function constructor() result(new_obj)
    !> A new constraint instance
    type(constant_constraint_t), pointer :: new_obj

    allocate (new_obj)
  end function constructor

  !> initialize constant constraint with provided information
  subroutine initializer(this)
    !> constraint data
    class(constant_constraint_t), intent(inout) :: this

    character(len=:), allocatable :: key_name

    call assert_msg(768609550, associated(this%property_set), &
      "Missing property set needed to initialize constraint")
    key_name = "value"

    call assert_msg(264110292, &
      this%property_set%get_real(key_name, this%value), &
      "Missing value needed for constant constraint")

    key_name = "unit"
    call assert_msg(425664187, &
      this%property_set%get_string(key_name, this%value_unit), &
      "Missing unit for constant constraint")

    this%time_unit = "unneeded"

  end subroutine initializer

  subroutine internal_print(this, f_unit)
    class(constant_constraint_t), intent(in) :: this
    integer(kind=i_kind), intent(in), optional :: f_unit
    if (present(f_unit)) then
      write (f_unit, *) "***************************"
      write (f_unit, *) "*** CONSTANT constraint ***"
      write (f_unit, *) "***************************"

      write (f_unit, *) " *** constant value = ", this%value
    else
      call thread_log%debug("***************************")
      call thread_log%debug("*** CONSTANT constraint ***")
      call thread_log%debug("***************************")
      call thread_log%debug("constant value="//to_string(this%value))

    endif
  end subroutine internal_print

  subroutine randomize(this)
    class(constant_constraint_t), intent(inout) :: this

    if (this%montecarlo_fg) then
      call this%montecarlo%val%randomize(this%value)
    end if

  end subroutine randomize

  real(kind=dp) function get_at(this, time)
    class(constant_constraint_t), intent(in) :: this
    real(kind=dp), intent(in) :: time

    !> TODO: account for unit here ? or let caller take of it
    ! by checking this%value_unit ?
    get_at = this%value

  end function get_at

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  integer(kind=i_kind) function internal_pack_size(this, comm)
    !> Reaction update data
    class(constant_constraint_t), intent(in) :: this
    !> MPI communicator
    integer, intent(in), optional :: comm

#ifdef CAMP_USE_MPI
    integer :: l_comm

    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if

    internal_pack_size = camp_mpi_pack_size_real(this%value, l_comm)
#else
    internal_pack_size = 0
#endif
  end function internal_pack_size

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine internal_bin_pack(this, buffer, pos, comm)
    class(constant_constraint_t), intent(in) :: this
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
    call camp_mpi_pack_real(buffer, pos, this%value, l_comm)

    call assert(357322251, &
      pos - prev_position <= this%pack_size(l_comm))
#endif
  end subroutine internal_bin_pack

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!> Unpack the given variable from a buffer, advancing position
  subroutine internal_bin_unpack(this, buffer, pos, comm)
    class(constant_constraint_t), intent(inout) :: this
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
    call camp_mpi_unpack_real(buffer, pos, this%value, l_comm)
    call assert(416612926, &
      pos - prev_position <= this%pack_size(l_comm))
#endif

  end subroutine internal_bin_unpack

  integer(kind=i_kind) function internal_type_id(this)
    class(constant_constraint_t), intent(in) :: this

    internal_type_id = CONSTANT_CONSTRAINT

  end function internal_type_id

end module boxmodel_constant_constraint
