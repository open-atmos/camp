module boxmodel_constraints

#ifdef CAMP_USE_JSON
  use json_module
#endif

  use camp_property
  use camp_constants
  use camp_util, only: assert_msg
  use camp_mpi
  use boxmodel_time_control, only: time_control_t
  use boxmodel_montecarlo
  use boxmodel_montecarlo_uniform
  use boxmodel_montecarlo_gaussian
  use boxmodel_montecarlo_utils

  use boxmodel_log

  implicit none
  private
  public :: constraint_t, constraint_ptr

  integer, parameter, public :: CONSTANT_CONSTRAINT = 1
  integer, parameter, public :: FILE_CONSTRAINT = 2
  integer, parameter, public :: MONARCH_CONSTRAINT = 3

  type, abstract :: constraint_t
    private

    !> Constraint parameters used to determine how to interpolate
    !! values
    type(property_t), pointer, public :: property_set => null()
    character(len=:), allocatable, public :: value_unit, time_unit
    logical :: is_initialized = .false.
    logical, public :: montecarlo_fg = .false.
    type(montecarlo_ptr), public :: montecarlo

  contains
    !> Wrapper around initializer
    procedure :: initialize
    !> Wrapper around get_at
    procedure :: get => get_val
    !> print constraint properties
    procedure :: print
    procedure(internal_print), deferred :: internal_print
    !> Constraint initialization from info
    procedure(initializer), deferred :: initializer
    !> Load constraint info from input file
    procedure :: load
    !> get constrained value at given time (s)
    procedure(get_at), deferred :: get_at

    !> randomize the constraint based on the montecarlo object
    procedure(randomize), deferred :: randomize

    !> Determine the number of bytes required to pack the variable
    procedure, public :: pack_size
    procedure(internal_pack_size), deferred :: internal_pack_size
    !> Pack the given variable into a buffer, advancing position
    procedure, public :: bin_pack
    procedure(internal_bin_pack), deferred :: internal_bin_pack
    !> Unpack the given variable from a buffer, advancing position
    procedure, public :: bin_unpack
    procedure(internal_bin_unpack), deferred :: internal_bin_unpack

    procedure, public :: constraint_type_id
    procedure(internal_type_id), deferred :: internal_type_id
  end type constraint_t

  !> Pointer type for building arrays of mixed constraints
  type :: constraint_ptr
    class(constraint_t), pointer :: val => null()
  contains
    procedure :: is_initialized
    procedure :: value_unit
    procedure :: get => get_ptr
    !> Dereference the pointer
    procedure :: dereference
    !> Finalize the pointer
    final :: ptr_finalize
  end type constraint_ptr

  interface

    !> Constraint initialization
    subroutine initializer(this)
      import :: constraint_t

      !> constraint data
      class(constraint_t), intent(inout) :: this

    end subroutine initializer

    !> Get constrained value at given time (s)
    real(kind=dp) function get_at(this, time)
      use camp_constants, only: dp
      import :: constraint_t

      !> constraint data
      class(constraint_t), intent(in) :: this
      !> current time (s)
      real(kind=dp), intent(in) :: time

    end function get_at

    !> randomize the constraint based on the given montecarlo object
    subroutine randomize(this)
      import:: constraint_t
      class(constraint_t), intent(inout) :: this
    end subroutine randomize

    !> Determine the number of bytes required to pack the variable
    integer(kind=i_kind) function internal_pack_size(this, comm)
      use camp_constants, only: i_kind
      import :: constraint_t
      class(constraint_t), intent(in) :: this
      !> MPI communicator
      integer, intent(in), optional :: comm

    end function internal_pack_size

    !> Pack the given variable into a buffer, advancing position
    subroutine internal_bin_pack(this, buffer, pos, comm)
      import :: constraint_t
      class(constraint_t), intent(in) :: this
      !> Memory buffer
      character, intent(inout) :: buffer(:)
      !> Current buffer position
      integer, intent(inout) :: pos
      !> MPI communicator
      integer, intent(in), optional :: comm

    end subroutine internal_bin_pack

    !> Unpack the given variable from a buffer, advancing position
    subroutine internal_bin_unpack(this, buffer, pos, comm)
      import :: constraint_t
      class(constraint_t), intent(inout) :: this
      !> Memory buffer
      character, intent(inout) :: buffer(:)
      !> Current buffer position
      integer, intent(inout) :: pos
      !> MPI communicator
      integer, intent(in), optional :: comm

    end subroutine internal_bin_unpack

    integer(kind=i_kind) function internal_type_id(this)
      use camp_constants, only: i_kind
      import :: constraint_t
      class(constraint_t), intent(in) :: this

    end function internal_type_id

    subroutine internal_print(this, f_unit)
      use camp_constants, only: i_kind
      import :: constraint_t
      class(constraint_t), intent(in) :: this
      integer(kind=i_kind), intent(in), optional :: f_unit

    end subroutine internal_print

  end interface

contains
  !> wrapper around get_at
  real(kind=dp) function get_val(this, time)
    class(constraint_t), intent(in) :: this
    real(kind=dp), intent(in) :: time

    call assert_msg(821459810, &
      this%is_initialized, &
      "Trying to use a non initialized constraint")

    get_val = this%get_at(time)
  end function get_val

  real(kind=dp) function get_ptr(this, time)
    class(constraint_ptr), intent(in) :: this
    real(kind=dp), intent(in) :: time

    get_ptr = this%val%get(time)
  end function get_ptr

  function value_unit(this)
    class(constraint_ptr), intent(in) :: this
    character(len=:), allocatable :: value_unit

    value_unit = this%val%value_unit

  end function value_unit

  !> wrapper around initializer
  subroutine initialize(this, random_type)
    class(constraint_t), intent(inout) :: this
    integer(kind=i_kind), intent(in) :: random_type
    character(len=:), allocatable :: key_name, montecarlo_type
    type(property_t), pointer :: montecarlo_subset => null()

    !> find montecarlo object if present
    key_name = "montecarlo"
    if (associated(montecarlo_subset)) then
      montecarlo_subset => null()
    end if

    if (this%property_set%get_property_t(key_name, montecarlo_subset)) then

      key_name = "type"
      call assert_msg(483819285, &
        montecarlo_subset%get_string(key_name, montecarlo_type), &
        "key '"//key_name//"' not provided for montecarlo object")

      select case (montecarlo_type)
       case ("uniform")
        this%montecarlo%val => montecarlo_uniform_t()
       case ("gaussian")
        this%montecarlo%val => montecarlo_gaussian_t()
       case default
        call die_msg(142497094, &
          "unknown montecarlo object type: '"//montecarlo_type//"'")
      end select

      this%montecarlo%val%property_set => montecarlo_subset
      call this%montecarlo%val%initialize(random_type)
      this%montecarlo_fg = .true.
    end if
    call this%initializer()

    this%is_initialized = .true.
  end subroutine initialize

  logical function is_initialized(this)
    class(constraint_ptr), intent(in) :: this

    is_initialized = this%val%is_initialized
  end function is_initialized

  !> load constraints from input file
#ifdef CAMP_USE_JSON
  subroutine load(this, json, j_obj)
    !> constraint data
    class(constraint_t), intent(inout) :: this
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
    owner_name = "constraint"

    ! cycle through the constraint, loading them into the reaction
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
    class(constraint_t), intent(inout) :: this

    call warn_msg(135648416, "No support for input files")
#endif

  end subroutine load

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Dereference a pointer to a constraint
  elemental subroutine dereference(this)

    !> Pointer to a reaction:w
    class(constraint_ptr), intent(inout) :: this

    this%val => null()

  end subroutine dereference

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Finalize a pointer to a constraint
  subroutine ptr_finalize(this)

    !> Pointer to a reaction
    type(constraint_ptr), intent(inout) :: this

    if (associated(this%val)) deallocate (this%val)

  end subroutine ptr_finalize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> Get constraint properties (for use by external modules)
  function get_property_set(this) result(prop_set)
    !> constraint data
    class(constraint_t), intent(in) :: this
    !> constraint properties
    type(property_t), pointer :: prop_set

    prop_set => this%property_set

  end function get_property_set

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> print the constraint properties
  subroutine print(this, file_unit)
    !> constraint data
    class(constraint_t), intent(in) :: this
    !> file unit for output
    integer(kind=i_kind), optional :: file_unit


    if (present(file_unit))  then
      write (file_unit, *) "*** Constraint ***"
      write (file_unit, *) "time_unit=", this%time_unit
      write (file_unit, *) "value_unit=", this%value_unit
      call this%internal_print(file_unit)
    else
      call thread_log%debug("*** Constraint ***")
      call thread_log%debug("time_unit="//this%time_unit)
      call thread_log%debug("value_unit="//this%value_unit)
      call this%internal_print()
    endif

  end subroutine print

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  integer(kind=i_kind) function pack_size(this, comm)
    !> Reaction update data
    class(constraint_t), intent(in) :: this
    !> MPI communicator
    integer, intent(in), optional :: comm

    integer(kind=i_kind) :: montecarlo_type_id

#ifdef CAMP_USE_MPI
    integer :: l_comm

    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if

    pack_size = &
      camp_mpi_pack_size_string(this%value_unit, l_comm) + &
      camp_mpi_pack_size_string(this%time_unit, l_comm) + &
      this%internal_pack_size(l_comm) + &
      camp_mpi_pack_size_logical(this%montecarlo_fg, l_comm)

    if (this%montecarlo_fg) then
      montecarlo_type_id = this%montecarlo%val%montecarlo_type_id()
      pack_size = pack_size + camp_mpi_pack_size_integer(montecarlo_type_id)
      pack_size = pack_size + this%montecarlo%val%pack_size(l_comm)
    end if
#else
    pack_size = 0
#endif
  end function pack_size

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine bin_pack(this, buffer, pos, comm)
    class(constraint_t), intent(in) :: this
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
    call camp_mpi_pack_string(buffer, pos, this%value_unit, l_comm)
    call camp_mpi_pack_string(buffer, pos, this%time_unit, l_comm)

    call this%internal_bin_pack(buffer, pos, l_comm)

    call camp_mpi_pack_logical(buffer, pos, this%montecarlo_fg, l_comm)

    if (this%montecarlo_fg) then
      call camp_mpi_pack_integer(buffer, pos, this%montecarlo%val%montecarlo_type_id(), l_comm)
      call this%montecarlo%val%bin_pack(buffer, pos, l_comm)
    end if

    call assert(357322251, &
      pos - prev_position <= this%pack_size(l_comm))
#endif
  end subroutine bin_pack

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> Unpack the given variable from a buffer, advancing position
  subroutine bin_unpack(this, buffer, pos, comm)
    class(constraint_t), intent(inout) :: this
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos
    !> MPI communicator
    integer, intent(in), optional :: comm
    character(len=100) :: val
    integer(kind=i_kind) :: n, montecarlo_type_id

#ifdef CAMP_USE_MPI
    integer(kind=i_kind) :: prev_position, l_comm

    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if

    prev_position = pos
    val = ''
    call camp_mpi_unpack_string(buffer, pos, val, l_comm)
    this%value_unit = trim(val)

    val = ''
    call camp_mpi_unpack_string(buffer, pos, val, l_comm)
    this%time_unit = trim(val)

    call this%internal_bin_unpack(buffer, pos, l_comm)

    call camp_mpi_unpack_logical(buffer, pos, this%montecarlo_fg, l_comm)

    if (this%montecarlo_fg) then
      call camp_mpi_unpack_integer(buffer, pos, montecarlo_type_id, l_comm)
      call montecarlo_from_typeid(this%montecarlo, montecarlo_type_id)
      call this%montecarlo%val%bin_unpack(buffer, pos, l_comm)
    end if

    call assert_msg(416612925, &
      pos - prev_position <= this%pack_size(l_comm), &
      "expected size: "//trim(to_string(this%pack_size(l_comm)))// &
      ",got size: "//trim(to_string(pos - prev_position)))
#endif

    this%is_initialized = .true.

  end subroutine bin_unpack

!> get the type id associated to the constraint_ptr
  integer(kind=i_kind) function constraint_type_id(this)
    class(constraint_t), intent(in) :: this

    constraint_type_id = this%internal_type_id()

  end function constraint_type_id

  !> allocate constraint from type id
  ! subroutine from_type_id(this, type_id)
  !   class(constraint_ptr), intent(inout) :: this
  !   integer(kind=i_kind), intent(in) :: type_id

  !   call assert_msg(376413702,.not. associated(this%val), &
  !                   "trying to allocate an already allocated constraint")

  !   select case (type_id)
  !   case (CONSTANT_CONSTRAINT)
  !     this%val => constant_constraint_t()

  !   case (FILE_CONSTRAINT)
  !     this%val => file_constraint_t()

  !   case (MONARCH_CONSTRAINT)
  !     this%val => monarch_constraint_t()

  !   case DEFAULT
  !     call die_msg(683213892, "unknown constraint type, need to be assigned a type_id")
  !   end select

  ! end subroutine from_type_id

end module boxmodel_constraints
