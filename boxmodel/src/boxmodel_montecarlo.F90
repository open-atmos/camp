module boxmodel_montecarlo
  use mpi, only: MPI_COMM_WORLD

  use camp_property
  use camp_constants, only: dp, i_kind, small_dp, large_dp
  use camp_mpi
  implicit none
  private

  public :: montecarlo_t, montecarlo_ptr

  integer(kind=i_kind), parameter, public :: MONTECARLO_UNIFORM = 1, MONTECARLO_GAUSSIAN = 2
  integer(kind=i_kind), public :: N_SOBOL_MONTECARLO = 0

  !> type of random number generation
  integer(kind=i_kind), parameter, public :: RANDOM = 0, SOBOL = 1

  type, abstract :: montecarlo_t
    private
    type(property_t), pointer, public :: property_set => null()

    !> lower bound
    real(kind=dp) :: lower_bound = small_dp
    !> upper bound
    real(kind=dp) :: upper_bound = large_dp
    !> randomize over log scale
    logical, public :: log_scale_fg = .FALSE.
    !> what kind of random generator to use
    integer(kind=i_kind), public :: random_type

    logical :: is_initialized
  contains

    !> initialize from a property_set
    procedure, public :: initialize
    procedure(internal_initialize), deferred :: internal_initialize

    !> randomize the given real value within the range defined by the relative_factor
    !> lower and upper bound are optionnal to define absolute limits (e.g. force positive values)
    procedure, public :: randomize
    procedure(internal_randomize), deferred :: internal_randomize

    !> bias the given array by a constant random factor defined by the relative_factor
    !> lower and upper bound are optionnal to define absolute limits (e.g. force positive values)
    procedure :: randomize_array

    !> Determine the number of bytes required to pack the variable
    procedure, public :: pack_size
    procedure(internal_pack_size), deferred :: internal_pack_size
    !> Pack the given variable into a buffer, advancing position
    procedure, public :: bin_pack
    procedure(internal_bin_pack), deferred :: internal_bin_pack
    !> Unpack the given variable from a buffer, advancing position
    procedure, public :: bin_unpack
    procedure(internal_bin_unpack), deferred :: internal_bin_unpack

    procedure, public :: montecarlo_type_id
    procedure(internal_type_id), deferred :: internal_type_id
  end type montecarlo_t

  !> Pointer type for building arrays of mixed constraints
  type :: montecarlo_ptr
    class(montecarlo_t), pointer :: val => null()
  contains
    procedure :: is_initialized
    !> Dereference the pointer
    procedure :: dereference
    !> Finalize the pointer
    final :: ptr_finalize
  end type montecarlo_ptr
  interface
    subroutine internal_initialize(this)
      import :: montecarlo_t
      class(montecarlo_t), intent(inout) :: this

    end subroutine internal_initialize

    !> randomize around the given value
    subroutine internal_randomize(this, value)
      use camp_constants, only: dp
      import :: montecarlo_t
      class(montecarlo_t), intent(in) :: this
      real(kind=dp), intent(inout) :: value

    end subroutine internal_randomize

    !> Determine the number of bytes required to pack the variable
    integer(kind=i_kind) function internal_pack_size(this, comm)
      use camp_constants, only: i_kind
      import :: montecarlo_t
      class(montecarlo_t), intent(in) :: this
      !> MPI communicator
      integer, intent(in), optional :: comm

    end function internal_pack_size

    !> Pack the given variable into a buffer, advancing position
    subroutine internal_bin_pack(this, buffer, pos, comm)
      import :: montecarlo_t
      class(montecarlo_t), intent(in) :: this
      !> Memory buffer
      character, intent(inout) :: buffer(:)
      !> Current buffer position
      integer, intent(inout) :: pos
      !> MPI communicator
      integer, intent(in), optional :: comm

    end subroutine internal_bin_pack

    !> Unpack the given variable from a buffer, advancing position
    subroutine internal_bin_unpack(this, buffer, pos, comm)
      import :: montecarlo_t
      class(montecarlo_t), intent(inout) :: this
      !> Memory buffer
      character, intent(inout) :: buffer(:)
      !> Current buffer position
      integer, intent(inout) :: pos
      !> MPI communicator
      integer, intent(in), optional :: comm

    end subroutine internal_bin_unpack

    integer(kind=i_kind) function internal_type_id(this)
      use camp_constants, only: i_kind
      import :: montecarlo_t
      class(montecarlo_t), intent(in) :: this

    end function internal_type_id
  end interface

contains

  !> initialize from a property_set
  subroutine initialize(this, random_type, comm)
    class(montecarlo_t), intent(inout) :: this
    integer(kind=i_kind), intent(in)   :: random_type

    character(len=:), allocatable :: key_name, random_text

    !> MPI communicator
    integer, intent(in), optional :: comm



#ifdef CAMP_USE_MPI
    integer(kind=i_kind) :: l_comm
    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if
#endif

    key_name = "lower_bound"
    if (.not. this%property_set%get_real(key_name, this%lower_bound)) then
      this%lower_bound = small_dp
    end if

    key_name = "upper_bound"
    if (.not. this%property_set%get_real(key_name, this%upper_bound)) then
      this%upper_bound = large_dp
    end if

    key_name = "log_scale"
    if (.not. this%property_set%get_logical(key_name, this%log_scale_fg)) then
      this%log_scale_fg = .FALSE.
    end if

    key_name = "random_type"
    if (.not. this%property_set%get_string(key_name, random_text)) then
      this%random_type = random_type
    else
      select case (random_text)
       case ("RANDOM", "random", "rand")
        this%random_type = RANDOM
       case("SOBOL", "sobol")
        this%random_type = SOBOL
       case DEFAULT
        call die_msg(45102664, random_text//" is an random number generator type, choose among 'random', 'sobol',")
      end select
    endif

    call this%internal_initialize()
    this%is_initialized = .TRUE.

  end subroutine initialize

  !> randomize the given real value within the range defined by the relative_factor
  !> the random value will fall within the range [value * (1 - this%relative_factor), value * (1 + this%relative_factor)]
  !> lower and upper bound are optionnal to define absolute limits (e.g. force positive values)
  subroutine randomize(this, value)
    class(montecarlo_t), intent(in) :: this
    real(kind=dp), intent(inout) :: value

    call this%internal_randomize(value)

    if (value < this%lower_bound) then
      value = this%lower_bound
    end if

    if (value > this%upper_bound) then
      value = this%upper_bound
    end if

  end subroutine randomize

  !> bias the given array by a constant random factor defined by the montecarlo object
  subroutine randomize_array(this, values)
    class(montecarlo_t), intent(in) :: this
    real(kind=dp), intent(inout), dimension(:) :: values

    real(kind=dp) :: random_relative_bias, value
    integer(kind=i_kind) :: i, i_nonzero

    if (all(values == 0.)) then
      return
    end if

    ! find first non zero value
    do i = lbound(values, 1), ubound(values, 1)
      if (values(i) .ne. 0.0) then
        i_nonzero = i
        exit
      end if
    end do

    value = values(i_nonzero)
    call this%internal_randomize(value)

    random_relative_bias = value/values(i_nonzero)

    values = values*random_relative_bias

    where (values < this%lower_bound)
      values = this%lower_bound
    end where

    where (values > this%upper_bound)
      values = this%upper_bound
    end where

  end subroutine randomize_array

!> get the type id associated to the montecarlo object
  integer(kind=i_kind) function montecarlo_type_id(this)
    class(montecarlo_t), intent(in) :: this

    montecarlo_type_id = this%internal_type_id()

  end function montecarlo_type_id

!> Determine the number of bytes required to pack the variable
  integer(kind=i_kind) function pack_size(this, comm)
    class(montecarlo_t), intent(in) :: this
    !> MPI communicator
    integer, intent(in), optional :: comm

#ifdef CAMP_USE_MPI
    integer(kind=i_kind) :: l_comm
    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if

    pack_size = camp_mpi_pack_size_real(this%lower_bound, l_comm) + &
      camp_mpi_pack_size_real(this%upper_bound, l_comm) + &
      camp_mpi_pack_size_logical(this%log_scale_fg) + &
      camp_mpi_pack_size_logical(this%is_initialized, l_comm) + &
      camp_mpi_pack_size_integer(this%random_type, l_comm)

    pack_size = pack_size + this%internal_pack_size(l_comm)

#else
    pack_size = 0
#endif

  end function pack_size

  !> Pack the given variable into a buffer, advancing position
  subroutine bin_pack(this, buffer, pos, comm)
    class(montecarlo_t), intent(in) :: this
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
    call camp_mpi_pack_real(buffer, pos, this%lower_bound, l_comm)
    call camp_mpi_pack_real(buffer, pos, this%upper_bound, l_comm)
    call camp_mpi_pack_logical(buffer, pos, this%log_scale_fg, l_comm)
    call camp_mpi_pack_logical(buffer, pos, this%is_initialized, l_comm)
    call camp_mpi_pack_integer(buffer, pos, this%random_type, l_comm)

    call this%internal_bin_pack(buffer, pos, l_comm)

    call assert(1111237251, &
      pos - prev_position == this%pack_size(l_comm))
#endif

  end subroutine bin_pack

  !> Unpack the given variable from a buffer, advancing position
  subroutine bin_unpack(this, buffer, pos, comm)
    class(montecarlo_t), intent(inout) :: this
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
    call camp_mpi_unpack_real(buffer, pos, this%lower_bound, l_comm)
    call camp_mpi_unpack_real(buffer, pos, this%upper_bound, l_comm)
    call camp_mpi_unpack_logical(buffer, pos, this%log_scale_fg, l_comm)
    call camp_mpi_unpack_logical(buffer, pos, this%is_initialized, l_comm)
    call camp_mpi_unpack_integer(buffer, pos, this%random_type, l_comm)

    call this%internal_bin_unpack(buffer, pos, l_comm)

    call assert(163463116, &
      pos - prev_position == this%pack_size(l_comm))
#endif

  end subroutine bin_unpack

  logical function is_initialized(this)
    class(montecarlo_ptr), intent(in) :: this

    is_initialized = this%val%is_initialized
  end function is_initialized

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Dereference a pointer to a montecarlo_t
  elemental subroutine dereference(this)

    !> Pointer to a reaction:w
    class(montecarlo_ptr), intent(inout) :: this

    this%val => null()

  end subroutine dereference

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!> Finalize a pointer to a montecarlo_t
  subroutine ptr_finalize(this)

    !> Pointer to a reaction
    type(montecarlo_ptr), intent(inout) :: this

    if (associated(this%val)) deallocate (this%val)

  end subroutine ptr_finalize
end module boxmodel_montecarlo
