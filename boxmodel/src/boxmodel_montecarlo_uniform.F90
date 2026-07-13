module boxmodel_montecarlo_uniform
  use camp_mpi
  use camp_rand, only: camp_random
  use camp_constants, only: dp, i_kind

  use boxmodel_sobol_numbers, only: next_sobol_number
  use boxmodel_montecarlo
  implicit none
  private

  public :: montecarlo_uniform_t

  type, extends(montecarlo_t) :: montecarlo_uniform_t
    real(kind=dp) :: relative_factor

  contains
    procedure :: internal_initialize
    procedure :: internal_randomize
    procedure :: internal_pack_size
    procedure :: internal_bin_pack
    procedure :: internal_bin_unpack
    procedure :: internal_type_id
  end type montecarlo_uniform_t

  !> Constructor for montecarlo_uniform_t
  interface montecarlo_uniform_t
    procedure :: constructor
  end interface montecarlo_uniform_t
contains

  !> constructor for montecarlo_uniform_t
  function constructor() result(new_obj)
    !> A new montecarlo instance
    type(montecarlo_uniform_t), pointer :: new_obj

    allocate (new_obj)
  end function constructor

  subroutine internal_initialize(this)
    class(montecarlo_uniform_t), intent(inout) :: this
    character(len=:), allocatable :: key_name

    key_name = "relative_factor"

    call assert_msg(1297744147, &
      this%property_set%get_real(key_name, this%relative_factor), &
      "Could not find key '"//key_name//"' for uniform montecarlo ")

    if (this%random_type == SOBOL) then
      N_SOBOL_MONTECARLO = N_SOBOL_MONTECARLO + 1
    endif

  end subroutine internal_initialize

  subroutine internal_randomize(this, value)
    class(montecarlo_uniform_t), intent(in) :: this
    real(kind=dp), intent(inout) :: value

    real(kind=dp) :: rnd, m, p, min_val, max_val, log_val

    ! produce a uniformly distributed random value between value / f and f * value
    if (this%random_type == SOBOL) then
      rnd = next_sobol_number()
    else if (this%random_type == RANDOM) then
      rnd = camp_random()
    endif
    ! m = (this%relative_factor * this%relative_factor - 1.0) / this%relative_factor
    ! p = 1.0 / this%relative_factor

    min_val = value
    max_val = value * this%relative_factor

    if (this%log_scale_fg) then
      min_val = log10(min_val)
      max_val = log10(max_val)
      log_val = rnd * (max_val - min_val) + min_val
      value = 10**(log_val)
    else
      value = (rnd * (max_val - min_val) + min_val)
    endif


  end subroutine internal_randomize

  integer(kind=i_kind) function internal_type_id(this)
    use camp_constants, only: i_kind
    class(montecarlo_uniform_t), intent(in) :: this

    internal_type_id = MONTECARLO_UNIFORM
  end function internal_type_id

  !> Determine the number of bytes required to pack the variable
  integer(kind=i_kind) function internal_pack_size(this, comm)
    class(montecarlo_uniform_t), intent(in) :: this
    !> MPI communicator
    integer, intent(in), optional :: comm

    internal_pack_size = camp_mpi_pack_size_real(this%relative_factor, comm)

  end function internal_pack_size

!> Pack the given variable into a buffer, advancing position
  subroutine internal_bin_pack(this, buffer, pos, comm)
    class(montecarlo_uniform_t), intent(in) :: this
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos
    !> MPI communicator
    integer, intent(in), optional :: comm

    call camp_mpi_pack_real(buffer, pos, this%relative_factor, comm)

  end subroutine internal_bin_pack

!> Unpack the given variable from a buffer, advancing position
  subroutine internal_bin_unpack(this, buffer, pos, comm)
    class(montecarlo_uniform_t), intent(inout) :: this
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos
    !> MPI communicator
    integer, intent(in), optional :: comm

    call camp_mpi_unpack_real(buffer, pos, this%relative_factor, comm)

  end subroutine internal_bin_unpack

end module boxmodel_montecarlo_uniform
