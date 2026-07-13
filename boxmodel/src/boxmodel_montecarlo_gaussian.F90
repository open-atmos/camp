module boxmodel_montecarlo_gaussian
  use camp_mpi
  use camp_rand, only: camp_random
  use camp_constants, only: dp, i_kind, const

  use boxmodel_sobol_numbers, only: next_sobol_number
  use boxmodel_montecarlo
  implicit none
  private

  public :: montecarlo_gaussian_t

  type, extends(montecarlo_t) :: montecarlo_gaussian_t
    real(kind=dp) :: stdev

  contains
    procedure :: internal_initialize
    procedure :: internal_randomize
    procedure :: internal_pack_size
    procedure :: internal_bin_pack
    procedure :: internal_bin_unpack
    procedure :: internal_type_id
  end type montecarlo_gaussian_t

  !> Constructor for montecarlo_gaussian_t
  interface montecarlo_gaussian_t
    procedure :: constructor
  end interface montecarlo_gaussian_t
contains

  !> constructor for montecarlo_uniform_t
  function constructor() result(new_obj)
    !> A new montecarlo instance
    type(montecarlo_gaussian_t), pointer :: new_obj

    allocate (new_obj)
  end function constructor

  subroutine internal_initialize(this)
    class(montecarlo_gaussian_t), intent(inout) :: this
    character(len=:), allocatable :: key_name

    key_name = "standard_deviation"

    call assert_msg(310312902, &
      this%property_set%get_real(key_name, this%stdev), &
      "Could not find key '"//key_name//"' for gaussian montecarlo ")

    if (this%random_type == SOBOL) then
      N_SOBOL_MONTECARLO = N_SOBOL_MONTECARLO + 2 ! sampling gaussian distributions requires 2 random numbers
    endif
  end subroutine internal_initialize

  !> sample from gaussian distribution based on two random numbers
  function normal_from_rand(mean, stddev, u1, u2)
    real(kind=dp), intent(in) :: mean, stddev, u1, u2
    real(kind=dp) :: normal_from_rand
    real(kind=dp) :: r, theta, z0, z1
! Uses the Box-Muller transform
    ! http://en.wikipedia.org/wiki/Box-Muller_transform
    r = sqrt(-2d0 * log(u1))
    theta = 2d0 * const%pi * u2
    z0 = r * cos(theta)
    z1 = r * sin(theta)
    ! z0 and z1 are now independent N(0,1) random variables
    ! We throw away z1, but we could use a SAVE variable to only do
    ! the computation on every second call of this function.
    normal_from_rand = stddev * z0 + mean
  end function normal_from_rand

  !> sample gaussian distribution
  subroutine internal_randomize(this, value)
    class(montecarlo_gaussian_t), intent(in) :: this
    real(kind=dp), intent(inout) :: value

    if (this%random_type == RANDOM) then
      value = normal_from_rand(value, this%stdev, camp_random(), camp_random())
    else if (this%random_type == SOBOL) then
      value = normal_from_rand(value, this%stdev, next_sobol_number(), next_sobol_number())
    endif

  end subroutine internal_randomize

  integer(kind=i_kind) function internal_type_id(this)
    use camp_constants, only: i_kind
    class(montecarlo_gaussian_t), intent(in) :: this

    internal_type_id = MONTECARLO_GAUSSIAN
  end function internal_type_id

  !> Determine the number of bytes required to pack the variable
  integer(kind=i_kind) function internal_pack_size(this, comm)
    class(montecarlo_gaussian_t), intent(in) :: this
    !> MPI communicator
    integer, intent(in), optional :: comm

    internal_pack_size = camp_mpi_pack_size_real(this%stdev, comm)

  end function internal_pack_size

!> Pack the given variable into a buffer, advancing position
  subroutine internal_bin_pack(this, buffer, pos, comm)
    class(montecarlo_gaussian_t), intent(in) :: this
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos
    !> MPI communicator
    integer, intent(in), optional :: comm

    call camp_mpi_pack_real(buffer, pos, this%stdev, comm)

  end subroutine internal_bin_pack

!> Unpack the given variable from a buffer, advancing position
  subroutine internal_bin_unpack(this, buffer, pos, comm)
    class(montecarlo_gaussian_t), intent(inout) :: this
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos
    !> MPI communicator
    integer, intent(in), optional :: comm

    call camp_mpi_unpack_real(buffer, pos, this%stdev, comm)

  end subroutine internal_bin_unpack

end module boxmodel_montecarlo_gaussian
