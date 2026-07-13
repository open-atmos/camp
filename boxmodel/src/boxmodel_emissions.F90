module boxmodel_emissions
  use mpi, only: MPI_COMM_WORLD

  use camp_mpi
  use boxmodel_constraints_utils, only: constraint_from_type_id
  use camp_constants, only: i_kind
  use camp_util, only: die_msg, assert_msg, string_t
  use camp_rxn_emission, only: rxn_update_data_emission_t
  use camp_property, only: property_t
  use boxmodel_constraints, only: constraint_ptr

  implicit none

  !> map between boxmodel emission constraints and camp mechanism emisssion reactions
  type emissions_map_t
    !> number of emission reaction considered
    integer(kind=i_kind) :: n_emission
    !> array of emitted species names
    type(string_t), dimension(:), allocatable :: emitted_species
    !> array of emitted species state id
    integer(kind=i_kind), dimension(:), allocatable :: emitted_id
    !> indices of matching camp reactions
    integer(kind=i_kind), dimension(:), allocatable :: emission_rxn_ind
    !> current value of emission rates (molec/cm2/s) for exporting to netcdf
    real(kind=dp), dimension(:), allocatable :: current_emissions_rates
    !> the array that takes care of updating the emission reaction rates
    type(rxn_update_data_emission_t), dimension(:), allocatable :: emissions_rates_updates
    class(constraint_ptr), dimension(:), pointer  :: emissions_constraints
  contains
    procedure :: initialize_constraints

    !> Determine the number of bytes required to pack the variable
    procedure, public :: pack_size
    !> Pack the given variable into a buffer, advancing position
    procedure, public :: bin_pack
    !> Unpack the given variable from a buffer, advancing position
    procedure, public :: bin_unpack
  end type emissions_map_t

contains

  subroutine initialize_constraints(this, random_type)
    class(emissions_map_t), intent(inout) :: this

    integer(kind=i_kind), intent(in) :: random_type

    integer(kind=i_kind) :: i_emi

    do i_emi = 1, this%n_emission
      print *, 2400444, " i_emi =", i_emi
      if (.not. associated(this%emissions_constraints(i_emi)%val)) then
        call die_msg(1415186559, "Trying to initialize unassociated emissions constraint ")
      end if
      call this%emissions_constraints(i_emi)%val%initialize(random_type)
    end do

  end subroutine initialize_constraints

  !> Determine the number of bytes required to pack the variable
  integer(kind=i_kind) function pack_size(this, comm)
    class(emissions_map_t), intent(in) :: this
    !> MPI communicator
    integer, intent(in), optional :: comm
    integer :: i_emission, constraint_type_id
#ifdef CAMP_USE_MPI
    integer(kind=i_kind) :: l_comm
    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if
    pack_size = 0
    pack_size = pack_size + camp_mpi_pack_size_integer(this%n_emission, l_comm)

    do i_emission = 1, this%n_emission
      pack_size = pack_size + this%emissions_rates_updates(i_emission)%pack_size()

      constraint_type_id = this%emissions_constraints(i_emission)%val%constraint_type_id()
      pack_size = pack_size + &
        camp_mpi_pack_size_integer(constraint_type_id, l_comm) + &
        this%emissions_constraints(i_emission)%val%pack_size(l_comm)
    end do

    pack_size = pack_size + camp_mpi_pack_size_integer_array(this%emitted_id, l_comm)

#else
    pack_size = 0
#endif
  end function pack_size

  !> Pack the given variable into a buffer, advancing position
  subroutine bin_pack(this, buffer, pos, comm)
    class(emissions_map_t), intent(in) :: this
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos
    !> MPI communicator
    integer, intent(in), optional :: comm

#ifdef CAMP_USE_MPI
    integer(kind=i_kind) :: prev_position, l_comm, i_emission

    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if

    prev_position = pos
    call camp_mpi_pack_integer(buffer, pos, this%n_emission, l_comm)
    do i_emission = 1, this%n_emission
      call this%emissions_rates_updates(i_emission)%bin_pack(buffer, pos, l_comm)

      call camp_mpi_pack_integer(buffer, pos, this%emissions_constraints(i_emission)%val%constraint_type_id(), l_comm)
      call this%emissions_constraints(i_emission)%val%bin_pack(buffer, pos, l_comm)
    end do

    call camp_mpi_pack_integer_array(buffer, pos, this%emitted_id, l_comm)

    call assert(133646395, &
      pos - prev_position == this%pack_size(l_comm))
#endif

  end subroutine bin_pack

  !> Unpack the given variable from a buffer, advancing position
  subroutine bin_unpack(this, buffer, pos, comm)
    class(emissions_map_t), intent(inout) :: this
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos
    !> MPI communicator
    integer, intent(in), optional :: comm

#ifdef CAMP_USE_MPI
    integer(kind=i_kind) :: prev_position, l_comm, i_emission, constraint_type_id

    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if

    prev_position = pos

    call camp_mpi_unpack_integer(buffer, pos, this%n_emission, l_comm)

    allocate(this%emissions_rates_updates(this%n_emission))
    allocate(this%emissions_constraints(this%n_emission))
    allocate(this%current_emissions_rates(this%n_emission))

    do i_emission = 1, this%n_emission
      call this%emissions_rates_updates(i_emission)%bin_unpack(buffer, pos, l_comm)

      call camp_mpi_unpack_integer(buffer, pos, constraint_type_id, l_comm)
      call constraint_from_type_id(this%emissions_constraints(i_emission), constraint_type_id)
      call this%emissions_constraints(i_emission)%val%bin_unpack(buffer, pos, l_comm)
    end do

    call camp_mpi_unpack_integer_array(buffer, pos, this%emitted_id, l_comm)

    call assert(284994225, &
      pos - prev_position == this%pack_size(l_comm))
#endif
  end subroutine bin_unpack

end module boxmodel_emissions
