module boxmodel_deposition
  use mpi, only: MPI_COMM_WORLD

  use camp_mpi
  use camp_rxn_first_order_loss, only : rxn_update_data_first_order_loss_t

  implicit none

  type deposition_map_t
    !> number of deposition reactions considered
    integer(kind=i_kind) :: n_deposition
    !> array of deposited species names
    type(string_t), dimension(:), allocatable :: deposited_species
    !> indices of matching camp reactions
    integer(kind=i_kind), dimension(:), allocatable :: deposition_rxn_ind
    !> the array that takes care of updating the emission reaction rates
    type(rxn_update_data_first_order_loss_t), dimension(:), allocatable :: deposition_rates_updates

  contains
  
    !> Determine the number of bytes required to pack the variable
    procedure, public :: pack_size
    !> Pack the given variable into a buffer, advancing position
    procedure, public :: bin_pack
    !> Unpack the given variable from a buffer, advancing position
    procedure, public :: bin_unpack

  end type deposition_map_t

contains

!> Determine the number of bytes required to pack the variable
  integer(kind=i_kind) function pack_size(this, comm)
    class(deposition_map_t), intent(in) :: this
    !> MPI communicator
    integer, intent(in), optional :: comm
    integer :: i_deposition
#ifdef CAMP_USE_MPI
    integer(kind=i_kind) :: l_comm
    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if
    pack_size = 0
    pack_size = pack_size + camp_mpi_pack_size_integer(this%n_deposition, l_comm)

    do i_deposition = 1, this%n_deposition
      pack_size = pack_size + this%deposition_rates_updates(i_deposition)%pack_size()
    end do

#else
    pack_size = 0
#endif
  end function pack_size


    !> Pack the given variable into a buffer, advancing position
  subroutine bin_pack(this, buffer, pos, comm)
    class(deposition_map_t), intent(in) :: this
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos
    !> MPI communicator
    integer, intent(in), optional :: comm

#ifdef CAMP_USE_MPI
    integer(kind=i_kind) :: prev_position, l_comm, i_deposition

    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if

    prev_position = pos
    call camp_mpi_pack_integer(buffer, pos, this%n_deposition, l_comm)
    do i_deposition = 1, this%n_deposition
      call this%deposition_rates_updates(i_deposition)%bin_pack(buffer, pos, l_comm)
    end do

    call assert(421398495, &
                pos - prev_position == this%pack_size(l_comm))
#endif

  end subroutine bin_pack


  !> Unpack the given variable from a buffer, advancing position
  subroutine bin_unpack(this, buffer, pos, comm)
    class(deposition_map_t), intent(inout) :: this
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos
    !> MPI communicator
    integer, intent(in), optional :: comm

#ifdef CAMP_USE_MPI
    integer(kind=i_kind) :: prev_position, l_comm, i_deposition

    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if

    prev_position = pos

    call camp_mpi_unpack_integer(buffer, pos, this%n_deposition, l_comm)

    allocate(this%deposition_rates_updates(this%n_deposition))

    do i_deposition = 1, this%n_deposition
      call this%deposition_rates_updates(i_deposition)%bin_unpack(buffer, pos, l_comm)
    end do

    call assert(191232193, &
                pos - prev_position == this%pack_size(l_comm))
#endif
  end subroutine bin_unpack

end module boxmodel_deposition