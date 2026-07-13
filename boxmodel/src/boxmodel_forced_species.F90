module boxmodel_forced_species

  use camp_constants, only: i_kind
  use camp_mpi

  use boxmodel_constraints, only: constraint_ptr
  use boxmodel_constraints_utils, only: constraint_from_type_id

  implicit none

  !> map between boxmodel forced species constraints and camp state vector indices
  type forced_species_map_t
    !> number of forced species
    integer(kind=i_kind) :: n_species
    !> indices of species in the state vector
    integer(kind=i_kind), dimension(:), allocatable :: forced_species_ind
    !> the constraints themselves
    class(constraint_ptr), dimension(:), allocatable :: species_constraints
    !> flags to identify aerosol species (required to handle units)
    logical, dimension(:), allocatable :: is_aerosol
  contains
    !> initialize constraints
    procedure :: initialize_constraints

    !> Determine the number of bytes required to pack the variable
    procedure, public :: pack_size
    !> Pack the given variable into a buffer, advancing position
    procedure, public :: bin_pack
    !> Unpack the given variable from a buffer, advancing position
    procedure, public :: bin_unpack

    !> print debugging information
    procedure, public :: print

  end type forced_species_map_t

contains
  subroutine initialize_constraints(this, random_type)
    class(forced_species_map_t), intent(inout) :: this
    integer(kind=i_kind), intent(in) :: random_type

    integer(kind=i_kind) :: i_species

    do i_species = 1, this%n_species
      print *, 337119031, " i_species =", i_species
      if (.not. associated(this%species_constraints(i_species)%val)) then
        call die_msg(228655363, "Trying to initialize unassociated species constraint ")
      end if
      call this%species_constraints(i_species)%val%initialize(random_type)
    end do

  end subroutine initialize_constraints

  !> print debugging information
  subroutine print(this, file_unit)
    class(forced_species_map_t), intent(in) :: this
    integer(kind=i_kind), intent(in), optional :: file_unit

    integer(kind=i_kind) :: f_unit, i_spec

    f_unit = 6

    if (present(file_unit)) then
      f_unit = file_unit
    end if

    write (f_unit, *) " ************************** "
    write (f_unit, *) " *** forced species map *** "
    write (f_unit, *) " ************************** "

    write (f_unit, *) "  *** number of species : ", this%n_species

    do i_spec = 1, this%n_species
      write (f_unit, *) "  state vector index =", this%forced_species_ind(i_spec)
      write (f_unit, *) "  is_aerosol = ", this%is_aerosol(i_spec)
      write (f_unit, *) "  constraint:"
      call this%species_constraints(i_spec)%val%print()
    end do

  end subroutine print

  !> Determine the number of bytes required to pack the variable
  integer(kind=i_kind) function pack_size(this, comm)
    class(forced_species_map_t), intent(in) :: this
    !> MPI communicator
    integer, intent(in), optional :: comm
    integer :: i_species, constraint_type_id
#ifdef CAMP_USE_MPI
    integer(kind=i_kind) :: l_comm
    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if
    pack_size = 0
    pack_size = pack_size + camp_mpi_pack_size_integer(this%n_species, l_comm)
    pack_size = pack_size + camp_mpi_pack_size_integer_array(this%forced_species_ind, l_comm)
    pack_size = pack_size + camp_mpi_pack_size_logical_array(this%is_aerosol, l_comm)

    do i_species = 1, this%n_species

      constraint_type_id = this%species_constraints(i_species)%val%constraint_type_id()
      pack_size = pack_size + &
        camp_mpi_pack_size_integer(constraint_type_id, l_comm) + &
        this%species_constraints(i_species)%val%pack_size(l_comm)
    end do

#else
    pack_size = 0
#endif
  end function pack_size

  !> Pack the given variable into a buffer, advancing position
  subroutine bin_pack(this, buffer, pos, comm)
    class(forced_species_map_t), intent(in) :: this
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos
    !> MPI communicator
    integer, intent(in), optional :: comm

#ifdef CAMP_USE_MPI
    integer(kind=i_kind) :: prev_position, l_comm, i_species

    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if

    prev_position = pos
    call camp_mpi_pack_integer(buffer, pos, this%n_species, l_comm)

    call camp_mpi_pack_integer_array(buffer, pos, this%forced_species_ind, l_comm)
    call camp_mpi_pack_logical_array(buffer, pos, this%is_aerosol, l_comm)

    do i_species = 1, this%n_species

      call camp_mpi_pack_integer(buffer, pos, this%species_constraints(i_species)%val%constraint_type_id(), l_comm)
      call this%species_constraints(i_species)%val%bin_pack(buffer, pos, l_comm)
    end do

    call assert(225826514, &
      pos - prev_position == this%pack_size(l_comm))
#endif

  end subroutine bin_pack

  !> Unpack the given variable from a buffer, advancing position
  subroutine bin_unpack(this, buffer, pos, comm)
    class(forced_species_map_t), intent(inout) :: this
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos
    !> MPI communicator
    integer, intent(in), optional :: comm

#ifdef CAMP_USE_MPI
    integer(kind=i_kind) :: prev_position, l_comm, i_species, constraint_type_id

    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if

    prev_position = pos

    call camp_mpi_unpack_integer(buffer, pos, this%n_species, l_comm)
    call camp_mpi_unpack_integer_array(buffer, pos, this%forced_species_ind, l_comm)
    call camp_mpi_unpack_logical_array(buffer, pos, this%is_aerosol, l_comm)

    allocate (this%species_constraints(this%n_species))

    do i_species = 1, this%n_species

      call camp_mpi_unpack_integer(buffer, pos, constraint_type_id, l_comm)
      call constraint_from_type_id(this%species_constraints(i_species), constraint_type_id)
      call this%species_constraints(i_species)%val%bin_unpack(buffer, pos, l_comm)
    end do

    call assert(225826515, &
      pos - prev_position == this%pack_size(l_comm))
#endif
  end subroutine bin_unpack

end module boxmodel_forced_species
