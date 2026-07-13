
!> This module contains what is needed to control aerosol and water microphysics in the boxmodel
!> - update mean diameters and sigma of modal distributions
module boxmodel_microphysics
  use mpi, only: MPI_COMM_WORLD

  use camp_mpi
  use camp_constants, only: i_kind, dp
  use camp_aero_rep_modal_binned_mass, only: aero_rep_update_data_modal_binned_mass_GMD_t, &
    aero_rep_update_data_modal_binned_mass_GSD_t
  use boxmodel_constraints, only: constraint_ptr
  use boxmodel_constraints_utils, only: constraint_from_type_id

  implicit none

  !> map between microphysics constraint and update_data
  type microphysics_map_t
    !> number of distributions to update
    integer(kind=i_kind) :: n_distrib

    !> update_data objects ( one for each aerosol representation,
    !> the section to update is chosen on the call to the update function)
    type(aero_rep_update_data_modal_binned_mass_GMD_t) :: update_data_GMD
    type(aero_rep_update_data_modal_binned_mass_GSD_t) :: update_data_GSD

    !> array containing the indices of the sections to update
    integer(kind=i_kind), dimension(:), allocatable :: section_ids
    !> array containing the constraint objects
    class(constraint_ptr), dimension(:), allocatable  :: diameter_constraints, stdev_constraints
  contains
    procedure :: initialize_constraints

    !> print informations for debugging
    procedure :: print

    !> Determine the number of bytes required to pack the variable
    procedure, public :: pack_size
    !> Pack the given variable into a buffer, advancing position
    procedure, public :: bin_pack
    !> Unpack the given variable from a buffer, advancing position
    procedure, public :: bin_unpack
  end type microphysics_map_t

contains

  subroutine initialize_constraints(this, random_type)
    class(microphysics_map_t), intent(inout) :: this
    integer(kind=i_kind), intent(in) :: random_type

    integer(kind=i_kind) :: i_distrib

    do i_distrib = 1, this%n_distrib
      print *, 1875623, " i_distrib =", i_distrib
      if (.not. associated(this%diameter_constraints(i_distrib)%val)) then
        call die_msg(245962056, "Trying to initialize unassociated diameter constraint ")
      end if
      if (.not. associated(this%stdev_constraints(i_distrib)%val)) then
        call die_msg(245962057, "Trying to initialize unassociated stdev constraint ")
      end if
      call this%diameter_constraints(i_distrib)%val%initialize(random_type)
      call this%stdev_constraints(i_distrib)%val%initialize(random_type)
    end do

  end subroutine initialize_constraints

  !> Determine the number of bytes required to pack the variable
  integer(kind=i_kind) function pack_size(this, comm)
    class(microphysics_map_t), intent(in) :: this
    !> MPI communicator
    integer, intent(in), optional :: comm
    integer :: i_distrib, constraint_type_id
#ifdef CAMP_USE_MPI
    integer(kind=i_kind) :: l_comm
    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if
    pack_size = 0
    pack_size = pack_size + camp_mpi_pack_size_integer(this%n_distrib, l_comm)

    pack_size = pack_size + this%update_data_GMD%pack_size()
    pack_size = pack_size + this%update_data_GSD%pack_size()

    pack_size = pack_size + camp_mpi_pack_size_integer_array(this%section_ids, l_comm)

    do i_distrib = 1, this%n_distrib

      constraint_type_id = this%diameter_constraints(i_distrib)%val%constraint_type_id()
      pack_size = pack_size + &
        camp_mpi_pack_size_integer(constraint_type_id, l_comm) + &
        this%diameter_constraints(i_distrib)%val%pack_size(l_comm)

      constraint_type_id = this%stdev_constraints(i_distrib)%val%constraint_type_id()
      pack_size = pack_size + &
        camp_mpi_pack_size_integer(constraint_type_id, l_comm) + &
        this%stdev_constraints(i_distrib)%val%pack_size(l_comm)
    end do

#else
    pack_size = 0
#endif
  end function pack_size

  !> Pack the given variable into a buffer, advancing position
  subroutine bin_pack(this, buffer, pos, comm)
    class(microphysics_map_t), intent(in) :: this
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos
    !> MPI communicator
    integer, intent(in), optional :: comm

#ifdef CAMP_USE_MPI
    integer(kind=i_kind) :: prev_position, l_comm, i_distrib

    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if

    prev_position = pos
    call camp_mpi_pack_integer(buffer, pos, this%n_distrib, l_comm)

    call this%update_data_GMD%bin_pack(buffer, pos, l_comm)
    call this%update_data_GSD%bin_pack(buffer, pos, l_comm)

    call camp_mpi_pack_integer_array(buffer, pos, this%section_ids, l_comm)

    do i_distrib = 1, this%n_distrib

      call camp_mpi_pack_integer(buffer, pos, this%diameter_constraints(i_distrib)%val%constraint_type_id(), l_comm)
      call this%diameter_constraints(i_distrib)%val%bin_pack(buffer, pos, l_comm)

      call camp_mpi_pack_integer(buffer, pos, this%stdev_constraints(i_distrib)%val%constraint_type_id(), l_comm)
      call this%stdev_constraints(i_distrib)%val%bin_pack(buffer, pos, l_comm)

    end do

    call assert(423463611, &
      pos - prev_position == this%pack_size(l_comm))
#endif

  end subroutine bin_pack

  !> Unpack the given variable from a buffer, advancing position
  subroutine bin_unpack(this, buffer, pos, comm)
    class(microphysics_map_t), intent(inout) :: this
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos
    !> MPI communicator
    integer, intent(in), optional :: comm

#ifdef CAMP_USE_MPI
    integer(kind=i_kind) :: prev_position, l_comm, i_distrib, constraint_type_id

    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if

    prev_position = pos

    call camp_mpi_unpack_integer(buffer, pos, this%n_distrib, l_comm)

    call this%update_data_GMD%bin_unpack(buffer, pos, l_comm)
    call this%update_data_GSD%bin_unpack(buffer, pos, l_comm)

    call camp_mpi_unpack_integer_array(buffer, pos, this%section_ids, l_comm)

    allocate (this%diameter_constraints(this%n_distrib))
    allocate (this%stdev_constraints(this%n_distrib))

    do i_distrib = 1, this%n_distrib

      call camp_mpi_unpack_integer(buffer, pos, constraint_type_id, l_comm)
      call constraint_from_type_id(this%diameter_constraints(i_distrib), constraint_type_id)
      call this%diameter_constraints(i_distrib)%val%bin_unpack(buffer, pos, l_comm)

      call camp_mpi_unpack_integer(buffer, pos, constraint_type_id, l_comm)
      call constraint_from_type_id(this%stdev_constraints(i_distrib), constraint_type_id)
      call this%stdev_constraints(i_distrib)%val%bin_unpack(buffer, pos, l_comm)
    end do

    call assert(258945752, &
      pos - prev_position == this%pack_size(l_comm))
#endif
  end subroutine bin_unpack

  subroutine print(this, file_unit)
    class(microphysics_map_t), intent(in) :: this
    !> File unit for output
    integer(kind=i_kind), intent(in), optional :: file_unit
    integer(kind=i_kind) :: f_unit, i_distrib

    f_unit = 6
    if (present(file_unit)) f_unit = file_unit
    write (f_unit, *) "***************************"
    write (f_unit, *) "** Microphysics map data **"
    write (f_unit, *) "***************************"

    write (f_unit, *) "*** number of modes to updates: ", this%n_distrib
    do i_distrib = 1, this%n_distrib
      write (f_unit, *) "*** diameter constraints #", i_distrib
      call this%diameter_constraints(i_distrib)%val%print(f_unit)
      write (f_unit, *) "*** stdev constraints #", i_distrib
      call this%stdev_constraints(i_distrib)%val%print(f_unit)

    end do
  end subroutine print

end module boxmodel_microphysics
