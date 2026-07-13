module boxmodel_sum_species
  use mpi, only: MPI_COMM_WORLD

  use camp_chem_spec_data
  use camp_util, only: assert_msg, die_msg
  use camp_camp_state
  use camp_constants, only: i_kind
  use camp_mpi

  implicit none

  public :: sum_species_t

  type :: sum_species_t
    integer :: species_idx
    integer :: n_components
    integer, dimension(:), allocatable :: components_idx

  contains
    !> initialize
    procedure :: update_sum_concentrations

    !> Determine the number of bytes required to pack the variable
    procedure, public :: pack_size
    !> Pack the given variable into a buffer, advancing position
    procedure, public :: bin_pack
    !> Unpack the given variable from a buffer, advancing position
    procedure, public :: bin_unpack

  end type sum_species_t
  !> CAMP <-> boxmodel interface constructor
  interface sum_species_t
    procedure :: constructor
  end interface sum_species_t

contains

  function constructor(chem_spec_data, spec_name, components_names, phase) result(this)
    type(sum_species_t), pointer :: this

    type(chem_spec_data_t), pointer, intent(in) :: chem_spec_data
    character(len=*), intent(in) :: spec_name
    character(len=*), intent(in), dimension(:) :: components_names
    character(len=*), intent(in) :: phase

    integer :: i_comp

    allocate (this)

    select case (phase)
    case ("GAS")
      this%species_idx = chem_spec_data%gas_state_id(spec_name)

      this%n_components = len(components_names)

      if (this%n_components > 0) then
        allocate (this%components_idx(this%n_components))

        do i_comp = 1, this%n_components
          this%components_idx(i_comp) = chem_spec_data%gas_state_id(components_names(i_comp))
          call assert_msg(330765791, &
                          this%components_idx(i_comp) > 0, &
                          "component "//components_names(i_comp)//" of sum species "//spec_name//" not found.")
        end do

      end if

    case ("AEROSOL")
      ! \todo find a way to deal with sum species and aerosol representations
      call die_msg(89984978, &
                   "Cannot build aerosol phase sum species"//spec_name)
    end select

  end function constructor

  subroutine update_sum_concentrations(this, camp_state)
    class(sum_species_t), intent(in) :: this

    !> CAMP-chem state
    type(camp_state_t), pointer, intent(inout) :: camp_state

    integer :: i_comp

    camp_state%state_var(this%species_idx) = 0.0
    if (this%n_components == 0) return

    do i_comp = 1, this%n_components
      camp_state%state_var(this%species_idx) = camp_state%state_var(this%species_idx) + &
                                               camp_state%state_var(this%components_idx(i_comp))
    end do

  end subroutine update_sum_concentrations

  !> Determine the number of bytes required to pack the variable
  integer(kind=i_kind) function pack_size(this, comm)
    class(sum_species_t), intent(in) :: this
    !> MPI communicator
    integer, intent(in), optional :: comm

#ifdef CAMP_USE_MPI
    integer(kind=i_kind) :: l_comm
    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if
    pack_size = 0

    pack_size = pack_size + camp_mpi_pack_size_integer(this%species_idx, l_comm) + &
                camp_mpi_pack_size_integer(this%n_components, l_comm) + &
                camp_mpi_pack_size_integer_array(this%components_idx, l_comm)

#else
    pack_size = 0
#endif
  end function pack_size

  !> Pack the given variable into a buffer, advancing position
  subroutine bin_pack(this, buffer, pos, comm)
    class(sum_species_t), intent(in) :: this
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
    call camp_mpi_pack_integer(buffer, pos, this%species_idx, l_comm)
    call camp_mpi_pack_integer(buffer, pos, this%n_components, l_comm)
    call camp_mpi_pack_integer_array(buffer, pos, this%components_idx, l_comm)

    call assert(109979676, &
                pos - prev_position == this%pack_size(l_comm))
#endif

  end subroutine bin_pack

  !> Unpack the given variable from a buffer, advancing position
  subroutine bin_unpack(this, buffer, pos, comm)
    class(sum_species_t), intent(inout) :: this
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

    call camp_mpi_unpack_integer(buffer, pos, this%species_idx, l_comm)
    call camp_mpi_unpack_integer(buffer, pos, this%n_components, l_comm)
    call camp_mpi_unpack_integer_array(buffer, pos, this%components_idx, l_comm)

    call assert(529867398, &
                pos - prev_position == this%pack_size(l_comm))
#endif
  end subroutine bin_unpack

end module boxmodel_sum_species
