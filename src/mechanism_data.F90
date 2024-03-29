! Copyright (C) 2021 Barcelona Supercomputing Center and University of
! Illinois at Urbana-Champaign
! SPDX-License-Identifier: MIT

!> \file
!> The camp_mechanism_data module.

!> \page camp_mechanism CAMP: Chemical Mechanism
!!
!! A mechanism in the \ref index "camp-chem" module is a set of
!! \ref camp_rxn "reactions" that occur in the gas-phase or within one of
!! several \ref camp_aero_phase "aerosol phases" or across an interface
!! between two phases (gas or aerosol). One or several mechanisms may be
!! included in a \ref index "camp-chem" model run.
!!
!! Every mechanism in a \ref index "camp-chem" run will have access to
!! the same set of \ref camp_species "chemical species" and \ref
!! camp_aero_phase "aerosol phases", so phase and species names must be
!! consistent across all concurrently loaded mechanisms. The division of \ref
!! camp_rxn "reactions" into distinct mechanisms permits a host model to
!! specificy which mechanisms should be solved during a call to
!! \c camp_camp_core::camp_core_t::solve().
!!
!! The input format for mechanism data can be found \ref
!! input_format_mechanism "here".

!> The mechanism_data_t structure and associated subroutines.
module camp_mechanism_data

#ifdef CAMP_USE_JSON
  use json_module
#endif
#ifdef CAMP_USE_MPI
  use mpi
#endif
  use camp_aero_rep_data
  use camp_chem_spec_data
  use camp_constants,                  only : i_kind, dp
  use camp_mpi
  use camp_camp_state
  use camp_rxn_data
  use camp_rxn_factory
  use camp_util,                       only : die_msg, string_t

  implicit none
  private

  public :: mechanism_data_t, mechanism_data_ptr

  !> Reallocation increment
  integer(kind=i_kind), parameter :: REALLOC_INC = 50
  !> Fixed module file unit
  integer(kind=i_kind), parameter :: MECH_FILE_UNIT = 16

  !> A chemical mechanism
  !!
  !! Instances of mechanism_data_t represent complete \ref camp_mechanism
  !! chemical mechanism. Multiple mechanisms  may be used during one model run
  !! and will be solved simultaneously.
  type :: mechanism_data_t
    private
    !> Number of reactions
    integer(kind=i_kind) :: num_rxn = 0
    !> Mechanism name
    character(len=:), allocatable :: mech_name
    !> Path and prefix for fixed module output
    character(len=:), allocatable :: fixed_file_prefix
    !> Reactions
    type(rxn_data_ptr), pointer :: rxn_ptr(:) => null()
  contains
    !> Load reactions from an input file
    procedure :: load
    !> Initialize the mechanism
    procedure :: initialize
    !> Get the mechanism name
    procedure :: name => get_name
    !> Get the size of the reaction database
    procedure :: size => get_size
    !> Get a reaction by its index
    procedure :: get_rxn
    !> Determine the number of bytes required to pack the given value
    procedure :: pack_size
    !> Packs the given value into the buffer, advancing position
    procedure :: bin_pack
    !> Unpacks the given value from the buffer, advancing position
    procedure :: bin_unpack
    !> Print the mechanism data
    procedure :: print => do_print
    !> Finalize the mechanism
    final :: finalize

    ! Private functions
    !> Ensure there is enough room in the reaction dataset to add a
    !! specified number of reactions
    procedure, private :: ensure_size
  end type mechanism_data_t

  ! Constructor for mechanism_data_t
  interface mechanism_data_t
    procedure :: constructor
  end interface mechanism_data_t

  !> Pointer type for building arrays
  type mechanism_data_ptr
    type(mechanism_data_t), pointer :: val => null()
  contains
    !> Dereference the pointer
    procedure :: dereference
    !> Finalize the pointer
    final :: ptr_finalize
  end type mechanism_data_ptr

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Constructor for mechanism_data_t
  function constructor(mech_name, init_size) result(new_obj)

    !> Chemical mechanism
    type(mechanism_data_t), pointer :: new_obj
    !> Name of the mechanism
    character(len=*), intent(in), optional :: mech_name
    !> Number of reactions to allocate space for initially
    integer(i_kind), intent(in), optional :: init_size

    integer(i_kind) :: alloc_size

    alloc_size = REALLOC_INC

    allocate(new_obj)
    if (present(init_size)) alloc_size = init_size
    if (present(mech_name)) then
      new_obj%mech_name = mech_name
    else
      new_obj%mech_name = ""
    endif
    allocate(new_obj%rxn_ptr(alloc_size))

  end function constructor

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Ensure there is enough room in the reaction dataset to add a specified
  !! number of reactions
  subroutine ensure_size(this, num_rxn)

    !> Chemical mechanism
    class(mechanism_data_t), intent(inout) :: this
    !> Number of new reactions to ensure space for
    integer(i_kind), intent(in) :: num_rxn

    integer(kind=i_kind) :: new_size
    type(rxn_data_ptr), pointer :: new_rxn_ptr(:)

    if (size(this%rxn_ptr) .ge. this%num_rxn + num_rxn) return
    new_size = this%num_rxn + num_rxn + REALLOC_INC
    allocate(new_rxn_ptr(new_size))
    new_rxn_ptr(1:this%num_rxn) = this%rxn_ptr(1:this%num_rxn)
    call this%rxn_ptr(:)%dereference()
    deallocate(this%rxn_ptr)
    this%rxn_ptr => new_rxn_ptr

  end subroutine ensure_size

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> \page input_format_mechanism Input JSON Object Format: Mechanism
  !!
  !! A \c json object containing information about a \ref camp_mechanism
  !! "chemical mechanism" has the following format :
  !! \code{.json}
  !! { "camp-data" : [
  !!   {
  !!     "name" : "my mechanism",
  !!     "type" : "MECHANISM",
  !!     "reactions" : [
  !!       ...
  !!     ]
  !!   }
  !! ]}
  !! \endcode
  !! A \ref camp_mechanism "mechanism" object must have a unique \b name,
  !! a \b type of \b MECHANISM and an array of \ref input_format_rxn
  !! "reaction objects" labelled \b reactions. Mechanism data may be split
  !! into multiple mechanism objects across input files - they will be
  !! combined based on the mechanism name.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Load a chemical mechanism from an input file
#ifdef CAMP_USE_JSON
  subroutine load(this, json, j_obj)

    !> Chemical mechanism
    class(mechanism_data_t), intent(inout) :: this
    !> JSON core
    type(json_core), pointer, intent(in) :: json
    !> JSON object
    type(json_value), pointer, intent(in) :: j_obj

    type(json_value), pointer :: child, next
    character(kind=json_ck, len=:), allocatable :: unicode_str_val
    type(rxn_factory_t) :: rxn_factory
    logical :: found

    ! Cycle through the set of reactions in the json file
    next => null()

    ! Get the reaction set
    call json%get(j_obj, 'reactions(1)', child, found)
    do while (associated(child) .and. found)

      ! Increase the size of the mechanism
      call this%ensure_size(1)
      this%num_rxn = this%num_rxn + 1

      ! Load the reaction into the mechanism
      this%rxn_ptr(this%num_rxn)%val => rxn_factory%load(json, child)

      ! Get the next reaction in the json file
      call json%get_next(child, next)
      child => next
    end do

    ! Determine whether and where to build fixed module code
    call json%get(j_obj, 'build fixed module', unicode_str_val, found)
    if (found) then
      call assert_msg(410823202, .not.allocated(this%fixed_file_prefix), &
              "Received multiple file prefixes for fixed mechanism module.")
      this%fixed_file_prefix = trim(unicode_str_val)
    end if

#else
  subroutine load(this)

    !> Chemical mechanism
    class(mechanism_data_t), intent(inout) :: this

    call warn_msg(384838139, "No support for input files")
#endif

  end subroutine load

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Initialize the mechanism
  subroutine initialize(this, chem_spec_data, aero_rep_data, n_cells)

    !> Chemical mechanism
    class(mechanism_data_t), intent(inout) :: this
    !> Chemical species data
    type(chem_spec_data_t), intent(in) :: chem_spec_data
    !> Aerosol representation data
    type(aero_rep_data_ptr), pointer, intent(in) :: aero_rep_data(:)
    !> Number of grid cells to solve simultaneously
    integer(kind=i_kind), intent(in) :: n_cells

    integer(kind=i_kind) :: i_rxn

    do i_rxn = 1, this%num_rxn
      call assert(340397127, associated(this%rxn_ptr(i_rxn)%val))
      call this%rxn_ptr(i_rxn)%val%initialize(chem_spec_data, aero_rep_data, &
                                              n_cells)
    end do

  end subroutine initialize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Get the current size of the chemical mechanism
  integer(kind=i_kind) function get_size(this)

    !> Chemical mechanism
    class(mechanism_data_t), intent(in) :: this

    get_size = this%num_rxn

  end function get_size

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Get a reaction by its index
  function get_rxn(this, rxn_id) result (rxn_ptr)

    !> Pointer to the reaction
    class(rxn_data_t), pointer :: rxn_ptr
    !> Mechanism data
    class(mechanism_data_t), intent(in) :: this
    !> Reaction index
    integer(kind=i_kind), intent(in) :: rxn_id

    call assert_msg(129484547, rxn_id.gt.0 .and. rxn_id .le. this%num_rxn, &
            "Invalid reaction id: "//trim(to_string(rxn_id))//&
            "exptected a value between 1 and "//trim(to_string(this%num_rxn)))

    rxn_ptr => this%rxn_ptr(rxn_id)%val

  end function get_rxn

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Get the name of the mechanism
  function get_name(this) result(mech_name)

    !> Name of the mechanism
    character(len=:), allocatable :: mech_name
    !> Chemical mechanism
    class(mechanism_data_t), intent(in) :: this

    mech_name = this%mech_name

  end function get_name

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Determine the size of a binary required to pack the mechanism
  integer(kind=i_kind) function pack_size(this, comm)

    !> Chemical mechanism
    class(mechanism_data_t), intent(in) :: this
    !> MPI communicator
    integer, intent(in) :: comm

    type(rxn_factory_t) :: rxn_factory
    integer(kind=i_kind) :: i_rxn

    pack_size =  camp_mpi_pack_size_integer(this%num_rxn, comm)
    do i_rxn = 1, this%num_rxn
      pack_size = pack_size + rxn_factory%pack_size(this%rxn_ptr(i_rxn)%val, &
                                                    comm)
    end do

  end function pack_size

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Pack the given value to the buffer, advancing position
  subroutine bin_pack(this, buffer, pos, comm)

    !> Chemical mechanism
    class(mechanism_data_t), intent(in) :: this
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos
    !> MPI communicator
    integer, intent(in) :: comm

#ifdef CAMP_USE_MPI
    type(rxn_factory_t) :: rxn_factory
    integer :: i_rxn, prev_position

    prev_position = pos
    call camp_mpi_pack_integer(buffer, pos, this%num_rxn, comm)
    do i_rxn = 1, this%num_rxn
      associate (rxn => this%rxn_ptr(i_rxn)%val)
      call rxn_factory%bin_pack(rxn, buffer, pos, comm)
      end associate
    end do
    call assert(669506045, &
         pos - prev_position <= this%pack_size(comm))
#endif

  end subroutine bin_pack

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Unpack the given value to the buffer, advancing position
  subroutine bin_unpack(this, buffer, pos, comm)

    !> Chemical mechanism
    class(mechanism_data_t), intent(inout) :: this
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos
    !> MPI communicator
    integer, intent(in) :: comm

#ifdef CAMP_USE_MPI
    type(rxn_factory_t) :: rxn_factory
    integer :: i_rxn, prev_position, num_rxn

    prev_position = pos
    call camp_mpi_unpack_integer(buffer, pos, num_rxn, comm)
    call this%ensure_size(num_rxn)
    this%num_rxn = num_rxn
    do i_rxn = 1, this%num_rxn
      this%rxn_ptr(i_rxn)%val => rxn_factory%bin_unpack(buffer, pos, comm)
    end do
    call assert(360900030, &
         pos - prev_position <= this%pack_size(comm))
#endif

  end subroutine bin_unpack

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Print the mechanism data
  subroutine do_print(this, file_unit)

    !> Chemical mechanism
    class(mechanism_data_t), intent(in) :: this
    !> File unit for output
    integer(kind=i_kind), optional :: file_unit

    integer :: i_rxn
    integer(kind=i_kind) :: f_unit

    f_unit = 6

    if (present(file_unit)) f_unit = file_unit

    write(f_unit,*) "Mechanism: "//trim(this%name())
    do i_rxn = 1, this%num_rxn
      call this%rxn_ptr(i_rxn)%val%print(f_unit)
    end do
    write(f_unit,*) "End mechanism: "//trim(this%name())

  end subroutine do_print

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Finalize the mechanism
  elemental subroutine finalize(this)

    !> Mechanism data
    type(mechanism_data_t), intent(inout) :: this

    if (allocated(this%mech_name))         deallocate(this%mech_name)
    if (allocated(this%fixed_file_prefix)) deallocate(this%fixed_file_prefix)
    if (associated(this%rxn_ptr))          deallocate(this%rxn_ptr)

  end subroutine finalize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Dereference a pointer to a mechanism
  elemental subroutine dereference(this)

    !> Pointer to the mechanism
    class(mechanism_data_ptr), intent(inout) :: this

    this%val => null()

  end subroutine dereference

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Finalize a pointer to mechanism data
  elemental subroutine ptr_finalize(this)

    !> Pointer to mechanism data
    type(mechanism_data_ptr), intent(inout) :: this

    if (associated(this%val)) deallocate(this%val)

  end subroutine ptr_finalize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end module camp_mechanism_data
