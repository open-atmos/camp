! Copyright (C) 2021 Barcelona Supercomputing Center and University of
! Illinois at Urbana-Champaign
! SPDX-License-Identifier: MIT

!> \file
!> The camp_rxn_emission module.

!> \page camp_rxn_emission CAMP: Emission
!!
!! Emission reactions take the form:
!!
!! \f[
!!   \rightarrow \mbox{X}
!! \f]
!!
!! where \f$\ce{X}\f$ is the species being emitted.
!!
!! Emission rates can be constant or set from an external module using the
!! \c camp_rxn_emission::rxn_update_data_emission_t object.
!! External modules can use the
!! \c camp_rxn_emission::rxn_emission_t::get_property_set()
!! function during initilialization to access any needed reaction parameters
!! to identify certain emission reactions.
!! An \c camp_rxn_emission::update_data_emission_t object should be
!! initialized for each emissions reaction. These objects can then be used
!! during solving to update the emission rate from an external module.
!!
!! Input data for emission reactions have the following format :
!! \code{.json}
!!   {
!!     "type" : "EMISSION",
!!     "species" : "species_name",
!!     "scaling factor" : 1.2,
!!     ...
!!   }
!! \endcode
!! The key-value pair \b species is required and its value must be the name
!! of the species being emitted. The \b scaling
!! \b factor is optional, and can be used to set a constant scaling
!! factor for the rate. When a \b scaling \b factor is not provided, it is
!! assumed to be 1.0. All other data is optional and will be available to
!! external modules during initialization. Rates are in units of
!! \f$\mbox{concentration units} \quad s^{-1}\f$, and must be set using a
!! \c rxn_update_data_emission_t object.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!> The rxn_emission_t type and associated functions.
module camp_rxn_emission

  use camp_aero_rep_data
  use camp_chem_spec_data
  use camp_constants,                        only: const
  use camp_camp_state
  use camp_mpi
  use camp_property
  use camp_rxn_data
  use camp_util,                             only: i_kind, dp, to_string, &
                                                  assert_msg

  use iso_c_binding

  implicit none
  private

#define RXN_ID_ this%condensed_data_int(1)
#define SPECIES_ this%condensed_data_int(2)
#define DERIV_ID_ this%condensed_data_int(3)
#define SCALING_ this%condensed_data_real(1)
#define NUM_INT_PROP_ 3
#define NUM_REAL_PROP_ 1
#define NUM_ENV_PARAM_ 2

public :: rxn_emission_t, rxn_update_data_emission_t

  !> Generic test reaction data type
  type, extends(rxn_data_t) :: rxn_emission_t
  contains
    !> Reaction initialization
    procedure :: initialize
    !> Get the reaction property set
    procedure :: get_property_set
    !> Initialize update data
    procedure :: update_data_initialize
    !> Finalize the reaction
    final :: finalize, finalize_array
  end type rxn_emission_t

  !> Constructor for rxn_emission_t
  interface rxn_emission_t
    procedure :: constructor
  end interface rxn_emission_t

  !> Emission rate update object
  type, extends(rxn_update_data_t) :: rxn_update_data_emission_t
  private
    !> Flag indicating whether the update data as been allocated
    logical :: is_malloced = .false.
    !> Unique id for finding reactions during model initialization
    integer(kind=i_kind) :: rxn_unique_id = 0
  contains
    !> Update the rate data
    procedure :: set_rate => update_data_rate_set
    !> Determine the pack size of the local update data
    procedure :: internal_pack_size
    !> Pack the local update data to a binary
    procedure :: internal_bin_pack
    !> Unpack the local update data from a binary
    procedure :: internal_bin_unpack
    !> Finalize the rate update data
    final :: update_data_finalize, update_data_finalize_array
  end type rxn_update_data_emission_t

  !> Interface to c reaction functions
  interface

    !> Allocate space for a rate update
    function rxn_emission_create_rate_update_data() &
              result (update_data) bind (c)
      use iso_c_binding
      !> Allocated update_data object
      type(c_ptr) :: update_data
    end function rxn_emission_create_rate_update_data

    !> Set a new emission rate
    subroutine rxn_emission_set_rate_update_data(update_data, &
              rxn_unique_id, base_rate) bind (c)
      use iso_c_binding
      !> Update data
      type(c_ptr), value :: update_data
      !> Reaction unique id
      integer(kind=c_int), value :: rxn_unique_id
      !> New pre-scaling base emission rate
      real(kind=c_double), value :: base_rate
    end subroutine rxn_emission_set_rate_update_data

    !> Free an update rate data object
    pure subroutine rxn_free_update_data(update_data) bind (c)
      use iso_c_binding
      !> Update data
      type(c_ptr), value, intent(in) :: update_data
    end subroutine rxn_free_update_data

  end interface

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Constructor for Emission reaction
  function constructor() result(new_obj)

    !> A new reaction instance
    type(rxn_emission_t), pointer :: new_obj

    allocate(new_obj)
    new_obj%rxn_phase = GAS_RXN

  end function constructor

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Initialize the reaction data, validating component data and loading
  !! any required information into the condensed data arrays for use during
  !! solving
  subroutine initialize(this, chem_spec_data, aero_rep, n_cells)

    !> Reaction data
    class(rxn_emission_t), intent(inout) :: this
    !> Chemical species data
    type(chem_spec_data_t), intent(in) :: chem_spec_data
    !> Aerosol representations
    type(aero_rep_data_ptr), pointer, intent(in) :: aero_rep(:)
    !> Number of grid cells to solve simultaneously
    integer(kind=i_kind), intent(in) :: n_cells

    type(property_t), pointer :: spec_props
    character(len=:), allocatable :: key_name, spec_name
    integer(kind=i_kind) :: i_spec, i_qty

    integer(kind=i_kind) :: temp_int
    real(kind=dp) :: temp_real

    ! Get the species involved
    call assert_msg(135066145, associated(this%property_set), &
            "Missing property set needed to initialize reaction")
    key_name = "species"
    call assert_msg(247384490, &
            this%property_set%get_string(key_name, spec_name), &
            "Emission reaction is missing species name")

    ! Allocate space in the condensed data arrays
    allocate(this%condensed_data_int(NUM_INT_PROP_))
    allocate(this%condensed_data_real(NUM_REAL_PROP_))
    this%condensed_data_int(:) = int(0, kind=i_kind)
    this%condensed_data_real(:) = real(0.0, kind=dp)

    ! Save space for the environment-dependent parameters
    this%num_env_params = NUM_ENV_PARAM_

    ! Get reaction parameters
    key_name = "scaling factor"
    if (.not. this%property_set%get_real(key_name, SCALING_)) then
      SCALING_ = real(1.0, kind=dp)
    end if

    ! Save the index of this species in the state variable array
    SPECIES_ = chem_spec_data%gas_state_id(spec_name)

    ! Make sure the species exists
    call assert_msg(814240522, SPECIES_.gt.0, &
            "Missing emission species: "//spec_name)

    ! Initialize the rxn id
    RXN_ID_ = -1

  end subroutine initialize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Get the reaction properties. (For use by external modules.)
  function get_property_set(this) result(prop_set)

    !> Reaction properties
    type(property_t), pointer :: prop_set
    !> Reaction data
    class(rxn_emission_t), intent(in) :: this

    prop_set => this%property_set

  end function get_property_set

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Finalize the reaction
  subroutine finalize(this)

    !> Reaction data
    type(rxn_emission_t), intent(inout) :: this

    if (associated(this%property_set)) &
            deallocate(this%property_set)
    if (allocated(this%condensed_data_real)) &
            deallocate(this%condensed_data_real)
    if (allocated(this%condensed_data_int)) &
            deallocate(this%condensed_data_int)

  end subroutine finalize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Finalize an array of reactions
  subroutine finalize_array(this)

    !> Array of reactions
    type(rxn_emission_t), intent(inout) :: this(:)

    integer(kind=i_kind) :: i

    do i = 1, size(this)
      call finalize(this(i))
    end do

  end subroutine finalize_array

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Set packed update data for emission rate constants
  subroutine update_data_rate_set(this, base_rate)

    !> Update data
    class(rxn_update_data_emission_t), intent(inout) :: this
    !> Updated pre-scaling emission rate
    real(kind=dp), intent(in) :: base_rate

    call rxn_emission_set_rate_update_data(this%get_data(), &
            this%rxn_unique_id, base_rate)

  end subroutine update_data_rate_set

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Initialize update data
  subroutine update_data_initialize(this, update_data, rxn_type)

    use camp_rand,                                only : generate_int_id

    !> The reaction to be udpated
    class(rxn_emission_t), intent(inout) :: this
    !> Update data object
    class(rxn_update_data_emission_t), intent(out) :: update_data
    !> Reaction type id
    integer(kind=i_kind), intent(in) :: rxn_type

    ! If a reaction id has not yet been generated, do it now
    if (RXN_ID_.eq.-1) then
      RXN_ID_ = generate_int_id()
    endif

    update_data%rxn_unique_id = RXN_ID_
    update_data%rxn_type = int(rxn_type, kind=c_int)
    update_data%update_data = rxn_emission_create_rate_update_data()
    update_data%is_malloced = .true.

  end subroutine update_data_initialize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Determine the size of a binary required to pack the reaction data
  integer(kind=i_kind) function internal_pack_size(this, comm) &
      result(pack_size)

    !> Reaction update data
    class(rxn_update_data_emission_t), intent(in) :: this
    !> MPI communicator
    integer, intent(in) :: comm

    pack_size = &
      camp_mpi_pack_size_logical(this%is_malloced, comm) + &
      camp_mpi_pack_size_integer(this%rxn_unique_id, comm)

  end function internal_pack_size

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Pack the given value to the buffer, advancing position
  subroutine internal_bin_pack(this, buffer, pos, comm)

    !> Reaction update data
    class(rxn_update_data_emission_t), intent(in) :: this
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos
    !> MPI communicator
    integer, intent(in) :: comm

#ifdef CAMP_USE_MPI
    integer :: prev_position

    prev_position = pos
    call camp_mpi_pack_logical(buffer, pos, this%is_malloced, comm)
    call camp_mpi_pack_integer(buffer, pos, this%rxn_unique_id, comm)
    call assert(945453741, &
         pos - prev_position <= this%pack_size(comm))
#endif

  end subroutine internal_bin_pack

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Unpack the given value from the buffer, advancing position
  subroutine internal_bin_unpack(this, buffer, pos, comm)

    !> Reaction update data
    class(rxn_update_data_emission_t), intent(inout) :: this
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos
    !> MPI communicator
    integer, intent(in) :: comm

#ifdef CAMP_USE_MPI
    integer :: prev_position

    prev_position = pos
    call camp_mpi_unpack_logical(buffer, pos, this%is_malloced, comm)
    call camp_mpi_unpack_integer(buffer, pos, this%rxn_unique_id, comm)
    call assert(775296837, &
         pos - prev_position <= this%pack_size(comm))
    this%update_data = rxn_emission_create_rate_update_data()
#endif

  end subroutine internal_bin_unpack

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Finalize an update data object
  subroutine update_data_finalize(this)

    !> Update data object to free
    type(rxn_update_data_emission_t), intent(inout) :: this

    if (this%is_malloced) call rxn_free_update_data(this%update_data)

  end subroutine update_data_finalize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Finalize an array of update data objects
  subroutine update_data_finalize_array(this)

    !> Array of update data objects to free
    type(rxn_update_data_emission_t), intent(inout) :: this(:)

    integer(kind=i_kind) :: i

    do i = 1, size(this)
      call update_data_finalize(this(i))
    end do

  end subroutine update_data_finalize_array

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end module camp_rxn_emission
