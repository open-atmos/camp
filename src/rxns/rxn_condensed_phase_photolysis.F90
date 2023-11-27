! Copyright (C) 2021 Barcelona Supercomputing Center and University of
! Illinois at Urbana-Champaign
! SPDX-License-Identifier: MIT

!> \file
!> The camp_rxn_condensed_phase_photolysis module.

!> \page camp_rxn_condensed_phase_photolysis CAMP: Condensed-Phase Photolysis Reaction
!!
!! Condensed-Phase Photolysis reactions take the form:
!!
!! \f[\ce{
!!   X + h $\nu$ -> Y_1 ( + Y_2 \dots )
!! }\f]
!!
!! where \f$\ce{X}\f$ is the species being photolyzed, and
!! \f$\ce{Y_n}\f$ are the photolysis products.
!!
!! The reaction rate can be scaled by providing the "scaling factor" keyword in the json configuration.
!! 
!!
!! Input data for condensed-phase Photolysis reactions have the following
!! format:
!! \code{.json}
!!   {
!!     "type" : "CONDENSED_PHASE_PHOTOLYSIS",
!!     "rate" : 123.45,
!!     "units" : "M",
!!     "aerosol phase" : "my aqueous phase",
!!     "aerosol-phase water" : "H2O_aq",
!!     "reactants" : {
!!       "spec1" : {},
!!       ...
!!     },
!!     "products" : {
!!       "spec3" : {},
!!       "spec4" : { "yield" : 0.65 },
!!       ...
!!     },
!!     "scaling factor": 11.20
!!   }
!! \endcode
!! The key-value pairs \b reactants, and \b products are required. Reactants
!! without a \b qty value are assumed to appear once in the reaction equation.
!! Products without a specified \b yield are assumed to have a \b yield of
!! 1.0.
!!
!! Units for the reactants and products must be specified using the key
!! \b units and can be either \b M or \b mol \b m-3. If units of \b M are
!! specified, a key-value pair \b aerosol-phase \b water must also be included
!! whose value is a string specifying the name for water in the aerosol phase.
!!
!! The unit for time is assumed to be s, but inclusion of the optional
!! key-value pair \b time \b unit = \b MIN can be used to indicate a rate
!! with min as the time unit.
!!
!! The key-value pair \b aerosol \b phase is required and must specify the name
!! of the aerosol-phase in which the reaction occurs.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!> The rxn_condensed_phase_photolysis_t type and associated functions.
module camp_rxn_condensed_phase_photolysis

  use camp_aero_phase_data
  use camp_aero_rep_data
  use camp_chem_spec_data
  use camp_constants, only: const
  use camp_camp_state
  use camp_mpi
  use camp_property
  use camp_rxn_data
  use camp_util,      only: i_kind, dp, to_string, assert, assert_msg, die_msg, string_t

  use iso_c_binding

  implicit none
  private

#define NUM_REACT_ this%condensed_data_int(1)
#define NUM_PROD_ this%condensed_data_int(2)
#define NUM_AERO_PHASE_ this%condensed_data_int(3)
#define RXN_ID_ this%condensed_data_int(4)
#define SCALING_ this%condensed_data_real(1)
#define NUM_REAL_PROP_ 1
#define NUM_INT_PROP_ 4
#define NUM_ENV_PARAM_ 2
#define REACT_(x) this%condensed_data_int(NUM_INT_PROP_+x)
#define PROD_(x) this%condensed_data_int(NUM_INT_PROP_+NUM_REACT_*NUM_AERO_PHASE_+x)
#define WATER_(x) this%condensed_data_int(NUM_INT_PROP_+(NUM_REACT_+NUM_PROD_)*NUM_AERO_PHASE_+x)
#define DERIV_ID_(x) this%condensed_data_int(NUM_INT_PROP_+(NUM_REACT_+NUM_PROD_+1)*NUM_AERO_PHASE_+x)
#define JAC_ID_(x) this%condensed_data_int(NUM_INT_PROP_+(2*(NUM_REACT_+NUM_PROD_)+1)*NUM_AERO_PHASE_+x)
#define YIELD_(x) this%condensed_data_real(NUM_REAL_PROP_+x)
#define KGM3_TO_MOLM3_(x) this%condensed_data_real(NUM_REAL_PROP_+NUM_PROD_+x)

  public :: rxn_condensed_phase_photolysis_t, rxn_update_data_condensed_phase_photolysis_t

  !> Generic test reaction data type
  type, extends(rxn_data_t) :: rxn_condensed_phase_photolysis_t
  contains
    !> Reaction initialization
    procedure :: initialize
    !> Get the reaction property set
    procedure :: get_property_set
    !> Initialize update data
    procedure :: update_data_initialize
    !> Finalize the reaction
    final :: finalize
  end type rxn_condensed_phase_photolysis_t

  !> Constructor for rxn_condensed_phase_photolysis_t
  interface rxn_condensed_phase_photolysis_t
    procedure :: constructor
  end interface rxn_condensed_phase_photolysis_t

  !> Condensed-phase Photolysis rate update object
  type, extends(rxn_update_data_t) :: rxn_update_data_condensed_phase_photolysis_t
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
    final :: update_data_finalize
  end type rxn_update_data_condensed_phase_photolysis_t

    !> Interface to c reaction functions
  interface

    !> Allocate space for a rate update
    function rxn_condensed_phase_photolysis_create_rate_update_data() result (update_data) &
              bind (c)
      use iso_c_binding
      !> Allocated update_data object
      type(c_ptr) :: update_data
    end function rxn_condensed_phase_photolysis_create_rate_update_data

    !> Set a new photolysis rate
    subroutine rxn_condensed_phase_photolysis_set_rate_update_data(update_data, photo_id, &
              base_rate) bind (c)
      use iso_c_binding
      !> Update data
      type(c_ptr), value :: update_data
      !> Photo id
      integer(kind=c_int), value :: photo_id
      !> New pre-scaling base photolysis rate
      real(kind=c_double), value :: base_rate
    end subroutine rxn_condensed_phase_photolysis_set_rate_update_data

    !> Free an update rate data object
    pure subroutine rxn_free_update_data(update_data) bind (c)
      use iso_c_binding
      !> Update data
      type(c_ptr), value, intent(in) :: update_data
    end subroutine rxn_free_update_data

  end interface

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Constructor for Condensed-phase Photolysis reaction
  function constructor() result(new_obj)

    !> A new reaction instance
    type(rxn_condensed_phase_photolysis_t), pointer :: new_obj

    allocate(new_obj)
    new_obj%rxn_phase = AERO_RXN

  end function constructor

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Initialize the reaction data, validating component data and loading
  !! any required information into the condensed data arrays for use during
  !! solving
  subroutine initialize(this, chem_spec_data, aero_rep, n_cells)

    !> Reaction data
    class(rxn_condensed_phase_photolysis_t), intent(inout) :: this
    !> Chemical species data
    type(chem_spec_data_t), intent(in) :: chem_spec_data
    !> Aerosol representations
    type(aero_rep_data_ptr), pointer, intent(in) :: aero_rep(:)
    !> Number of grid cells to solve simultaneously
    integer(kind=i_kind), intent(in) :: n_cells

    type(property_t), pointer :: spec_props, reactants, products
    character(len=:), allocatable :: key_name, spec_name, water_name, &
            phase_name, temp_string
    integer(kind=i_kind) :: i_spec, i_phase_inst, i_qty, i_aero_rep, &
            i_aero_phase, num_spec_per_phase, num_phase, num_react, num_prod
    type(string_t), allocatable :: unique_names(:)
    type(string_t), allocatable :: react_names(:), prod_names(:)

    integer(kind=i_kind) :: temp_int
    real(kind=dp) :: temp_real
    logical :: is_aqueous

    is_aqueous = .false.

    ! Get the property set
    call assert_msg(852163263, associated(this%property_set), &
            "Missing property set needed to initialize reaction")

    ! Get the aerosol phase name
    key_name = "aerosol phase"
    call assert_msg(845445717, &
            this%property_set%get_string(key_name, phase_name), &
            "Missing aerosol phase in condensed-phase Photolysis reaction")

    ! Get reactant species
    key_name = "reactants"
    call assert_msg(501773136, &
            this%property_set%get_property_t(key_name, reactants), &
            "Missing reactant species in condensed-phase Photolysis reaction")

    ! Get product species
    key_name = "products"
    call assert_msg(556252922, &
            this%property_set%get_property_t(key_name, products), &
            "Missing product species in condensed-phase Photolysis reaction")

    ! Count the number of species per phase instance (including reactants
    ! with a "qty" specified)
    call reactants%iter_reset()
    num_react = 0
    do while (reactants%get_key(spec_name))
      ! Get properties included with this reactant in the reaction data
      call assert(428593951, reactants%get_property_t(val=spec_props))
      key_name = "qty"
      if (spec_props%get_int(key_name, temp_int)) &
              num_react = num_react + temp_int - 1
      call reactants%iter_next()
      num_react = num_react + 1
    end do

    ! Photolysis reactions only have 1 reactant, enforce that here
    call assert_msg(890212987, num_react .eq. 1, &
            "Too many species in condensed-phase Photolysis reaction. Only one reactant is expected.")

    num_prod = products%size()
    num_spec_per_phase = num_prod + num_react

    ! Check for aerosol representations
    call assert_msg(370755392, associated(aero_rep), &
            "Missing aerosol representation for condensed-phase "// &
            "Photolysis reaction")
    call assert_msg(483073737, size(aero_rep).gt.0, &
            "Missing aerosol representation for condensed-phase "// &
            "Photolysis reaction")

    ! Count the instances of the specified aerosol phase
    num_phase = 0
    do i_aero_rep = 1, size(aero_rep)
      num_phase = num_phase + &
              aero_rep(i_aero_rep)%val%num_phase_instances(phase_name)
    end do

    ! Allocate space in the condensed data arrays
    allocate(this%condensed_data_int(NUM_INT_PROP_ + &
            num_phase * (num_spec_per_phase * (num_react + 3) + 1)))
    allocate(this%condensed_data_real(NUM_REAL_PROP_ + &
            num_spec_per_phase + num_prod))
    this%condensed_data_int(:) = int(0, kind=i_kind)
    this%condensed_data_real(:) = real(0.0, kind=dp)

    ! Save space for the environment-dependent parameters
    this%num_env_params = NUM_ENV_PARAM_

    ! Set the number of products, reactants and aerosol phase instances
    NUM_REACT_ = num_react
    NUM_PROD_ = num_prod
    NUM_AERO_PHASE_ = num_phase

    ! Get reaction parameters (it might be easiest to keep these at the
    ! beginning of the condensed data array, so they can be accessed using
    ! compliler flags)
    key_name = "scaling factor"
    if (.not. this%property_set%get_real(key_name, SCALING_)) then
      SCALING_ = real(1.0, kind=dp)
    end if

    ! Set up an array to the reactant and product names
    allocate(react_names(NUM_REACT_))
    allocate(prod_names(NUM_PROD_))

    ! Get the chemical properties for the reactants
    call reactants%iter_reset()
    i_spec = 0
    do while (reactants%get_key(spec_name))

      ! Get the reactant species properties
      call assert_msg(365229559, &
           chem_spec_data%get_property_set(spec_name, spec_props), &
           "Missing properties required for condensed-phase Photolysis "// &
           "reaction involving species '"//trim(spec_name)//"'")

      ! Get the molecular weight
      key_name = "molecular weight [kg mol-1]"
      call assert_msg(409180731, spec_props%get_real(key_name, temp_real), &
           "Missing 'molecular weight' for species '"//trim(spec_name)// &
           "' in condensed-phase Photolysis reaction.")

      ! Set properties for each occurance of a reactant in the rxn equation
      call assert(186449575, reactants%get_property_t(val=spec_props))
      key_name = "qty"
      if (.not.spec_props%get_int(key_name, temp_int)) temp_int = 1
      do i_qty = 1, temp_int
        i_spec = i_spec + 1

        ! Add the reactant name to the list
        react_names(i_spec)%string = spec_name

        ! Use the MW to calculate the kg/m3 -> mol/m3 conversion
        KGM3_TO_MOLM3_(i_spec) = 1.0/temp_real

      end do

      ! Go to the next reactant
      call reactants%iter_next()

    end do

    ! Get the chemical properties for the products
    call products%iter_reset()
    i_spec = 0
    do while (products%get_key(spec_name))

      ! Get the product species properties
      call assert_msg(450225425, &
           chem_spec_data%get_property_set(spec_name, spec_props), &
           "Missing properties required for condensed-phase Photolysis "// &
           "reaction involving species '"//trim(spec_name)//"'")

      ! Increment the product counter
      i_spec = i_spec + 1

      ! Get the molecular weight
      key_name = "molecular weight [kg mol-1]"
      call assert_msg(504705211, spec_props%get_real(key_name, temp_real), &
           "Missing 'molecular weight' for species '"//trim(spec_name)// &
           "' in condensed phase Photolysis reaction.")

      ! Use the MW to calculate the kg/m3 -> mol/m3 conversion
      KGM3_TO_MOLM3_(NUM_REACT_+i_spec) = 1.0/temp_real

      ! Set properties for each occurance of a reactant in the rxn equation
      call assert(846924553, products%get_property_t(val=spec_props))
      key_name = "yield"
      if (spec_props%get_real(key_name, temp_real)) then
        YIELD_(i_spec) = temp_real
      else
        YIELD_(i_spec) = 1.0d0
      end if

      ! Add the product name to the list
      prod_names(i_spec)%string = spec_name

      ! Go to the next product
      call products%iter_next()

    end do

    ! Get the units for the reactants and products and name for water
    ! if this is an aqueous reaction
    key_name = "units"
    call assert_msg(348722817, &
            this%property_set%get_string(key_name, temp_string), &
            "Missing units for condensed-phase Photolysis reaction.")
    if (trim(temp_string).eq."mol m-3") then
      is_aqueous = .false.
      key_name = "aerosol-phase water"
      call assert_msg(767767240, &
              .not.this%property_set%get_string(key_name, temp_string), &
              "Aerosol-phase water specified for non-aqueous condensed-"// &
              "phase Photolysis reaction. Change units to 'M' or remove "// &
              "aerosol-phase water")
    else if (trim(temp_string).eq."M") then
      is_aqueous = .true.
      key_name = "aerosol-phase water"
      call assert_msg(199910264, &
              this%property_set%get_string(key_name, water_name), &
              "Missing aerosol-phase water for aqeuous condensed-phase "// &
              "Photolysis reaction.")
    else
      call die_msg(161772048, "Received invalid units for condensed-"// &
              "phase Photolysis reaction: '"//temp_string//"'. Valid "// &
              "units are 'mol m-3' or 'M'.")
    end if

    ! Set the state array indices for the reactants, products and water
    i_aero_phase = 0
    do i_aero_rep = 1, size(aero_rep)

      ! Check for the specified phase in this aero rep
      num_phase = aero_rep(i_aero_rep)%val%num_phase_instances(phase_name)
      if (num_phase.eq.0) cycle

      ! Save the state ids for aerosol-phase water for aqueous reactions.
      ! For non-aqueous reactions, set the water id to -1
      if (is_aqueous) then

        ! Get the unique names for aerosol-phase water
        unique_names = aero_rep(i_aero_rep)%val%unique_names( &
                phase_name = phase_name, spec_name = water_name)

        ! Make sure water is present
        call assert_msg(196838614, size(unique_names).eq.num_phase, &
                "Missing aerosol-phase water species '"//water_name// &
                "' in phase '"//phase_name//"' in aqueous condensed-"// &
                "phase Photolysis reacion.")

        ! Save the ids for water in this phase
        do i_phase_inst = 1, num_phase
          WATER_(i_aero_phase + i_phase_inst) = &
                  aero_rep(i_aero_rep)%val%spec_state_id( &
                  unique_names(i_phase_inst)%string)
        end do

        deallocate(unique_names)

      else

        ! Set the water ids to -1 for non-aqueous condensed-phase reactions
        do i_phase_inst = 1, num_phase
          WATER_(i_aero_phase + i_phase_inst) = -1
        end do

      end if

      ! Loop through the reactants
      do i_spec = 1, NUM_REACT_

        ! Get the unique names for the reactants
        unique_names = aero_rep(i_aero_rep)%val%unique_names( &
                phase_name = phase_name, spec_name = react_names(i_spec)%string)

        ! Make sure the right number of instances are present
        call assert_msg(360730267, size(unique_names).eq.num_phase, &
                "Incorrect instances of reactant '"// &
                react_names(i_spec)%string//"' in phase '"//phase_name// &
                "' in a condensed-phase Photolysis reaction")

        ! Save the state ids for the reactant concentration
        ! IDs are grouped by phase instance:
        !   R1(phase1), R2(phase1), ..., R1(phase2)...
        do i_phase_inst = 1, num_phase
          REACT_((i_aero_phase+i_phase_inst-1)*NUM_REACT_ + i_spec) = &
                  aero_rep(i_aero_rep)%val%spec_state_id( &
                  unique_names(i_phase_inst)%string)
        end do

        deallocate(unique_names)

      end do

      ! Loop through the products
      do i_spec = 1, NUM_PROD_

        ! Get the unique names for the products
        unique_names = aero_rep(i_aero_rep)%val%unique_names( &
                phase_name = phase_name, spec_name = prod_names(i_spec)%string)

        ! Make sure the right number of instances are present
        call assert_msg(399869427, size(unique_names).eq.num_phase, &
                "Incorrect instances of product '"// &
                prod_names(i_spec)%string//"' in phase '"//phase_name// &
                "' in a condensed-phase Photolysis reaction")

        ! Save the state ids for the product concentration
        ! IDs are grouped by phase instance:
        !   P1(phase1), P2(phase1), ..., P1(phase2)...
        do i_phase_inst = 1, num_phase
          PROD_((i_aero_phase+i_phase_inst-1)*NUM_PROD_ + i_spec) = &
                  aero_rep(i_aero_rep)%val%spec_state_id( &
                  unique_names(i_phase_inst)%string)
        end do

        deallocate(unique_names)

      end do

      ! Increment the index offset for the next aerosol representation
      i_aero_phase = i_aero_phase + num_phase

    end do

    ! Initialize the reaction id
    RXN_ID_ = -1

  end subroutine initialize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Get the reaction properties. (For use by external photolysis modules.)
  function get_property_set(this) result(prop_set)

    !> Reaction properties
    type(property_t), pointer :: prop_set
    !> Reaction data
    class(rxn_condensed_phase_photolysis_t), intent(in) :: this

    prop_set => this%property_set

  end function get_property_set

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Finalize the reaction
  elemental subroutine finalize(this)

    !> Reaction data
    type(rxn_condensed_phase_photolysis_t), intent(inout) :: this

    if (associated(this%property_set)) &
            deallocate(this%property_set)
    if (allocated(this%condensed_data_real)) &
            deallocate(this%condensed_data_real)
    if (allocated(this%condensed_data_int)) &
            deallocate(this%condensed_data_int)

  end subroutine finalize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Set packed update data for photolysis rate constants
  subroutine update_data_rate_set(this, base_rate)

    !> Update data
    class(rxn_update_data_condensed_phase_photolysis_t), intent(inout) :: this
    !> Updated pre-scaling photolysis rate
    real(kind=dp), intent(in) :: base_rate

    call rxn_condensed_phase_photolysis_set_rate_update_data(this%get_data(), &
            this%rxn_unique_id, base_rate)

  end subroutine update_data_rate_set

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Initialize update data
  subroutine update_data_initialize(this, update_data, rxn_type)

    use camp_rand,                                only : generate_int_id

    !> The reaction to update
    class(rxn_condensed_phase_photolysis_t), intent(inout) :: this
    !> Update data object
    class(rxn_update_data_condensed_phase_photolysis_t), intent(out) :: update_data
    !> Reaction type id
    integer(kind=i_kind), intent(in) :: rxn_type

    ! If a reaction id has not yet been generated, do it now
    if (RXN_ID_.eq.-1) then
      RXN_ID_ = generate_int_id()
    endif

    update_data%rxn_unique_id = RXN_ID_
    update_data%rxn_type = int(rxn_type, kind=c_int)
    update_data%update_data = rxn_condensed_phase_photolysis_create_rate_update_data()
    update_data%is_malloced = .true.

  end subroutine update_data_initialize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Determine the size of a binary required to pack the reaction data
  integer(kind=i_kind) function internal_pack_size(this, comm) &
      result(pack_size)

    !> Reaction update data
    class(rxn_update_data_condensed_phase_photolysis_t), intent(in) :: this
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
    class(rxn_update_data_condensed_phase_photolysis_t), intent(in) :: this
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
    call assert(649543400, &
         pos - prev_position <= this%pack_size(comm))
#endif

  end subroutine internal_bin_pack

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Unpack the given value from the buffer, advancing position
  subroutine internal_bin_unpack(this, buffer, pos, comm)

    !> Reaction update data
    class(rxn_update_data_condensed_phase_photolysis_t), intent(inout) :: this
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
    call assert(254749806, &
         pos - prev_position <= this%pack_size(comm))
    this%update_data = rxn_condensed_phase_photolysis_create_rate_update_data()
#endif

  end subroutine internal_bin_unpack

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Finalize an update data object
  elemental subroutine update_data_finalize(this)

    !> Update data object to free
    type(rxn_update_data_condensed_phase_photolysis_t), intent(inout) :: this

    if (this%is_malloced) call rxn_free_update_data(this%update_data)

  end subroutine update_data_finalize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end module camp_rxn_condensed_phase_photolysis
