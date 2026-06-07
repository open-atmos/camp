! Copyright (C) 2021 Barcelona Supercomputing Center and University of
! Illinois at Urbana-Champaign
! SPDX-License-Identifier: MIT

!> \file
!> The camp_rxn_condensed_phase_diffusion module.

!> \page camp_rxn_condensed_phase_diffusion CAMP: Condensed Phase Diffusion Reaction
!!
!! Condensed phase diffusion reactions are based on Fick's Law of 
!! diffusion and align with kinetic modeling (e.g. \cite Shiraiwa et al. (2010))
!! diffusion representation. 
!!
!!
!! Input data for condensed phase diffusion reactions have the following format :
!! \code{.json}
!!     {
!!    "name" : "condensed phase diffusion",
!!    "type" : "MECHANISM",
!!    "reactions" : [
!!      {
!!        "type" : "CONDENSED_PHASE_DIFFUSION",
!!        "species": [{
!!          "phase": "aqueous",
!!          "name": "H2O_aq"
!!      },
!!      {
!!        "phase": "organic",
!!        "name": "H2O_org"
!!      }]
!!      }
!!    ]}
!! \endcode
!! The key-value pairs \b condensed phase and associated \b species 
!! of the diffusing species are required. The species indicated must 
!! have an associated diffusion coefficient [m2 s-1] listed as a
!! property assocaited with the phase it exists in.
!!
!!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!> The rxn_condensed_phase_diffusion_t type and associated functions.
module camp_rxn_condensed_phase_diffusion

  use camp_aero_phase_data
  use camp_aero_rep_data
  use camp_chem_spec_data
  use camp_constants,                        only: const
  use camp_camp_state
  use camp_property
  use camp_rxn_data
  use camp_util,                             only: i_kind, dp, to_string, &
                                                  assert, assert_msg, &
                                                  die_msg, string_t

  implicit none
  private

#define NUM_ADJACENT_PAIRS_ this%condensed_data_int(1)

#define NUM_INT_PROP_ 1
#define NUM_REAL_PROP_ 0
#define NUM_ENV_PARAM_ 0
#define BLOCK_SIZE_ 1000

#define DIFF_COEFF_FIRST_(x) this%condensed_data_real(NUM_REAL_PROP_ + (x)) 
#define DIFF_COEFF_SECOND_(x) this%condensed_data_real(NUM_REAL_PROP_ + NUM_ADJACENT_PAIRS_ + (x))
! PHASE_ID_FIRST_ and PHASE_ID_SECOND_ are arrays of 
! length NUM_ADJACENT_PAIRS_
#define PHASE_ID_FIRST_(x) this%condensed_data_int(NUM_INT_PROP_ + (x))
#define PHASE_ID_SECOND_(x) this%condensed_data_int(NUM_INT_PROP_ + NUM_ADJACENT_PAIRS_ + (x))
#define AERO_REP_ID_(x) this%condensed_data_int(NUM_INT_PROP_ + 2*NUM_ADJACENT_PAIRS_ + (x))
#define AERO_SPEC_FIRST_(x) this%condensed_data_int(NUM_INT_PROP_ + 3*NUM_ADJACENT_PAIRS_ + (x))
#define AERO_SPEC_SECOND_(x) this%condensed_data_int(NUM_INT_PROP_ + 4*NUM_ADJACENT_PAIRS_ + (x))
#define DERIV_ID_(x) this%condensed_data_int(NUM_INT_PROP_ + 5*NUM_ADJACENT_PAIRS_ + (x))
!#define JAC_ID_(x) this%condensed_data_int(4*BLOCK_SIZE_ + x)
!#define PHASE_INT_LOC_(x) this%condensed_data_int(5*BLOCK_SIZE_ + x) 
!#define PHASE_REAL_LOC_(x) this%condensed_data_int(6*BLOCK_SIZE_ + x)
!#define NUM_AERO_PHASE_JAC_ELEM_FIRST_(x) this%condensed_data_int(10*BLOCK_SIZE_ + x)
!#define NUM_AERO_PHASE_JAC_ELEM_SECOND_(x) this%condensed_data_int(11*BLOCK_SIZE_ + x)
!#define PHASE_JAC_ID_(x) this%condensed_data_int(12*BLOCK_SIZE_ + x)
!#define NUM_CONC_JAC_ELEM_(x) this%condensed_data_int(13*BLOCK_SIZE_ + x)
!#define MASS_JAC_ELEM_(x) this%condensed_data_int(14*BLOCK_SIZE_ + x)

  public :: rxn_condensed_phase_diffusion_t

  !> Generic test reaction data type
  type, extends(rxn_data_t) :: rxn_condensed_phase_diffusion_t
  contains
    !> Reaction initialization
    procedure :: initialize
    !> Finalize the reaction
    final :: finalize, finalize_array
  end type rxn_condensed_phase_diffusion_t

  !> Constructor for rxn_condensed_phase_diffusion_t
  interface rxn_condensed_phase_diffusion_t
    procedure :: constructor
  end interface rxn_condensed_phase_diffusion_t

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Constructor for Phase transfer reaction
  function constructor() result(new_obj)

    !> A new reaction instance
    type(rxn_condensed_phase_diffusion_t), pointer :: new_obj

    allocate(new_obj)
    new_obj%rxn_phase = AERO_RXN

  end function constructor

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Initialize the reaction data, validating component data and loading
  !! any required information into the condensed data arrays for use during
  !! solving
  subroutine initialize(this, chem_spec_data, aero_phase, aero_rep, n_cells)

    !> Reaction data
    class(rxn_condensed_phase_diffusion_t), intent(inout) :: this
    !> Chemical species data
    type(chem_spec_data_t), intent(in) :: chem_spec_data
    !> Aerosol phase data
    type(aero_phase_data_ptr), intent(in) :: aero_phase(:)
    !> Aerosol representations
    type(aero_rep_data_ptr), pointer, intent(in) :: aero_rep(:)
    type(camp_state_t), pointer :: camp_state
    !> Number of grid cells to solve simultaneously
    integer(kind=i_kind), intent(in) :: n_cells

    character(len=:), allocatable :: phase_names_first(:), spec_names_first(:)
    character(len=:), allocatable :: phase_names_second(:), spec_names_second(:)
    type(aero_phase_data_t), pointer :: aero_phase_data

    type(string_t), allocatable :: diffusion_phase_names(:)
    type(string_t), allocatable :: diffusion_species_names(:)
    type(property_t), pointer :: species, spec_props, spec_property_set, aero_phase_property_set
    character(len=:), allocatable :: key_name, aero_spec_name, error_msg
    character(len=:), allocatable :: phase_name, species_name
    integer(kind=i_kind) :: num_adjacent_pairs, phase_id_array_size
    integer(kind=i_kind) :: state_size, max_particles, offset, i_particle 
    integer(kind=i_kind) :: i_spec, i_aero_rep, i_aero_id, i, i_adj_rep_id
    integer(kind=i_kind) :: i_phase, i_species, tmp_size
    integer(kind=i_kind) :: n_aero_jac_elem_first, n_aero_jac_elem_second
    integer(kind=i_kind) :: i_adj_pairs, i_phase_ids
    type(string_t), allocatable :: unique_spec_names(:)
    integer(kind=i_kind), allocatable :: adj_phase_size(:)
    type(index_pair_t), allocatable :: adjacent_phases(:)
    integer(kind=i_kind), allocatable :: phase_ids(:)
    real(kind=dp) :: temp_real

    ! Get the property set
    if (.not. associated(this%property_set)) call die_msg(300992470, &
            "Missing property set needed to initialize reaction")

    ! Get the species involved in diffusion
    key_name = "species"
    call assert_msg(712818751, &
                    this%property_set%get_property_t(key_name, species), &
                    "Missing species for condensed phase diffusion "// &
                    "reaction")
                    call assert_msg(340551815, species%size() .gt. 0, &
                    "No species specified for the condensed phase "// &
                    "diffusion reaction.")
                    call assert_msg(023007260, species%size() .lt. 3, &
                    "Too many species specified for condensed phase "// &
                    "diffusion reaction (two species maximum).")


    ! Allocate space for phases and species involved in reaction
    allocate(diffusion_phase_names(species%size()))
    allocate(diffusion_species_names(species%size()))
    
    call species%iter_reset()
    do i_species = 1, species%size()

      ! Get the species properties
      call assert_msg(815257799, species%get_property_t(val=spec_props), &
              "Invalid structure for species '"// &
              to_string(i_species)// &
              "' in condensed phase diffusion reaction.")

      ! Get the phase names
      key_name = "phase"
      call assert_msg(354574496, spec_props%get_string(key_name, phase_name), &
              "Missing phase name in condensed phase diffusion reaction for species "// &
              phase_name)
      diffusion_phase_names(i_species)%string = phase_name

      ! Get the associated species names
      key_name = "name"
      call assert_msg(629919883, spec_props%get_string(key_name, species_name), &
              "Missing species name in condensed phase reaction for species "// &
              species_name)
      diffusion_species_names(i_species)%string = species_name
      print *, "Condensed phase diffusion reaction species ", i_species, ": '", &
               diffusion_species_names(i_species)%string, "' in phase '", &
               diffusion_phase_names(i_species)%string, "'."

      call species%iter_next()
    end do

    ! If only one phase/species pair is found then add that phase/species pair
    ! to the second element of the diffusion_phase_names and diffusion_species_names
    ! arrays for indexing purposes
    if (allocated(diffusion_species_names)) then
      if (size(diffusion_species_names) == 1) then
        diffusion_phase_names = [diffusion_phase_names, diffusion_phase_names(1)]
        diffusion_species_names = [diffusion_species_names, diffusion_species_names(1)]
      end if
    end if

    ! Make sure the phase and species names array are the correct length.
    call assert_msg(593348903, size(diffusion_phase_names) .le. 2, &
       "Too many diffusing species in diffusion_phase_names array.")
    call assert_msg(379981970, size(diffusion_species_names) .le. 2, &
       "Too many diffusing species in diffusion_species_names array.") 
    
    ! Allocate space in the condensed data arrays early so macros can be used
    allocate(this%condensed_data_int(BLOCK_SIZE_ * 20 ))
    allocate(this%condensed_data_real(BLOCK_SIZE_ * 1 ))
    this%condensed_data_int(:) = int(0, kind=i_kind)
    this%condensed_data_real(:) = real(0.0, kind=dp)

    ! Check that the species exist in adjacent layers. 
    ! For the modal/binned aerosol represetnation (no layers) the adjacent_phases array
    ! is always 0.
    ! Accumulate adjacent phase pairs from all aerosol representations
    allocate(adj_phase_size(size(aero_rep)))
    num_adjacent_pairs = 0
    do i_aero_rep = 1, size(aero_rep) 
      adjacent_phases = aero_rep(i_aero_rep)%val%adjacent_phases(diffusion_phase_names(1)%string, &
         diffusion_phase_names(SIZE(diffusion_phase_names))%string)
      adj_phase_size(i_aero_rep) = size(adjacent_phases)
      !print *, "Aerosol representation ", i_aero_rep, " has ", size(adjacent_phases), " adjacent phase pairs for diffusion between '", &
      !         diffusion_phase_names(1)%string, "' and '", diffusion_phase_names(SIZE(diffusion_phase_names))%string, "'."
      if (size(adjacent_phases) .gt. 0) then
        do i = 1, size(adjacent_phases)
          num_adjacent_pairs = num_adjacent_pairs + 1
          PHASE_ID_FIRST_(num_adjacent_pairs) = adjacent_phases(i)%first_
        end do
        NUM_ADJACENT_PAIRS_ = num_adjacent_pairs
      end if
    end do

    num_adjacent_pairs = 0
    do i_aero_rep = 1, size(aero_rep) 
      adjacent_phases = aero_rep(i_aero_rep)%val%adjacent_phases(diffusion_phase_names(1)%string, &
         diffusion_phase_names(SIZE(diffusion_phase_names))%string)
      if (size(adjacent_phases) .gt. 0) then
        do i = 1, size(adjacent_phases)
          num_adjacent_pairs = num_adjacent_pairs + 1
          PHASE_ID_SECOND_(num_adjacent_pairs) = adjacent_phases(i)%second_
        end do
      end if
    end do
    
    call assert_msg(051987857, num_adjacent_pairs .gt. 0, &
       "No adjacent phases found condensed phase diffusion reaction.")

    ! Create the arrays of diffusion coefficients for each adjacent phase pair
    do i_aero_rep = 1, size(aero_rep) 
      do i_phase = 1, size(aero_phase)
        do i_adj_pairs = 1, num_adjacent_pairs
          ! Find the FIRST phase in the pair
          if (aero_phase(i_phase)%val%name() .eq. diffusion_phase_names(1)%string) then
            spec_property_set => aero_phase(i_phase)%val%get_spec_property_set( &
                    diffusion_species_names(1)%string)
            key_name = "diffusion coefficient [m2 s-1]"
            if (spec_property_set%get_real(key_name, temp_real)) then
              DIFF_COEFF_FIRST_(i_adj_pairs) = temp_real
            end if
          end if
        end do
      end do
      
      ! Find the SECOND phase in the pair
      do i_phase = 1, size(aero_phase)
        do i_adj_pairs = 1, num_adjacent_pairs
         ! Find the SECOND phase in the pair
          if (aero_phase(i_phase)%val%name() .eq. diffusion_phase_names(SIZE(diffusion_phase_names))%string) then
            spec_property_set => aero_phase(i_phase)%val%get_spec_property_set( &
                    diffusion_species_names(SIZE(diffusion_species_names))%string)
            key_name = "diffusion coefficient [m2 s-1]"
            if (spec_property_set%get_real(key_name, temp_real)) then
             DIFF_COEFF_SECOND_(i_adj_pairs) = temp_real
            else
              print *, "WARNING: Could not find diffusion coefficient for pair ", i_adj_pairs
            end if
          end if
        end do
      end do
    end do

    ! Set up a general error message
    error_msg = " for condensed phase diffusion of aerosol species '"// &
                diffusion_species_names(1)%string//"' to aerosol species '"// &
                diffusion_species_names(SIZE(diffusion_species_names))%string

    ! Check for aerosol representations
    call assert_msg(161043212, associated(aero_rep), &
            "Missing aerosol representation"//error_msg)
    call assert_msg(411220610, size(aero_rep).gt.0, &
            "Missing aerosol representation"//error_msg)

    ! Save space for the environment-dependent parameters
    this%num_env_params = NUM_ENV_PARAM_

     ! Set aerosol phase specific indices
    i_adj_rep_id = 1
    i_aero_id = 1
    do i_aero_rep = 1, size(aero_rep)
      do i_adj_pairs = 1, adj_phase_size(i_aero_rep)
        AERO_REP_ID_(i_adj_rep_id) = i_aero_id
        i_adj_rep_id = i_adj_rep_id + 1
      end do
      i_aero_id = i_aero_id + 1
    end do

    deallocate(adj_phase_size)

  end subroutine initialize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Finalize the reaction
  subroutine finalize(this)

    !> Reaction data
    type(rxn_condensed_phase_diffusion_t), intent(inout) :: this

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
  
    !> Array of reaction data
    type(rxn_condensed_phase_diffusion_t), intent(inout) :: this(:)

    integer(kind=i_kind) :: i

    do i = 1, size(this)
      call finalize(this(i))
    end do

  end subroutine finalize_array

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end module camp_rxn_condensed_phase_diffusion
