! Copyright (C) 2021 Barcelona Supercomputing Center and University of
! Illinois at Urbana-Champaign
! SPDX-License-Identifier: MIT


!> \file
!> The camp_rxn_vbs_nox_dependent module.


!> \page camp_rxn_vbs_nox_dependent CAMP: VBS reaction
!!
!! Calculate VBS species formation rates according to Shrivastava et al. (2019)
!! Forming VBS species follows these reactions:
!! \f[
!!   \mathrm{BVOC} + \mathrm{Ox} \rightarrow \sum_{i=1}^4 a_i \mathrm{VBS}_i
!! \f]
!! BVOC=ISOP, TERP, SQTERP
!! Ox=OH, O3, NO3
!!
!! The rate constants follow the usual Arrhenius kinetics.
!! The branching ratios depend on NO concentrations
!!
!! \f[a_i = a_{i,\mathrm{high}}B + a_{i,\mathrm{low}}(1-B)\f]
!! \f[
!!    B = \mathrm{NO_x ratio} = \frac{\mathrm{NO}\times k_{\mathrm{RO_2+NO}}}{\mathrm{NO}\times k_{\mathrm{RO_2+NO}} + \mathrm{HO_2}\times k_{\mathrm{RO_2+HO_2}} + \mathrm{RO_2}\times k_{\mathrm{RO_2+RO_2}}}$
!! \f]
!! 
!! Input format for VBS NOx dependent reactions have this format (similar to the ARRHENIUS format, but the product yields are mandatory)
!! \code{.json}
!!   {
!!     "type" : "VBS_NOX_DEPENDENT",
!!     "A" : 123.45,
!!     "Ea" : 123.45,
!!     "B"  : 1.3,
!!     "D"  : 300.0,
!!     "E"  : 0.6E-5,
!!     "time unit" : "MIN",
!!     "NO_name" : "NO",
!!     "HO2_name" : "HO2",
!!     "K_RO2_NO" : 9.04e-12,
!!     "K_RO2_HO2" : 2.28e-11,
!!     "reactants" : {
!!       "spec1" : {},
!!       "spec2" : { "qty" : 2 },
!!       ...
!!     },
!!     "products" : {
!!       "spec3" : { 
!!              "yield_high_nox" : 0,
!!              "yield_low_nox"  : 0.0008
!!       },
!!       "spec4" : { 
!!              "yield_high_nox" : 0.063,
!!              "yield_low_nox"  : 0.036
!!       },
!!       ...
!!     }
!!   }
!! \endcode
!! The key-value pairs \b reactants, and \b products are required. Reactants
!! without a \b qty value are assumed to appear once in the reaction equation.
!! For each product, key-value pairs \b yield_high_nox and \b yield_low_nox are required.
!!
!! Optionally, a parameter \b C may be included, and is taken to equal
!! \f$\frac{-E_a}{k_b}\f$. Note that either \b Ea or \b C may be included, but
!! not both. When neither \b Ea or \b C are included, they are assumed to be
!! 0.0. When \b A is not included, it is assumed to be 1.0, when \b D is not
!! included, it is assumed to be 300.0 K, when \b B is not included, it is
!! assumed to be 0.0, and when \b E is not included, it is assumed to be 0.0.
!! The unit for time is assumed to be s, but inclusion of the optional
!! key-value pair \b time \b unit = \b MIN can be used to indicate a rate
!! with min as the time unit.
!! The RO2 reaction rates used to calculate branching ratios may also be included
!! as \b K_RO2_NO and \b K_RO2_HO2. If they are not included, their values are 
!! assumed to be 9.04e-12 cm3/molec/s and 2.28e-11 cm3/molec/s respectively
!! following MCM values at 298K
!! Optionally, the names used for NO and HO2 in the gas phase mechanism may be indicated.
!! By default, "NO" and "HO2" are assumed to be their names.


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!> The rxn_vbs_nox_dependent_t type and associated functions.
module camp_rxn_vbs_nox_dependent

    use camp_aero_rep_data
    use camp_chem_spec_data
    use camp_constants,                        only: const
    use camp_camp_state
    use camp_property
    use camp_rxn_data
    use camp_util,                             only: i_kind, dp, to_string, &
                                                    assert, assert_msg, die_msg
  
    implicit none
    private

#define NUM_REACT_ this%condensed_data_int(1)
#define NUM_PROD_ this%condensed_data_int(2)
#define NO_ID_ this%condensed_data_int(3)
#define HO2_ID_ this%condensed_data_int(4)
#define A_ this%condensed_data_real(1)
#define B_ this%condensed_data_real(2)
#define C_ this%condensed_data_real(3)
#define D_ this%condensed_data_real(4)
#define E_ this%condensed_data_real(5)
#define K_RO2_NO_ this%condensed_data_real(6)
#define K_RO2_HO2_ this%condensed_data_real(7)
#define CONV_ this%condensed_data_real(8)
#define NUM_INT_PROP_ 4
#define NUM_REAL_PROP_ 8
#define NUM_ENV_PARAM_ 1
#define REACT_(x) this%condensed_data_int(NUM_INT_PROP_ + x)
#define PROD_(x) this%condensed_data_int(NUM_INT_PROP_ + NUM_REACT_ + x)
#define DERIV_ID_(x) this%condensed_data_int(NUM_INT_PROP_ + NUM_REACT_ + NUM_PROD_ + x)
#define JAC_ID_(x) this%condensed_data_int(NUM_INT_PROP_ + 2*(NUM_REACT_+NUM_PROD_) + x)
#define LOW_NOX_YIELD_(x) this%condensed_data_real(NUM_REAL_PROP_ + 2*x - 1)
#define HIGH_NOX_YIELD_(x) this%condensed_data_real(NUM_REAL_PROP_ + 2*x)
  
    public :: rxn_vbs_nox_dependent_t
  
    !> Generic test reaction data type
    type, extends(rxn_data_t) :: rxn_vbs_nox_dependent_t
    contains
      !> Reaction initialization
      procedure :: initialize
      !> Finalize the reaction
      final :: finalize
    end type rxn_vbs_nox_dependent_t
  
    !> Constructor for rxn_arrhenius_t
    interface rxn_vbs_nox_dependent_t
      procedure :: constructor
    end interface rxn_vbs_nox_dependent_t

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Constructor for VBS reaction
  function constructor() result(new_obj)

      !> A new reaction instance
      type(rxn_vbs_nox_dependent_t), pointer :: new_obj

      allocate(new_obj)
      new_obj%rxn_phase = GAS_RXN

  end function constructor


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Initialize the reaction data, validating component data and loading
  !! any required information into the condensed data arrays for use during
  !! solving
  subroutine initialize(this, chem_spec_data, aero_rep, n_cells)

    !> Reaction data
    class(rxn_vbs_nox_dependent_t), intent(inout) :: this
    !> Chemical species data
    type(chem_spec_data_t), intent(in) :: chem_spec_data
    !> Aerosol representations
    type(aero_rep_data_ptr), pointer, intent(in) :: aero_rep(:)
    !> Number of grid cells being solved simultaneously
    integer(kind=i_kind), intent(in) :: n_cells

    type(property_t), pointer :: spec_props, reactants, products
    character(len=:), allocatable :: key_name, spec_name, string_val
    integer(kind=i_kind) :: i_spec, i_qty

    integer(kind=i_kind) :: temp_int
    real(kind=dp) :: temp_real

    ! Get the species involved
    if (.not. associated(this%property_set)) call die_msg(820664040, &
            "Missing property set needed to initialize reaction")
    key_name = "reactants"
    call assert_msg(410497089, &
            this%property_set%get_property_t(key_name, reactants), &
            "VBS nox dependent reaction is missing reactants")
    key_name = "products"
    call assert_msg(423133845, &
            this%property_set%get_property_t(key_name, products), &
            "VBS nox dependent reaction is missing products")

    ! Count the number of reactants (including those with a qty specified)
    call reactants%iter_reset()
    i_spec = 0
    do while (reactants%get_key(spec_name))
      ! Get properties included with this reactant in the reaction data
      call assert(148873688, reactants%get_property_t(val=spec_props))
      key_name = "qty"
      if (spec_props%get_int(key_name, temp_int)) i_spec = i_spec+temp_int-1
      call reactants%iter_next()
      i_spec = i_spec + 1
    end do

    ! Allocate space in the condensed data arrays
    allocate(this%condensed_data_int(NUM_INT_PROP_ + &
            (i_spec + 2) * (i_spec + products%size())))            
    allocate(this%condensed_data_real(NUM_REAL_PROP_ + 2*products%size()))
    this%condensed_data_int(:) = int(0, kind=i_kind)
    this%condensed_data_real(:) = real(0.0, kind=dp)

    ! Save space for the environment dependent parameters
    this%num_env_params = NUM_ENV_PARAM_

    ! Save the size of the reactant and product arrays (for reactions where
    ! these can vary)
    NUM_REACT_ = i_spec
    NUM_PROD_ = products%size()

    ! Set the #/cc -> ppm conversion prefactor
    CONV_ = const%avagadro / const%univ_gas_const * 10.0d0**(-12.0d0)

    ! Get reaction parameters (it might be easiest to keep these at the
    ! beginning of the condensed data array, so they can be accessed using
    ! compliler flags)
    key_name = "A"
    if (.not. this%property_set%get_real(key_name, A_)) then
      A_ = 1.0
    end if
    key_name = "time unit"
    if (this%property_set%get_string(key_name, string_val)) then
      if (trim(string_val).eq."MIN") then
        A_ = A_ / 60.0
      end if
    endif
    key_name = "Ea"
    if (this%property_set%get_real(key_name, temp_real)) then
      C_ = -temp_real/const%boltzmann
      key_name = "C"
      call assert_msg(345898525, &
              .not.this%property_set%get_real(key_name, temp_real), &
              "Received both Ea and C parameter for VBS NOx dependent equation")
    else
      key_name = "C"
      if (.not. this%property_set%get_real(key_name, C_)) then
        C_ = 0.0
      end if
    end if
    key_name = "D"
    if (.not. this%property_set%get_real(key_name, D_)) then
      D_ = 300.0
    end if
    key_name = "B"
    if (.not. this%property_set%get_real(key_name, B_)) then
      B_ = 0.0
    end if
    key_name = "E"
    if (.not. this%property_set%get_real(key_name, E_)) then
      E_ = 0.0
    end if
    key_name = "K_RO2_NO"
    if (.not. this%property_set%get_real(key_name, K_RO2_NO_)) then
      K_RO2_NO_ = 9.04e-12
    end if
    key_name = "K_RO2_HO2"
    if (.not. this%property_set%get_real(key_name, K_RO2_HO2_)) then
      K_RO2_HO2_ = 2.28e-11
    end if

    
    call assert_msg(107242566, .not. ((B_.ne.real(0.0, kind=dp)) &
            .and.(D_.eq.real(0.0, kind=dp))), &
            "D cannot be zero if B is non-zero in Arrhenius reaction.")

    ! Get the indices and chemical properties for the reactants
    call reactants%iter_reset()
    i_spec = 1
    do while (reactants%get_key(spec_name))

      ! Save the index of this species in the state variable array
      REACT_(i_spec) = chem_spec_data%gas_state_id(spec_name)

      ! Make sure the species exists
      call assert_msg(918811704, REACT_(i_spec).gt.0, &
              "Missing VBS NOx dependent reactant: "//spec_name)

      ! Get properties included with this reactant in the reaction data
      call assert(238579749, reactants%get_property_t(val=spec_props))
      key_name = "qty"
      if (spec_props%get_int(key_name, temp_int)) then
        do i_qty = 1, temp_int - 1
          REACT_(i_spec + i_qty) = REACT_(i_spec)
        end do
        i_spec = i_spec + temp_int - 1
      end if

      call reactants%iter_next()
      i_spec = i_spec + 1
    end do

    ! Get the indices and chemical properties for the products
    call products%iter_reset()
    i_spec = 1
    do while (products%get_key(spec_name))

      ! Save the index of this species in the state variable array
      PROD_(i_spec) = chem_spec_data%gas_state_id(spec_name)

      ! Make sure the species exists
      call assert_msg(342745995, PROD_(i_spec).gt.0, &
              "Missing VBS NOx dependent product: "//spec_name)

      ! Get properties included with this product in the reaction data
      call assert(641950524, products%get_property_t(val=spec_props))

      ! get high nox yield
      key_name = "high_nox_yield"
      call assert_msg(297262940, spec_props%get_real(key_name, HIGH_NOX_YIELD_(i_spec)), &
                  "Missing high_nox_yield for product: "//spec_name)

      ! get low nox yield
      key_name = "low_nox_yield"
      call assert_msg(307620235, spec_props%get_real(key_name, LOW_NOX_YIELD_(i_spec)), &
                  "Missing low_nox_yield for product: "//spec_name)


      call products%iter_next()
      i_spec = i_spec + 1
    end do

  ! get NO and HO2 indices needed for branching ratio calculations
    key_name = "NO_name"
    if (this%property_set%get_string(key_name, string_val)) then
      NO_ID_ = chem_spec_data%gas_state_id(trim(string_val))
    else
      NO_ID_ = chem_spec_data%gas_state_id("NO")
    endif

    key_name = "HO2_name"
    if (this%property_set%get_string(key_name, string_val)) then
      HO2_ID_ = chem_spec_data%gas_state_id(trim(string_val))
    else
      HO2_ID_ = chem_spec_data%gas_state_id("HO2")
    endif

  end subroutine initialize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Finalize the reaction
  elemental subroutine finalize(this)

    !> Reaction data
    type(rxn_vbs_nox_dependent_t), intent(inout) :: this

    if (associated(this%property_set)) &
            deallocate(this%property_set)
    if (allocated(this%condensed_data_real)) &
            deallocate(this%condensed_data_real)
    if (allocated(this%condensed_data_int)) &
            deallocate(this%condensed_data_int)

  end subroutine finalize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end module camp_rxn_vbs_nox_dependent