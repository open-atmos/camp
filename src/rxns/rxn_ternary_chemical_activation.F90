! Copyright (C) 2021 Barcelona Supercomputing Center,
!  University of Illinois at Urbana-Champaign, and
!  National Center for Atmospheric Research
! SPDX-License-Identifier: MIT

!> \file
!> The camp_rxn_ternary_chemical_activation module.

!> \page camp_rxn_ternary_chemical_activation CAMP: Ternary Chemical Activation Reaction
!!
!! Ternary Chemical Activation reaction rate constant equations take the form:
!!
!! \f[
!!   \frac{k_0}{1+k_0[\mbox{M}]/k_{\inf}}F_C^{(1+1/N[log_{10}(k_0[\mbox{M}]/k_{\inf})]^2)^{-1}}
!! \f]
!!
!! where \f$k_0\f$ is the low-pressure limiting rate constant, \f$k_{\inf}\f$
!! is the high-pressure limiting rate constant, \f$[\mbox{M}]\f$ is the
!! density of air, and \f$F_C\f$ and \f$N\f$ are parameters
!! that determine the shape of the fall-off curve, and are typically 0.6 and
!! 1.0, respectively \cite JPL15. \f$k_0\f$ and
!! \f$k_{\inf}\f$ are calculated as \ref camp_rxn_arrhenius "Arrhenius" rate
!! constants with \f$D=300\f$ and \f$E=0\f$.
!!
!! Input data for Ternary Chemical Activation reactions have the following format :
!! \code{.json}
!!   {
!!     "type" : "TERNARY_CHEMICAL_ACTIVATION",
!!     "k0_A" : 5.6E-12,
!!     "k0_B" : -1.8,
!!     "k0_C" : 180.0,
!!     "kinf_A" : 3.4E-12,
!!     "kinf_B" : -1.6,
!!     "kinf_C" : 104.1,
!!     "Fc"  : 0.7,
!!     "N"  : 0.9,
!!     "time unit" : "MIN",
!!     "reactants" : {
!!       "spec1" : {},
!!       "spec2" : { "qty" : 2 },
!!       ...
!!     },
!!     "products" : {
!!       "spec3" : {},
!!       "spec4" : { "yield" : 0.65 },
!!       ...
!!     }
!!   }
!! \endcode
!! The key-value pairs \b reactants, and \b products are required. Reactants
!! without a \b qty value are assumed to appear once in the reaction equation.
!! Products without a specified \b yield are assumed to have a \b yield of
!! 1.0.
!!
!! The two sets of parameters beginning with \b k0_ and \b kinf_ are the
!! \ref camp_rxn_arrhenius "Arrhenius" parameters for the \f$k_0\f$ and
!! \f$k_{\inf}\f$ rate constants, respectively. When not present, \b _A
!! parameters are assumed to be 1.0, \b _B to be 0.0, \b _C to be 0.0, \b Fc
!! to be 0.6 and \b N to be 1.0.
!!
!! The unit for time is assumed to be s, but inclusion of the optional
!! key-value pair \b time \b unit = \b MIN can be used to indicate a rate
!! with min as the time unit.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!> The rxn_ternary_chemical_activation_t type and associated functions.
module camp_rxn_ternary_chemical_activation

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
#define K0_A_ this%condensed_data_real(1)
#define K0_B_ this%condensed_data_real(2)
#define K0_C_ this%condensed_data_real(3)
#define KINF_A_ this%condensed_data_real(4)
#define KINF_B_ this%condensed_data_real(5)
#define KINF_C_ this%condensed_data_real(6)
#define FC_ this%condensed_data_real(7)
#define N_ this%condensed_data_real(8)
#define SCALING_ this%condensed_data_real(9)
#define CONV_ this%condensed_data_real(10)
#define NUM_INT_PROP_ 2
#define NUM_REAL_PROP_ 10
#define NUM_ENV_PARAM_ 1
#define REACT_(x) this%condensed_data_int(NUM_INT_PROP_ + x)
#define PROD_(x) this%condensed_data_int(NUM_INT_PROP_ + NUM_REACT_ + x)
#define DERIV_ID_(x) this%condensed_data_int(NUM_INT_PROP_ + NUM_REACT_ + NUM_PROD_ + x)
#define JAC_ID_(x) this%condensed_data_int(NUM_INT_PROP_ + 2*(NUM_REACT_+NUM_PROD_) + x)
#define YIELD_(x) this%condensed_data_real(NUM_REAL_PROP_ + x)

public :: rxn_ternary_chemical_activation_t

  !> Generic test reaction data type
  type, extends(rxn_data_t) :: rxn_ternary_chemical_activation_t
  contains
    !> Reaction initialization
    procedure :: initialize
    !> Finalize the reaction
    final :: finalize
  end type rxn_ternary_chemical_activation_t

  !> Constructor for rxn_ternary_chemical_activation_t
  interface rxn_ternary_chemical_activation_t
    procedure :: constructor
  end interface rxn_ternary_chemical_activation_t

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Constructor for Ternary Chemical Activation reaction
  function constructor() result(new_obj)

    !> A new reaction instance
    type(rxn_ternary_chemical_activation_t), pointer :: new_obj

    allocate(new_obj)
    new_obj%rxn_phase = GAS_RXN

  end function constructor

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Initialize the reaction data, validating component data and loading
  !! any required information into the condensed data arrays for use during
  !! solving
  subroutine initialize(this, chem_spec_data, aero_rep, n_cells)

    !> Reaction data
    class(rxn_ternary_chemical_activation_t), intent(inout) :: this
    !> Chemical species data
    type(chem_spec_data_t), intent(in) :: chem_spec_data
    !> Aerosol representations
    type(aero_rep_data_ptr), pointer, intent(in) :: aero_rep(:)
    !> Number of grid cells to solve simultaneously
    integer(kind=i_kind), intent(in) :: n_cells

    type(property_t), pointer :: spec_props, reactants, products
    character(len=:), allocatable :: key_name, spec_name, string_val
    integer(kind=i_kind) :: i_spec, i_qty

    integer(kind=i_kind) :: temp_int
    real(kind=dp) :: temp_real

    ! Get the species involved
    if (.not. associated(this%property_set)) call die_msg(138420387, &
            "Missing property set needed to initialize reaction")
    key_name = "reactants"
    call assert_msg(250738732, &
            this%property_set%get_property_t(key_name, reactants), &
            "Ternary Chemical Activation reaction is missing reactants")
    key_name = "products"
    call assert_msg(980581827, &
            this%property_set%get_property_t(key_name, products), &
            "Ternary Chemical Activation reaction is missing products")

    ! Count the number of reactants (including those with a qty specified)
    call reactants%iter_reset()
    i_spec = 0
    do while (reactants%get_key(spec_name))
      ! Get properties included with this reactant in the reaction data
      call assert(475375422, reactants%get_property_t(val=spec_props))
      key_name = "qty"
      if (spec_props%get_int(key_name, temp_int)) i_spec = i_spec+temp_int-1
      call reactants%iter_next()
      i_spec = i_spec + 1
    end do

    ! Allocate space in the condensed data arrays
    ! Space in this example is allocated for two sets of inidices for the
    ! reactants and products, one molecular property for each reactant,
    ! yields for the products and three reaction parameters.
    allocate(this%condensed_data_int(NUM_INT_PROP_ + &
            (i_spec + 2) * (i_spec + products%size())))
    allocate(this%condensed_data_real(NUM_REAL_PROP_ + products%size()))
    this%condensed_data_int(:) = int(0, kind=i_kind)
    this%condensed_data_real(:) = real(0.0, kind=dp)

    ! Save space for the environment-dependent parameters
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
    key_name = "k0_A"
    if (.not. this%property_set%get_real(key_name, K0_A_)) then
      K0_A_ = 1.0
    end if
    key_name = "k0_B"
    if (.not. this%property_set%get_real(key_name, K0_B_)) then
      K0_B_ = 0.0
    end if
    key_name = "k0_C"
    if (.not. this%property_set%get_real(key_name, K0_C_)) then
      K0_C_ = 0.0
    end if
    key_name = "kinf_A"
    if (.not. this%property_set%get_real(key_name, KINF_A_)) then
      KINF_A_ = 1.0
    end if
    key_name = "kinf_B"
    if (.not. this%property_set%get_real(key_name, KINF_B_)) then
      KINF_B_ = 0.0
    end if
    key_name = "kinf_C"
    if (.not. this%property_set%get_real(key_name, KINF_C_)) then
      KINF_C_ = 0.0
    end if
    key_name = "Fc"
    if (.not. this%property_set%get_real(key_name, FC_)) then
      FC_ = 0.6
    end if
    key_name = "N"
    if (.not. this%property_set%get_real(key_name, N_)) then
      N_ = 1.0
    end if
    key_name = "time unit"
    SCALING_ = real(1.0, kind=dp)
    if (this%property_set%get_string(key_name, string_val)) then
      if (trim(string_val).eq."MIN") then
        SCALING_ = real(1.0d0/60.0d0, kind=dp)
      end if
    endif

    ! Include [M] in k0_A
    K0_A_ = K0_A_ * real(1.0d6, kind=dp)

    ! Get the indices and chemical properties for the reactants
    call reactants%iter_reset()
    i_spec = 1
    do while (reactants%get_key(spec_name))

      ! Save the index of this species in the state variable array
      REACT_(i_spec) = chem_spec_data%gas_state_id(spec_name)

      ! Make sure the species exists
      call assert_msg(194805707, REACT_(i_spec).gt.0, &
              "Missing Ternary Chemical Activation reactant: "//spec_name)

      ! Get properties included with this reactant in the reaction data
      call assert(524590901, reactants%get_property_t(val=spec_props))
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
      call assert_msg(249285493, PROD_(i_spec).gt.0, &
              "Missing Ternary Chemical Activation product: "//spec_name)

      ! Get properties included with this product in the reaction data
      call assert(296595438, products%get_property_t(val=spec_props))
      key_name = "yield"
      if (spec_props%get_real(key_name, temp_real)) then
        YIELD_(i_spec) = temp_real
      else
        YIELD_(i_spec) = 1.0
      end if

      call products%iter_next()
      i_spec = i_spec + 1
    end do

  end subroutine initialize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Finalize the reaction
  elemental subroutine finalize(this)

    !> Reaction data
    type(rxn_ternary_chemical_activation_t), intent(inout) :: this

    if (associated(this%property_set)) &
            deallocate(this%property_set)
    if (allocated(this%condensed_data_real)) &
            deallocate(this%condensed_data_real)
    if (allocated(this%condensed_data_int)) &
            deallocate(this%condensed_data_int)

  end subroutine finalize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end module camp_rxn_ternary_chemical_activation
