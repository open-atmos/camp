! Copyright (C) 2021 Barcelona Supercomputing Center and University of
! Illinois at Urbana-Champaign
! SPDX-License-Identifier: MIT

!> \file
!> The camp_rxn_isomerisation module.

!> \page camp_rxn_isomerisation CAMP: Isomerisation Reaction
!!
!! Isomerisation Arrhenius-like reaction rate constant equations are calculated as follows:
!!
!! \f[
!!   Ae^{(\frac{-E_a}{k_bT})} * (a_0 + a_1*T + a_2*T^2 + a_3*T^3 + a_4*T^4)
!! \f]
!!
!! where \f$A\f$ is the pre-exponential factor
!! (# \f$(\mbox{cm}^{-3})^{-(n-1)}\mbox{s}^{-1}\f$),
!! \f$n\f$ is the number of reactants, \f$E_a\f$ is the activation energy (J),
!! \f$k_b\f$ is the Boltzmann constant (J/K), \f$D\f$ (K), \f$B\f$ (unitless)
!! and \f$E\f$ (\f$Pa^{-1}\f$) are reaction parameters, \f$T\f$ is the
!! temperature (K), and \f$P\f$ is the pressure (Pa). The first two terms are
!! described in Finlayson-Pitts and Pitts (2000) \cite Finlayson-Pitts2000 .
!! the a_i terms are the temperature dependence of the isomerisation reaction
!!
!! Input data for Arrhenius equations has the following format:
!! \code{.json}
!!   {
!!     "type" : "ISOMERISATION",
!!     "A" : 123.45,
!!     "Ea" : 123.45,
!!     "a0" : 1.0,
!!     "a1" : 1.2,
!!     "a2" : 1.2,
!!     "a3" : 1.2,
!!     "a4" : 1.2,
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
!! Optionally, a parameter \b C may be included, and is taken to equal
!! \f$\frac{-E_a}{k_b}\f$. Note that either \b Ea or \b C may be included, but
!! not both. When neither \b Ea or \b C are included, they are assumed to be
!! 0.0. When \b A is not included, it is assumed to be 1.0, when \b D is not
!! included, it is assumed to be 300.0 K, when \b B is not included, it is
!! assumed to be 0.0, and when \b E is not included, it is assumed to be 0.0.
!! The unit for time is assumed to be s, but inclusion of the optional
!! key-value pair \b time \b unit = \b MIN can be used to indicate a rate
!! with min as the time unit.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!> The rxn_isomerisation_t type and associated functions.
module camp_rxn_isomerisation

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
#define A_ this%condensed_data_real(1)
#define C_ this%condensed_data_real(2)
#define A0_ this%condensed_data_real(3)
#define A1_ this%condensed_data_real(4)
#define A2_ this%condensed_data_real(5)
#define A3_ this%condensed_data_real(6)
#define A4_ this%condensed_data_real(7)
#define CONV_ this%condensed_data_real(8)
#define NUM_INT_PROP_ 2
#define NUM_REAL_PROP_ 8
#define NUM_ENV_PARAM_ 1
#define REACT_(x) this%condensed_data_int(NUM_INT_PROP_ + x)
#define PROD_(x) this%condensed_data_int(NUM_INT_PROP_ + NUM_REACT_ + x)
#define DERIV_ID_(x) this%condensed_data_int(NUM_INT_PROP_ + NUM_REACT_ + NUM_PROD_ + x)
#define JAC_ID_(x) this%condensed_data_int(NUM_INT_PROP_ + 2*(NUM_REACT_+NUM_PROD_) + x)
#define YIELD_(x) this%condensed_data_real(NUM_REAL_PROP_ + x)

   public :: rxn_isomerisation_t

   !> Generic test reaction data type
   type, extends(rxn_data_t) :: rxn_isomerisation_t
   contains
      !> Reaction initialization
      procedure :: initialize
      !> Finalize the reaction
      final :: finalize
   end type rxn_isomerisation_t

   !> Constructor for rxn_isomerisation_t
   interface rxn_isomerisation_t
      procedure :: constructor
   end interface rxn_isomerisation_t

contains

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   !> Constructor for Isomerisation reaction
   function constructor() result(new_obj)

      !> A new reaction instance
      type(rxn_isomerisation_t), pointer :: new_obj

      allocate(new_obj)
      new_obj%rxn_phase = GAS_RXN

   end function constructor

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   !> Initialize the reaction data, validating component data and loading
   !! any required information into the condensed data arrays for use during
   !! solving
   subroutine initialize(this, chem_spec_data, aero_rep, n_cells)

      !> Reaction data
      class(rxn_isomerisation_t), intent(inout) :: this
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
      if (.not. associated(this%property_set)) call die_msg(45489501, &
         "Missing property set needed to initialize reaction")
      key_name = "reactants"
      call assert_msg(85758899, &
         this%property_set%get_property_t(key_name, reactants), &
         "Isomerisation reaction is missing reactants")
      key_name = "products"
      call assert_msg(210750401, &
         this%property_set%get_property_t(key_name, products), &
         "Isomerisation reaction is missing products")

      ! Count the number of reactants (including those with a qty specified)
      call reactants%iter_reset()
      i_spec = 0
      do while (reactants%get_key(spec_name))
         ! Get properties included with this reactant in the reaction data
         call assert(332897359, reactants%get_property_t(val=spec_props))
         key_name = "qty"
         if (spec_props%get_int(key_name, temp_int)) i_spec = i_spec+temp_int-1
         call reactants%iter_next()
         i_spec = i_spec + 1
      end do

      ! Allocate space in the condensed data arrays
      allocate(this%condensed_data_int(NUM_INT_PROP_ + &
         (i_spec + 2) * (i_spec + products%size())))
      allocate(this%condensed_data_real(NUM_REAL_PROP_ + products%size()))
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
         call assert_msg(297370315, &
            .not.this%property_set%get_real(key_name, temp_real), &
            "Received both Ea and C parameter for Arrhenius equation")
      else
         key_name = "C"
         if (.not. this%property_set%get_real(key_name, C_)) then
            C_ = 0.0
         end if
      end if

      key_name = "a0"
      call assert_msg(17690471, &
         this%property_set%get_real(key_name, A0_), &
         "Isomerisation reaction is missing a0")

      key_name = "a1"
      call assert_msg(274494883, &
         this%property_set%get_real(key_name, A1_), &
         "Isomerisation reaction is missing a1")

      key_name = "a2"
      call assert_msg(734574038, &
         this%property_set%get_real(key_name, A2_), &
         "Isomerisation reaction is missing a2")

      key_name = "a3"
      call assert_msg(415871880, &
         this%property_set%get_real(key_name, A3_), &
         "Isomerisation reaction is missing a3")

      key_name = "a4"
      call assert_msg(152837294, &
         this%property_set%get_real(key_name, A4_), &
         "Isomerisation reaction is missing a4")

      ! Get the indices and chemical properties for the reactants
      call reactants%iter_reset()
      i_spec = 1
      do while (reactants%get_key(spec_name))

         ! Save the index of this species in the state variable array
         REACT_(i_spec) = chem_spec_data%gas_state_id(spec_name)

         ! Make sure the species exists
         call assert_msg(151676256, REACT_(i_spec).gt.0, &
            "Missing Isomerisation reactant: "//spec_name)

         ! Get properties included with this reactant in the reaction data
         call assert(111386982, reactants%get_property_t(val=spec_props))
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
         call assert_msg(299189042, PROD_(i_spec).gt.0, &
            "Missing Isomerisation product: "//spec_name)

         ! Get properties included with this product in the reaction data
         call assert(27497013, products%get_property_t(val=spec_props))
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
      type(rxn_isomerisation_t), intent(inout) :: this

      if (associated(this%property_set)) &
         deallocate(this%property_set)
      if (allocated(this%condensed_data_real)) &
         deallocate(this%condensed_data_real)
      if (allocated(this%condensed_data_int)) &
         deallocate(this%condensed_data_int)

   end subroutine finalize

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end module camp_rxn_isomerisation
