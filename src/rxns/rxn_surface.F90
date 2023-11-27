! Copyright (C) 2023 Barcelona Supercomputing Center, University of
! Illinois at Urbana-Champaign, and National Center for Atmospheric Research
! SPDX-License-Identifier: MIT

!> \file
!> The camp_rxn_surface module.

!> \page camp_rxn_surface CAMP: Surface (Hetergeneous) Reaction
!!
!! Surface reactions transform gas-phase species into gas-phase products
!! according to a rate that is calculated based on the total exposed surface
!! area of a condensed phase.
!!
!! For surface reactions of the gas phase species X, the reaction rate is
!! calculated assuming large particles (continuum regime) as:
!!
!! \f[
!!   r_{surface} = k_{surface}\mbox{[X]}
!! \f]
!!
!! where \f$[\mbox{X}]\f$ is the gas-phase concentration of species X [ppm]
!! and the rate constant \f$k_{surface}\f$ [1/s] is calculated as:
!!
!! \f[
!! k_{surface} = \frac{4N_a \pi r^2_e}{\left(\frac{r_e}{D_g} + \frac{4}{v(T)\gamma}\right)}
!! \f]
!!
!! where \f$N_a\f$ is the number concentration of particles
!! [particles\f$\mbox{m}^{-3}\f$], \f$r_e\f$ is the effective particle radius [m],
!! \f$D_g\f$ is the gas-phase diffusion coefficient of the reactant
!! [\f$\mbox{m}^2\mbox{s}^{-1}\f$], \f$\gamma\f$ is the reaction probability [unitless],
!! and v is the mean free speed of the gas-phase reactant:
!!
!! \f[
!!   v = \sqrt{\frac{8RT}{\pi MW}}
!! \f]
!!
!! where R is the ideal gas constant [\f$\mbox{J}\, \mbox{K}^{-1}\,
!! \mbox{mol}^{-1}\f$], T is temperature [K], and MW is the molecular weight of
!! the gas-phase reactant [\f$\mbox{kg}\, \mbox{mol}^{-1}\f$]
!!
!! Input data for surface reactions have the following format :
!! \code{.json}
!!   {
!!     "type" : "SURFACE",
!!     "gas-phase reactant" : "my gas species",
!!     "reaction probability" : 0.2,
!!     "gas-phase products" : {
!!        "my other gas species" : { },
!!        "another gas species" : { "yield" : 0.3 }
!!     },
!!     "areosol phase" : "my aqueous phase"
!!   }
!! \endcode
!! The key-value pairs \b gas-phase \b reactant, \b reaction \b probability,
!! and \b aerosol-phase are required.
!! Only one gas-phase reactant is allowed, but multiple products can be present.
!! The key-value pair \b yield for product species is optional and defaults to 1.0.
!!
!! The gas-phase reactant species must include properties
!! \b diffusion \b coeff \b [\b m2 \b s-1],
!! which specifies the diffusion coefficient in
!! \f$\mbox{m}^2\,\mbox{s}^{-1}\f$, and \b molecular \b weight
!! \b [\b kg \b mol-1], which specifies the molecular weight of the species in
!! \f$\mbox{kg}\,\mbox{mol}^{-1}\f$.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!> The rxn_surface_t type and associated functions.
module camp_rxn_surface

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

#define DIFF_COEFF_ this%condensed_data_real(1)
#define GAMMA_ this%condensed_data_real(2)
#define MW_ this%condensed_data_real(3)
#define NUM_AERO_PHASE_ this%condensed_data_int(1)
#define REACT_ID_ this%condensed_data_int(2)
#define NUM_PROD_ this%condensed_data_int(3)
#define NUM_INT_PROP_ 3
#define NUM_REAL_PROP_ 3
#define NUM_ENV_PARAM_ 1
#define PROD_ID_(x) this%condensed_data_int(NUM_INT_PROP_+x)
#define DERIV_ID_(x) this%condensed_data_int(NUM_INT_PROP_+NUM_PROD_+x)
#define JAC_ID_(x) this%condensed_data_int(NUM_INT_PROP_+1+2*NUM_PROD_+x)
#define PHASE_INT_LOC_(x) this%condensed_data_int(NUM_INT_PROP_+2+3*NUM_PROD_+x)
#define PHASE_REAL_LOC_(x) this%condensed_data_int(NUM_INT_PROP_+2+3*NUM_PROD_+NUM_AERO_PHASE_+x)
#define AERO_PHASE_ID_(x) this%condensed_data_int(PHASE_INT_LOC_(x))
#define AERO_REP_ID_(x) this%condensed_data_int(PHASE_INT_LOC_(x)+1)
#define NUM_AERO_PHASE_JAC_ELEM_(x) this%condensed_data_int(PHASE_INT_LOC_(x)+2)
#define PHASE_JAC_ID_(x,s,e) this%condensed_data_int(PHASE_INT_LOC_(x)+3+(s-1)*NUM_AERO_PHASE_JAC_ELEM_(x)+e)
#define YIELD_(x) this%condensed_data_real(NUM_REAL_PROP_+x)
#define EFF_RAD_JAC_ELEM_(x,e) this%condensed_data_real(PHASE_REAL_LOC_(x)+(e-1))
#define NUM_CONC_JAC_ELEM_(x,e) this%condensed_data_real(PHASE_REAL_LOC_(x)+NUM_AERO_PHASE_JAC_ELEM_(x)+(e-1))

  public :: rxn_surface_t

  !> Generic test reaction data type
  type, extends(rxn_data_t) :: rxn_surface_t
  contains
    !> Reaction initialization
    procedure :: initialize
    !> Finalize the reaction
    final :: finalize
  end type rxn_surface_t

  !> Constructor for rxn_surface_t
  interface rxn_surface_t
    procedure :: constructor
  end interface rxn_surface_t

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Constructor for surface reaction
  function constructor() result(new_obj)

    !> A new reaction instance
    type(rxn_surface_t), pointer :: new_obj

    allocate(new_obj)
    new_obj%rxn_phase = AERO_RXN

  end function constructor

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Initialize the reaction data, validating component data and loading
  !! any required information into the condensed data arrays for use during
  !! solving
  subroutine initialize(this, chem_spec_data, aero_rep, n_cells)

    !> Reaction data
    class(rxn_surface_t), intent(inout) :: this
    !> Chemical species data
    type(chem_spec_data_t), intent(in) :: chem_spec_data
    !> Aerosol representations
    type(aero_rep_data_ptr), pointer, intent(in) :: aero_rep(:)
    !> Number of grid cells to solve simultaneously
    integer(kind=i_kind), intent(in) :: n_cells

    type(property_t), pointer :: products, spec_props
    character(len=:), allocatable :: key_name, reactant_name, product_name, &
                                     phase_name, error_msg
    integer :: i_spec, n_aero_jac_elem, n_aero_phase, i_phase, i_aero_rep, &
               i_aero_id
    integer, allocatable :: phase_ids(:)
    real(kind=dp) :: temp_real

    if (.not. associated(this%property_set)) call die_msg(244070915, &
            "Missing property set needed to initialize surface reaction.")

    key_name = "gas-phase reactant"
    call assert_msg(807568174, &
            this%property_set%get_string(key_name, reactant_name), &
            "Missing gas-phase reactant name in surface reaction.")

    key_name = "gas-phase products"
    call assert_msg(285567904, &
            this%property_set%get_property_t(key_name, products), &
            "Missing gas-phase products for surface reaction.")

    key_name = "aerosol phase"
    call assert_msg(939211358, &
            this%property_set%get_string(key_name, phase_name), &
            "Missing aerosol phase in surface reaction.")

    error_msg = " for surface reaction of gas-phase species '"// &
                reactant_name//"' on aerosol phase '"//phase_name//"'"

    call assert(362731302, associated(aero_rep))
    call assert_msg(187310091, size(aero_rep) .gt. 0, &
            "Missing aerosol representation"//error_msg)

    ! Count the number of Jacobian elements needed in calculations of mass,
    ! volume, etc. and the number of instances of the aerosol phase
    n_aero_jac_elem = 0
    n_aero_phase = 0
    do i_aero_rep = 1, size(aero_rep)
      phase_ids = aero_rep(i_aero_rep)%val%phase_ids(phase_name)
      n_aero_phase = n_aero_phase + size(phase_ids)
      do i_phase = 1, size(phase_ids)
        n_aero_jac_elem = n_aero_jac_elem + &
            aero_rep(i_aero_rep)%val%num_jac_elem(phase_ids(i_phase))
      end do
    end do

    allocate(this%condensed_data_int(NUM_INT_PROP_             & ! NUM_AERO_PHASE, REACT_ID, NUM_PROD
                                     + 2 + 3 * products%size() & ! PROD_ID, DERIV_ID, JAC_ID
                                     + 2 * n_aero_phase        & ! PHASE_INT_LOC, PHASE_REAL_LOC
                                     + n_aero_phase * 3        & ! AERO_PHASE_ID, AERO_REP_ID, NUM_AERO_PHASE_JAC_ELEM
                                     + (1 + products%size()) * n_aero_jac_elem)) ! PHASE_JAC_ID
    allocate(this%condensed_data_real(NUM_REAL_PROP_           & ! DIFF_COEFF, GAMMA, MW
                                     + products%size()         & ! YIELD
                                     + 2 * n_aero_jac_elem))     ! EFF_RAD_JAC_ELEM, NUM_CONC_JAC_ELEM
    this%condensed_data_int(:) = 0_i_kind
    this%condensed_data_real(:) = 0.0_dp

    ! Save space for the environment-dependent parameters
    this%num_env_params = NUM_ENV_PARAM_

    NUM_AERO_PHASE_ = n_aero_phase

    key_name = "reaction probability"
    call assert_msg(388486564, &
            this%property_set%get_real(key_name, GAMMA_), &
            "Missing reaction probability for"//error_msg)

    ! Save the reactant information
    REACT_ID_ = chem_spec_data%gas_state_id(reactant_name)
    call assert_msg(908581300, REACT_ID_ .gt. 0, &
                    "Missing gas-phase species"//error_msg)
    call assert_msg(792904182, &
                    chem_spec_data%get_property_set(reactant_name, spec_props), &
                    "Missing gas-phase species properties"//error_msg)
    key_name = "molecular weight [kg mol-1]"
    call assert_msg(110823327, spec_props%get_real(key_name, MW_), &
                    "Missing molecular weight for gas-phase reactant"//error_msg)
    key_name = "diffusion coeff [m2 s-1]"
    call assert_msg(860403969, spec_props%get_real(key_name, DIFF_COEFF_), &
                    "Missing diffusion coefficient for gas-phase reactant"// &
                    error_msg)

    ! Save the product information
    NUM_PROD_ = products%size()
    call products%iter_reset()
    i_spec = 1
    do while (products%get_key(product_name))
      PROD_ID_(i_spec) = chem_spec_data%gas_state_id(product_name)
      call assert_msg(863839516, PROD_ID_(i_spec) .gt. 0, &
              "Missing surface reaction product: "//product_name)
      call assert(237691686, products%get_property_t(val=spec_props))
      YIELD_(i_spec) = 1.0_dp
      key_name = "yield"
      if (spec_props%get_real(key_name, temp_real)) YIELD_(i_spec) = temp_real
      call products%iter_next()
      i_spec = i_spec + 1
    end do

    ! Set aerosol phase specific indices
    i_aero_id = 1
    PHASE_INT_LOC_(i_aero_id) = NUM_INT_PROP_ + 2 + 3 * NUM_PROD_ + &
                                2 * NUM_AERO_PHASE_ + 1
    PHASE_REAL_LOC_(i_aero_id) = NUM_REAL_PROP_ + NUM_PROD_ + 1
    do i_aero_rep = 1, size(aero_rep)
      phase_ids = aero_rep(i_aero_rep)%val%phase_ids(phase_name)
      do i_phase = 1, size(phase_ids)
        NUM_AERO_PHASE_JAC_ELEM_(i_aero_id) = &
            aero_rep(i_aero_rep)%val%num_jac_elem(phase_ids(i_phase))
        AERO_PHASE_ID_(i_aero_id) = phase_ids(i_phase)
        AERO_REP_ID_(i_aero_id) = i_aero_rep
        i_aero_id = i_aero_id + 1
        if (i_aero_id .le. NUM_AERO_PHASE_) then
          PHASE_INT_LOC_(i_aero_id)  = PHASE_INT_LOC_(i_aero_id - 1) + 3 + &
                                       (1 + NUM_PROD_) * &
                                       NUM_AERO_PHASE_JAC_ELEM_(i_aero_id - 1)
          PHASE_REAL_LOC_(i_aero_id) = PHASE_REAL_LOC_(i_aero_id - 1) + &
                                   2 * NUM_AERO_PHASE_JAC_ELEM_(i_aero_id - 1)
        end if
      end do
    end do

  end subroutine initialize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Finalize the reaction
  elemental subroutine finalize(this)

    !> Reaction data
    type(rxn_surface_t), intent(inout) :: this

    if (associated(this%property_set)) &
            deallocate(this%property_set)
    if (allocated(this%condensed_data_real)) &
            deallocate(this%condensed_data_real)
    if (allocated(this%condensed_data_int)) &
            deallocate(this%condensed_data_int)

  end subroutine finalize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end module camp_rxn_surface
