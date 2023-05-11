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
#define PRE_C_AVG_ this%condensed_data_real(2)
#define GAMMA_ this%condensed_data_real(3)
#define MW_ this%condensed_data_real(4)
#define NUM_AERO_PHASE_ this%condensed_data_int(1)
#define GAS_SPEC_ this%condensed_data_int(2)
#define NUM_PROD_ this%condensed_data_int(3)
#define NUM_INT_PROP_ 3
#define NUM_REAL_PROP_ 4
#define NUM_ENV_PARAM_ 1
#define DERIV_ID_(x) this%condensed_data_int(NUM_INT_PROP_+x)
#define JAC_ID_(x) this%condensed_data_int(NUM_INT_PROP_+1+NUM_PROD_+x)
#define PHASE_INT_LOC_(x) this%condensed_data_int(NUM_INT_PROP_+2+2*NUM_PROD_+x)
#define PHASE_REAL_LOC_(x) this%condensed_data_int(NUM_INT_PROP_+2+2*NUM_PROD_+NUM_AERO_PHASE_+x)
#define AERO_PHASE_ID_(x) this%condensed_data_int(PHASE_INT_LOC_(x))
#define AERO_REP_ID_(x) this%condensed_data_int(PHASE_INT_LOC_(x)+1)
#define NUM_AERO_PHASE_JAC_ELEM_(x) this%condensed_data_int(PHASE_INT_LOC_(x)+2)
#define PHASE_JAC_ID_(x,s,e) this%condensed_data_int(PHASE_INT_LOC_(x)+2+(s-1)*NUM_AERO_PHASE_JAC_ELEM_(x)+e)
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

    ! Allocate space in the condensed data arrays
    allocate(this%condensed_data_int(0))
    allocate(this%condensed_data_real(0))

    ! Save space for the environment-dependent parameters
    this%num_env_params = NUM_ENV_PARAM_

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
