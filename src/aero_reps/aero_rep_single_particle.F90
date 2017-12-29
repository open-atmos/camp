! Copyright (C) 2017 Matt Dawson
! Licensed under the GNU General Public License version 2 or (at your
! option) any later version. See the file COPYING for details.

!> \file
!> The pmc_aero_rep_single_particle module.

!> The abstract aero_rep_single_particle_t structure and associated subroutines.
module pmc_aero_rep_single_particle

  use pmc_util,                                  only: dp, i_kind, &
                                                       string_t, assert_msg, &
                                                       die_msg
  use pmc_property
  use pmc_chem_spec_data
  use pmc_aero_rep_data
  use pmc_aero_phase_data
  use pmc_aero_rep_state
  use pmc_aero_rep_single_particle_state
  use pmc_phlex_state

  implicit none
  private

#define _NUM_PHASE_ this%condensed_data_int(1)
#define _AERO_STATE_ID_ this%condensed_data_int(2)
#define _NUM_INT_PROP_ 2
#define _NUM_REAL_PROP_ 0
#define _PHASE_STATE_ID_(x) this%condensed_data_int(_NUM_INT_PROP_ + x)
#define _SPEC_STATE_ID_(y,x) _PHASE_STATE_ID_(y) + x - 1
#define _PHASE_SPEC_ID_(x) this%condensed_data_int(_NUM_INT_PROP_ + _NUM_PHASE_ + x)
#define _NUM_SPEC_(x) this%condensed_data_int(_NUM_INT_PROP_ + (_NUM_PHASE_)*2 + x)
#define _DENSITY_(y,x) this%condensed_data_real(_PHASE_SPEC_ID_(y) + x - 1)
#define _MASS_(y,x) phlex_state%state_var(_PHASE_STATE_ID_(y) + x - 1)
  
  public :: aero_rep_single_particle_t

  !> Single particle aerosol representation
  !!
  !! Time-invariant data related to a single particle aerosol representation. 
  type, extends(aero_rep_data_t) :: aero_rep_single_particle_t
  contains
    !> Aerosol representation initialization
    procedure :: initialize
    !> Get the size of this representation on the state variable array
    procedure :: size => get_size
    !> Get a list of unique names for each element on the state array
    procedure :: unique_names
    !> Get a species state id by its unique name
    procedure :: state_id_by_unique_name
    !> Get an instance of the state variable for this aerosol representation
    procedure :: new_state
    !> Get an aerosol species state id
    procedure :: species_state_id
    !> Get the surface area concentration (m^2/m^3)
    procedure :: surface_area_conc
    !> Get the surface area concentration for a specific aerosol species
    !! (m^2/m^3)
    procedure :: species_surface_area_conc
    !> Get the vapor pressure scaling for a particular species (unitless)
    procedure :: kelvin_effect

    !> Private functions
    !> Get the associated aero_rep_state_t variable
    procedure, private :: get_state => get_state
  end type aero_rep_single_particle_t

  !> Constructor for aero_rep_single_particle_t
  interface aero_rep_single_particle_t
    procedure :: constructor
  end interface aero_rep_single_particle_t

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Constructor for aero_rep_single_particle_t
  function constructor() result (new_obj)

    !> New aerosol representation
    type(aero_rep_single_particle_t), pointer :: new_obj

    allocate(new_obj)

  end function constructor

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Initialize the aerosol representation data, validating component data and
  !! loading any required information from the  property_t object. This 
  !! routine should be called once for each aerosol representation
  !! at the beginning of a model run after all the input files have been
  !! read in. It ensures all data required during the model run are included
  !! in the condensed data arrays.
  subroutine initialize(this, aero_phase_set, &
                  spec_state_id, aero_state_id, chem_spec_data)

    !> Aerosol representation data
    class(aero_rep_single_particle_t), intent(inout) :: this
    !> The set of aerosol phases
    type(aero_phase_data_ptr), pointer, intent(in) :: aero_phase_set(:)
    !> Beginning state id for this aerosol representationin the model species
    !! state array
    integer(kind=i_kind), intent(in) :: spec_state_id
    !> Index for this representation in the model state aero_rep_state_t
    !! array
    integer(kind=i_kind), intent(in) :: aero_state_id
    !> Chemical species data
    type(chem_spec_data_t), intent(in) :: chem_spec_data

    integer(kind=i_kind) :: i_phase, i_spec, curr_id, curr_spec_id, num_spec
    type(property_t), pointer :: spec_props
    type(string_t), allocatable :: species(:)
    real(kind=dp) :: density
    character(len=:), allocatable :: key

    ! Assume all phases will be applied to each particle
    allocate(this%aero_phase(size(aero_phase_set)))
    do i_phase = 1, size(aero_phase_set)
      allocate(this%aero_phase(i_phase)%val)
      this%aero_phase(i_phase)%val = aero_phase_set(i_phase)%val
    end do

    ! Get the total number of species across all phases
    num_spec = 0
    do i_phase = 1, size(this%aero_phase)
      num_spec = num_spec + this%aero_phase(i_phase)%val%size()
    end do

    ! Allocate condensed data arrays
    allocate(this%condensed_data_int(_NUM_INT_PROP_ + 3*size(this%aero_phase)))
    allocate(this%condensed_data_real(_NUM_REAL_PROP_ + num_spec))

    ! Set indexes
    _NUM_PHASE_ = size(this%aero_phase)
    _AERO_STATE_ID_ = aero_state_id
    curr_id = spec_state_id
    curr_spec_id = _NUM_REAL_PROP_ + 1
    do i_phase = 1, _NUM_PHASE_
      _PHASE_STATE_ID_(i_phase) = curr_id
      _PHASE_SPEC_ID_(i_phase) = curr_spec_id
      _NUM_SPEC_(i_phase) = this%aero_phase(i_phase)%val%size()
      curr_spec_id = curr_spec_id + this%aero_phase(i_phase)%val%size()
      curr_id = curr_id + this%aero_phase(i_phase)%val%size()
    end do

    ! Set densities
    key = "density"
    do i_phase = 1, _NUM_PHASE_
      curr_spec_id = 1
      species = this%aero_phase(i_phase)%val%get_species()
      do i_spec = 1, size(species)
        spec_props = chem_spec_data%get_property_set(species(i_spec)%string)
        if (.not.associated(spec_props)) then
          call die_msg(204001989, "Missing properties for species "// &
                  species(i_spec)%string)
        end if
        if (.not.spec_props%get_real(key, density)) then
          call die_msg(532333944, "Missing density for species "// &
                  species(i_spec)%string)
        end if
        _DENSITY_(i_phase, curr_spec_id) = density
        curr_spec_id = curr_spec_id + 1
      end do
    end do

  end subroutine initialize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Get the size of this aerosol representation on the state variable array
  function get_size(this) result (state_size)

    !> Size on the state array
    integer(kind=i_kind) :: state_size
    !> Aerosol representation data
    class(aero_rep_single_particle_t), intent(in) :: this

    integer(kind=i_kind) :: i_phase

    ! Get the total number of species across all phases
    state_size = 0
    do i_phase = 1, size(this%aero_phase)
      state_size = state_size + this%aero_phase(i_phase)%val%size()
    end do

  end function get_size

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Get a list of unique names for each species on the state array
  function unique_names(this)

    !> List of unique names
    type(string_t), allocatable :: unique_names(:)
    !> Aerosol representation data
    class(aero_rep_single_particle_t), intent(in) :: this

    integer(kind=i_kind) :: num_spec, i_spec, j_spec, i_phase
    type(string_t), allocatable :: spec_names(:)
    character(len=:), allocatable :: phase_name
    
    num_spec = 0
    do i_phase = 1, size(this%aero_phase)
      num_spec = num_spec + this%aero_phase(i_phase)%val%size()
    end do
    allocate(unique_names(num_spec))
    i_spec = 1
    do i_phase = 1, size(this%aero_phase)
      phase_name = this%aero_phase(i_phase)%val%name()
      num_spec = this%aero_phase(i_phase)%val%size()
      spec_names = this%aero_phase(i_phase)%val%get_species()
      do j_spec = 1, num_spec
        unique_names(i_spec + j_spec - 1)%string = &
                phase_name//'.'//spec_names(j_spec)%string
      end do
      i_spec = i_spec + num_spec
    end do

  end function unique_names

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Get a species state id by its unique name
  function state_id_by_unique_name(this, unique_name) result (spec_id)

    !> Species state id
    integer(kind=i_kind) :: spec_id
    !> Aerosol representation data
    class(aero_rep_single_particle_t), intent(in) :: this
    !> Unique name
    character(len=:), allocatable, intent(in) :: unique_name

    type(string_t), allocatable :: unique_names(:)
    integer(kind=i_kind) :: i_spec

    spec_id = 0
    unique_names = this%unique_names()
    do i_spec = 1, size(unique_names)
      if (unique_names(i_spec)%string .eq. unique_name) then
        spec_id = _PHASE_STATE_ID_(1) + i_spec - 1
        return
      end if
    end do

  end function state_id_by_unique_name

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Get an instance of the state type for this aerosol representation
  !!
  !! For PartMC single particle runs, the aerosol state will be set by PartMC
  !! at the beginning of each chemistry integration
  function new_state(this) result (aero_rep_state)

    !> Aerosol representation state
    class(aero_rep_state_t), pointer :: aero_rep_state
    !> Aerosol representaiton data
    class(aero_rep_single_particle_t), intent(in) :: this

    ! Empty state variable
    aero_rep_state => aero_rep_single_particle_state_t()

  end function new_state

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

 !> Get an aerosol species state id(s). The returned array will have the
 !! index of the species in the specified phase in each aerosol group (e.g.,
 !! bin, mode). If the species is not present in a certain group-phase the
 !! index will be 0.
 !!
 !! This function should only be called during initialization.
 !!
 !! One of each aerosol phase exists for each single particle, so this array
 !! will always be of length 1.
 function species_state_id(this, phase_name, species_name) result(spec_index)
    use pmc_util,                                     only : i_kind

    !> Species index array
    integer(kind=i_kind), allocatable :: spec_index(:)
    !> Aerosol representation data
    class(aero_rep_single_particle_t), intent(in) :: this
    !> Aerosol phase id
    character(len=:), allocatable, intent(in) :: phase_name
    !> Species name
    character(len=:), allocatable, intent(in) :: species_name

    integer(kind=i_kind) :: i_phase, i_spec

    allocate(spec_index(1))
    i_phase = this%phase_id(phase_name)
    call assert_msg(820720954, i_phase.gt.0, "Invalid phase requested: "// &
            phase_name)
    i_spec = this%aero_phase(i_phase)%val%state_id(species_name)
    call assert_msg(413945507, i_spec.gt.0, "Invalid species requested: "// &
            species_name//" for phase: "//phase_name)
    spec_index(1) = _PHASE_STATE_ID_(i_phase) + i_spec - 1

  end function species_state_id

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Get surface area concentration (m^2/m^3) between two phases. One phase
  !! may be set to 0 to indicate the gas-phase, or both phases may be aerosol
  !! phases, with a corresponding index.
  !!
  !! For single particles, currently only gas-aerosol interfaces are solved
  !! for. Because species are tracked by mass, the gas-aerosol surface area
  !! density for a single phase (assuming each phase has an exposed surface
  !! proportional to its fractional aerosol volume) is:
  !!
  !!   \f$A_P = \frac{V_P}{r_T}\f$
  !!
  !! where A is the exposed surface, V is the volume, r is the radius of the 
  !! particle, T indicates the total set of aerosol species and P indicates
  !! the aerosol species within a particular phase. The radius and volumes are
  !! calculated as:
  !!
  !!   \f$r_T = (\frac{3V_T}{4\pi})^(\frac{1}{3})\f$
  !!   \f$V_{T/P} = \sum_x \rho_x m_x \forall x \in T or P\f$
  !!
  !! where \f$rho_x\f$ is the density of species x and \f$m_x\f$ is its mass.
  !! The partial derivative of \f$A_P\f$ with respect to a single aerosol 
  !! species is:
  !!
  !!   \f$\frac{dA_P}{dx} = \frac{(3r_T\frac{dV_P}{dx} - 
  !!       3V_P\frac{dr_T}{dx})}{r_T^2}\f$
  !!   \f$\frac{dV_P}{dx} = \rho_x\ \forall x \in P\f$
  !!   \f$\frac{dr_T}{dx} = \frac{1}{3}\(\frac{3V_T}{4\pi}\)^\frac{1}{3} 
  !!       \rho_x\ \forall x \in T\f$
  !!
  function surface_area_conc(this, i_phase1, i_phase2, phlex_state, &
                  jac_contrib)
    use pmc_util,                                     only : dp
    use pmc_constants

    !> Surface area concentration
    real(kind=dp) :: surface_area_conc
    !> Aerosol representation data
    class(aero_rep_single_particle_t), intent(in) :: this
    !> Aerosol phase 1 id
    integer(kind=i_kind), intent(in) :: i_phase1
    !> Aerosol phase 2 id
    integer(kind=i_kind), intent(in) :: i_phase2
    !> Model state
    type(phlex_state_t), intent(in) :: phlex_state
    !> Contribution to Jacobian matrix. An array of the same size as the
    !! state array that, when present, will be filled with the partial
    !! derivatives of the result of this calculation with each state
    !! variable.
    real(kind=dp), allocatable, intent(inout), optional :: jac_contrib(:)

    real(kind=dp) :: v_t, v_p, r_t, temp_jac
    integer(kind=i_kind) :: i_phase, j_phase, i_spec

    ! Currently, only gas-aerosol interfaces are supported
    if (i_phase1.ne.0 .and. i_phase2.ne.0) then
      call die_msg(768138678, "Internal aerosol interface surface area not "//&
              "currently available.")
    end if
    i_phase = i_phase1 + i_phase2

    ! Calculate phase and total volume denisty (m^3/m^3)
    v_t = 0.0
    v_p = 0.0
    do j_phase = 1, _NUM_PHASE_
      do i_spec = 1, _NUM_SPEC_(j_phase)
        if (j_phase.eq.i_phase) v_p = v_p + &
               _MASS_(j_phase, i_spec) * &
               _DENSITY_(j_phase, i_spec)
        v_t = v_t + _MASS_(j_phase, i_spec) * &
               _DENSITY_(j_phase, i_spec)
      end do 
    end do

    ! Calculate radius (m)
    r_t = (3.0/4.0*v_t/const%pi)**(1.0/3.0)

    ! Calculate the surface area density (m^2/m^3)
    surface_area_conc = 3.0 * v_p / r_t

    write(*,*) v_p, v_t, r_t, surface_area_conc

    ! Calculate jac_contrib
    if (present(jac_contrib)) then
      jac_contrib(:) = real(0.0, kind=dp)
      do j_phase = 1, _NUM_PHASE_
        do i_spec = 1, _NUM_SPEC_(j_phase)
          temp_jac = 0.0
          if (j_phase.eq.i_phase) temp_jac = 3.0 * r_t * &
                  _DENSITY_(j_phase, i_spec)
          temp_jac = temp_jac - v_p * (3.0/4.0*v_t/const%pi)**(1.0/3.0) * &
                  _DENSITY_(j_phase, i_spec)
          jac_contrib(_SPEC_STATE_ID_(j_phase, i_spec)) = &
                  temp_jac / (r_t * r_t)
        end do
      end do
    end if

  end function surface_area_conc

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Get the surface area concentration for a specific aerosol species
  !! (m^2/m^3). It is assumed the surface is between the gas-phase and an
  !! aerosol phase.
  !!
  !! Species surface area concentration for single particles is calculated the
  !! same way as surface area concentration, but with \f$P\f$ being the 
  !! single-element set of species \f$x\f$.
  !!
  function species_surface_area_conc(this,  i_phase, i_spec, phlex_state, &
                jac_contrib) result(surface_area_conc)
    use pmc_constants
    use pmc_util,                                     only : dp

    !> Surface area concentration
    real(kind=dp) :: surface_area_conc
    !> Aerosol representation data
    class(aero_rep_single_particle_t), intent(in) :: this
    !> Aerosol phase index
    integer(kind=i_kind), intent(in) :: i_phase
    !> Species id
    integer(kind=i_kind), intent(in) :: i_spec
    !> Model state
    type(phlex_state_t), intent(in) :: phlex_state
    !> Contribution to Jacobian matrix. An array of the same size as the
    !! state array that, when present, will be filled with the partial
    !! derivatives of the result of this calculation with each state
    !! variable.
    real(kind=dp), allocatable, intent(inout), optional :: jac_contrib(:)

    real(kind=dp) :: surface_area, v_t, v_p, r_t, temp_jac
    integer(kind=i_kind) :: j_spec, j_phase
    
    ! Calculate phase and total volume denisty (m^3/m^3)
    v_t = 0.0
    v_p = 0.0
    do j_phase = 1, _NUM_PHASE_
      do j_spec = 1, _NUM_SPEC_(j_phase)
        if (j_phase.eq.i_phase .and. j_spec.eq.i_spec) v_p = v_p + &
               _MASS_(j_phase, j_spec) * &
               _DENSITY_(j_phase, j_spec)
        v_t = v_t + _MASS_(j_phase, j_spec) * &
               _DENSITY_(j_phase, j_spec)
      end do 
    end do

    ! Calculate radius (m)
    r_t = (3.0/4.0*v_t/const%pi)**(1.0/3.0)

    ! Calculate the surface area density (m^2/m^3)
    surface_area_conc = 3.0 * v_p / r_t

    ! Calculate jac_contrib
    if (present(jac_contrib)) then
      jac_contrib(:) = real(0.0, kind=dp)
      do j_phase = 1, _NUM_PHASE_
        do j_spec = 1, _NUM_SPEC_(j_phase)
          temp_jac = 0.0
          if (j_phase.eq.i_phase .and. j_spec.eq.i_spec) temp_jac = 3.0 * &
                  r_t * _DENSITY_(j_phase, j_spec)
          temp_jac = temp_jac - v_p * (3.0/4.0*v_t/const%pi)**(1.0/3.0) * &
                  _DENSITY_(j_phase, j_spec)
          jac_contrib(_SPEC_STATE_ID_(j_phase, j_spec)) = &
                  temp_jac / (r_t * r_t)
        end do
      end do
    end if

  end function species_surface_area_conc

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
  !> Get the Kelvin effect adjustment to a species vapor pressure (unitless)
  function kelvin_effect(this, i_spec, phlex_state, jac_contrib)
    use pmc_util,                                     only : dp

    !> Vapor pressure scaling
    real(kind=dp) :: kelvin_effect
    !> Aerosol representation data
    class(aero_rep_single_particle_t), intent(in) :: this
    !> Species id
    integer(kind=i_kind), intent(in) :: i_spec
    !> Model state
    type(phlex_state_t), intent(in) :: phlex_state
    !> Contribution to Jacobian matrix. An array of the same size as the
    !! state array that, when present, will be filled with the partial
    !! derivatives of the result of this calculation with each state
    !! variable.
    real(kind=dp), allocatable, intent(inout), optional :: jac_contrib(:)

    ! TODO Finish

    call die_msg(787876225, "Kelvin effect not available yet.")
    
  end function kelvin_effect

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Get the associated aero_rep_state_t variable
  function get_state(this, phlex_state) &
                  result (aero_rep_state)

    !> Aerosol representation state
    type(aero_rep_single_particle_state_t), pointer :: aero_rep_state
    !> Aerosol representation data
    class(aero_rep_single_particle_t), intent(in) :: this
    !> Model state
    type(phlex_state_t), intent(in) :: phlex_state

    select type (state_ptr => phlex_state%aero_rep_state(_AERO_STATE_ID_)%val)
      type is (aero_rep_single_particle_state_t)
        aero_rep_state => state_ptr
      class default
        call die_msg(814359298, "Received incorrect aero_rep_state_t variable "//&
                "for aero_rep_single_particle_t")
    end select

  end function get_state

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end module pmc_aero_rep_single_particle