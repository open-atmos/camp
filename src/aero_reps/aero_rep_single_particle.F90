! Copyright (C) 2021 Barcelona Supercomputing Center and University of
! Illinois at Urbana-Champaign
! SPDX-License-Identifier: MIT

!> \file
!> The camp_aero_rep_single_particle module.

!> \page camp_aero_rep_single_particle CAMP: Single Particle Aerosol Representation
!!
!! The single particle aerosol representation is for use with a PartMC
!! particle-resolved run. The \c json object for this \ref camp_aero_rep
!! "aerosol representation" has the following format:
!! \code{.json}
!!  { "camp-data" : [
!!    {
!!      "name" : "my single particle aero rep",
!!      "type" : "AERO_REP_SINGLE_PARTICLE"
!!    },
!!    ...
!!  ]}
!! \endcode
!! The key-value pair \b type is required and must be \b
!! AERO_REP_SINGLE_PARTICLE. This representation assumes that every \ref
!! input_format_aero_phase "aerosol phase" available will be present
!! once in each particle.
!!
!! The number concentration for each particle must be
!! set from an external model using
!! \c camp_aero_rep_single_particle::aero_rep_update_data_single_particle_number_t
!! objects.

!> The aero_rep_single_particle_t type and associated subroutines.
module camp_aero_rep_single_particle

  use camp_aero_phase_data
  use camp_aero_rep_data
  use camp_chem_spec_data
  use camp_camp_state
  use camp_mpi
  use camp_property
  use camp_util,                                  only: dp, i_kind, &
                                                       string_t, assert_msg, &
                                                       die_msg, to_string, &
                                                       assert

  use iso_c_binding

  implicit none
  private

#define NUM_LAYERS_ this%condensed_data_int(1)
#define AERO_REP_ID_ this%condensed_data_int(2)
#define MAX_PARTICLES_ this%condensed_data_int(3)
#define PARTICLE_STATE_SIZE_ this%condensed_data_int(4)
#define NUM_INT_PROP_ 4
#define NUM_REAL_PROP_ 0
#define NUM_ENV_PARAM_PER_PARTICLE_ 1
#define LAYER_PHASE_START_(l) this%condensed_data_int(NUM_INT_PROP_+l)
#define LAYER_PHASE_END_(l) this%condensed_data_int(NUM_INT_PROP_+NUM_LAYERS_+l)
#define TOTAL_NUM_PHASES_ (LAYER_PHASE_END_(NUM_LAYERS_))
#define NUM_PHASES_(l) (LAYER_PHASE_END_(l)-LAYER_PHASE_START_(l)+1)
#define PHASE_STATE_ID_(l,p) this%condensed_data_int(NUM_INT_PROP_+2*NUM_LAYERS_+LAYER_PHASE_START_(l)+p-1)
#define PHASE_MODEL_DATA_ID_(l,p) this%condensed_data_int(NUM_INT_PROP_+2*NUM_LAYERS_+TOTAL_NUM_PHASES_+LAYER_PHASE_START_(l)+p-1)
#define PHASE_NUM_JAC_ELEM_(l,p) this%condensed_data_int(NUM_INT_PROP_+2*NUM_LAYERS_+2*TOTAL_NUM_PHASES_+LAYER_PHASE_START_(l)+p-1)

  ! Update types (These must match values in aero_rep_single_particle.c)
  integer(kind=i_kind), parameter, public :: UPDATE_NUMBER_CONC = 0

  public :: aero_rep_single_particle_t, &
            aero_rep_update_data_single_particle_number_t, &
            ordered_layer_ids

  !> Single particle aerosol representation
  !!
  !! Time-invariant data related to a single particle aerosol representation.
  type, extends(aero_rep_data_t) :: aero_rep_single_particle_t
    !> Unique names for each instance of every chemical species in the
    !! aerosol representaiton
    type(string_t), allocatable, private :: unique_names_(:)
    !> Layer names, ordered inner-most to outer-most
    type(string_t), allocatable, private :: layer_names_(:)
    !> Boolean array, true for phases that exist in the surface layer
    logical, allocatable, private :: aero_is_at_surface_(:)
    !> First state id for the representation (only used during initialization)
    integer(kind=i_kind) :: state_id_start = -99999
  contains
    !> Initialize the aerosol representation data, validating component data and
    !! loading any required information from the \c
    !! aero_rep_data_t::property_set. This routine should be called once for
    !! each aerosol representation at the beginning of a model run after all
    !! the input files have been read in. It ensures all data required during
    !! the model run are included in the condensed data arrays.
    procedure :: initialize
    !> Returns the maximum number of computational particles
    procedure :: maximum_computational_particles
    !> Initialize an update data number object
    procedure :: update_data_initialize_number => update_data_init_number
    !> Get the size of the section of the
    !! \c camp_camp_state::camp_state_t::state_var array required for this
    !! aerosol representation.
    !!
    !! For a single particle representation, the size will correspond to the
    !! the sum of the sizes of a single instance of each aerosol phase
    !! provided to \c aero_rep_single_particle::initialize()
    procedure :: size => get_size
    !> Get the number of state variables per-particle
    !!
    !! Calling functions can assume each particle has the same size on the
    !! state array, and that individual particle states are contiguous and
    !! arranged sequentially
    procedure :: per_particle_size
    !> Get a list of unique names for each element on the
    !! \c camp_camp_state::camp_state_t::state_var array for this aerosol
    !! representation. The list may be restricted to a particular phase and/or
    !! aerosol species by including the phase_name and spec_name arguments.
    !!
    !! For a single particle representation, the unique names will be the
    !! phase name with the species name separated by a '.'
    procedure :: unique_names
    !> Get a species id on the \c camp_camp_state::camp_state_t::state_var
    !! array by its unique name. These are unique ids for each element on the
    !! state array for this \ref camp_aero_rep "aerosol representation" and
    !! are numbered:
    !!
    !!   \f[x_u \in x_f ... (x_f+n-1)\f]
    !!
    !! where \f$x_u\f$ is the id of the element corresponding to the species
    !! with unique name \f$u\f$ on the \c
    !! camp_camp_state::camp_state_t::state_var array, \f$x_f\f$ is the index
    !! of the first element for this aerosol representation on the state array
    !! and \f$n\f$ is the total number of variables on the state array from
    !! this aerosol representation.
    procedure :: spec_state_id
    !> Get the non-unique name of a species by its unique name
    procedure :: spec_name
    !> Get the number of instances of an aerosol phase in this representation
    procedure :: num_phase_instances
    !> Get the number of Jacobian elements used in calculations of aerosol
    !! mass, volume, number, etc. for a particular phase
    procedure :: num_jac_elem
    !> Returns the number of layers
    procedure :: num_layers
    !> Returns array of booleans indicating is phase is at surface
    procedure :: aero_is_at_surface
    !> Returns the number of phases in a layer or overall
    procedure :: num_phases
    !> Returns the number of state variables for a layer and phase
    procedure :: phase_state_size
    !> Finalize the aerosol representation
    final :: finalize
  end type aero_rep_single_particle_t

  ! Constructor for aero_rep_single_particle_t
  interface aero_rep_single_particle_t
    procedure :: constructor
  end interface aero_rep_single_particle_t

  !> Single particle update number concentration object
  type, extends(aero_rep_update_data_t) :: &
            aero_rep_update_data_single_particle_number_t
  private
    !> Flag indicating whether the update data is allocated
    logical :: is_malloced = .false.
    !> Unique id for finding aerosol representations during initialization
    integer(kind=i_kind) :: aero_rep_unique_id = 0
    !> Maximum number of computational particles
    integer(kind=i_kind) :: maximum_computational_particles = 0
  contains
    !> Update the number
    procedure :: set_number__n_m3 => update_data_set_number__n_m3
    !> Determine the pack size of the local update data
    procedure :: internal_pack_size => internal_pack_size_number
    !> Pack the local update data to a binary
    procedure :: internal_bin_pack => internal_bin_pack_number
    !> Unpack the local update data from a binary
    procedure :: internal_bin_unpack => internal_bin_unpack_number
    !> Finalize the number update data
    final :: update_data_number_finalize
  end type aero_rep_update_data_single_particle_number_t

  !> Interface to c aerosol representation functions
  interface

    !> Allocate space for a number update
    function aero_rep_single_particle_create_number_update_data() &
              result (update_data) bind (c)
      use iso_c_binding
      !> Allocated update_data object
      type(c_ptr) :: update_data
    end function aero_rep_single_particle_create_number_update_data

    !> Set a new particle number concentration
    subroutine aero_rep_single_particle_set_number_update_data__n_m3( &
              update_data, aero_rep_unique_id, particle_id, number_conc) &
              bind (c)
      use iso_c_binding
      !> Update data
      type(c_ptr), value :: update_data
      !> Aerosol representation unique id
      integer(kind=c_int), value :: aero_rep_unique_id
      !> Computational particle index
      integer(kind=c_int), value :: particle_id
      !> New number (m)
      real(kind=c_double), value :: number_conc
    end subroutine aero_rep_single_particle_set_number_update_data__n_m3

    !> Free an update data object
    pure subroutine aero_rep_free_update_data(update_data) bind (c)
      use iso_c_binding
      !> Update data
      type(c_ptr), value, intent(in) :: update_data
    end subroutine aero_rep_free_update_data

  end interface

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
  !! loading any required information from the \c
  !! aero_rep_data_t::property_set. This routine should be called once for
  !! each aerosol representation at the beginning of a model run after all
  !! the input files have been read in. It ensures all data required during
  !! the model run are included in the condensed data arrays.
  subroutine initialize(this, aero_phase_set, spec_state_id)

    !> Aerosol representation data
    class(aero_rep_single_particle_t), intent(inout) :: this
    !> The set of aerosol phases
    type(aero_phase_data_ptr), pointer, intent(in) :: aero_phase_set(:)
    !> Beginning state id for this aerosol representation in the model species
    !! state array
    integer(kind=i_kind), intent(in) :: spec_state_id

    ! Unordered layer names (only used during initialization)
    type(string_t), allocatable :: layer_names_unordered(:)
    ! Cover names (only used during initialization)
    type(string_t), allocatable :: cover_names_unordered(:)
    ! Index in layer_names_unordered for each layer from inner- to outer-most
    ! layer
    integer(kind=i_kind), allocatable :: ordered_layer_id(:)

    character(len=:), allocatable :: key_name, layer_name, layer_covers, &
                                     phase_name
    type(property_t), pointer :: layers, layer
    type(property_ptr), allocatable :: phases(:)
    integer(kind=i_kind) :: i_particle, i_phase, i_layer, i_aero, curr_id
    integer(kind=i_kind) :: i_cover, j_phase, j_layer, i_map, curr_phase
    integer(kind=i_kind) :: num_phases, num_int_param, num_float_param, &
                            num_particles
    logical :: found

    ! Get the maximum number of computational particles
    key_name = "maximum computational particles"
    call assert_msg(331697425, &
                    this%property_set%get_int(key_name, num_particles), &
                    "Missing maximum number of computational particles")

    ! Get the set of layers
    key_name = "layers"
    call assert_msg(314292954, &
                    this%property_set%get_property_t(key_name, layers), &
                    "Missing layers for single-particle aerosol "// &
                    "representation '"//this%rep_name//"'")
                    call assert_msg(168669831, layers%size() .gt. 0, &
                    "No Layers specified for single-particle layer "// &
                    "aerosol representation '"//this%rep_name//"'")

    ! Allocate space for the working arrays
    allocate(phases(layers%size()))
    allocate(cover_names_unordered(layers%size()))
    allocate(layer_names_unordered(layers%size()))

    ! Loop through the layers, adding names and counting the spaces needed
    ! on the condensed data arrays, and counting the total phases instances
    num_phases = 0
    call layers%iter_reset()
    do i_layer = 1, layers%size()

      ! Get the layer properties
      call assert_msg(303808978, layers%get_property_t(val=layer), &
              "Invalid structure for layer '"// &
              layer_names_unordered(i_layer)%string// &
              "' in single-particle layer representation '"// &
              this%rep_name//"'")

      ! Get the layer name
      key_name = "name"
      call assert_msg(364496472, layer%get_string(key_name, layer_name), &
              "Missing layer name in single-particle layer aerosol "// &
              "representation '"//this%rep_name//"'")
      layer_names_unordered(i_layer)%string = layer_name

      ! Get the cover name
      key_name = "covers"
      call assert_msg(350939595, layer%get_string(key_name, layer_covers), &
                "Missing cover name in layer'"// &
                layer_names_unordered(i_layer)%string// &
                "' in single-particle layer aerosol representation '"// &
                this%rep_name//"'")
      cover_names_unordered(i_layer)%string = layer_covers
           
      ! Get the set of phases
      key_name = "phases"
      call assert_msg(647756433, &
              layer%get_property_t(key_name, phases(i_layer)%val_), &
              "Missing phases for layer '"// &
              layer_names_unordered(i_layer)%string// &
              "' in single-particle layer aerosol representation '"// &
              this%rep_name//"'")

      ! Add the phases to the counter
      call assert_msg(002679882, phases(i_layer)%val_%size().gt.0, &
              "No phases specified for layer '"// &
              layer_names_unordered(i_layer)%string// &
              "' in single-particle layer aerosol representation '"// &
              this%rep_name//"'")

      ! add to running total of phase count
      num_phases = num_phases + phases(i_layer)%val_%size()

      call layers%iter_next()
    end do

    ! get the map of layer names after reordering
    ordered_layer_id = ordered_layer_ids(layer_names_unordered, &
                                           cover_names_unordered)

    ! set the layer names
    allocate(this%layer_names_(size(ordered_layer_id)))
    this%layer_names_(:) = layer_names_unordered(ordered_layer_id(:))

    ! Allocate condensed data arrays
    num_int_param = NUM_INT_PROP_ + 2 * layers%size() + 3 * num_phases
    num_float_param = NUM_REAL_PROP_
    allocate(this%condensed_data_int(num_int_param))
    allocate(this%condensed_data_real(num_float_param))
    this%condensed_data_int(:) = int(0, kind=i_kind)
    this%condensed_data_real(:) = real(0.0, kind=dp)

    ! Save space for the environment-dependent parameters
    this%num_env_params = NUM_ENV_PARAM_PER_PARTICLE_ * num_particles

    ! Save representation dimensions
    NUM_LAYERS_ = layers%size()
    MAX_PARTICLES_ = num_particles

    ! validate phase names, assign aero_phase pointers for each phase in
    ! each layer in each particle, and set PHASE_STATE_ID and
    ! PHASE_MODEL_DATA_ID for each phase
    allocate(this%aero_phase(num_phases * num_particles))
    allocate(this%aero_is_at_surface_(num_phases * num_particles))
    curr_phase = 1
    do i_layer = 1, size(ordered_layer_id)
      j_layer = ordered_layer_id(i_layer)

      ! Set the starting and ending indices for the phases in this layer
      LAYER_PHASE_START_(i_layer) = curr_phase
      LAYER_PHASE_END_(i_layer) = curr_phase + phases(j_layer)%val_%size() - 1

      curr_phase = curr_phase + phases(j_layer)%val_%size()
    end do

    curr_id = spec_state_id
    this%state_id_start = spec_state_id
    curr_phase = 1
    do i_layer = 1, size(ordered_layer_id)
      j_layer = ordered_layer_id(i_layer)

      ! Loop through the phases and make sure they exist
      call phases(j_layer)%val_%iter_reset()
      do i_phase = 1, phases(j_layer)%val_%size()

        ! Get the phase name
        call assert_msg(566480284, &
                phases(j_layer)%val_%get_string(val=phase_name), &
                "Non-string phase name for layer '"// &
                layer_names_unordered(j_layer)%string// &
                "' in single-particle layer aerosol representation '"// &
                this%rep_name//"'")
  
        ! find phase and set pointer and indices
        found = .false.
        do j_phase = 1, size(aero_phase_set)
          if (aero_phase_set(j_phase)%val%name() .eq. phase_name) then
            found = .true.
            do i_particle = 0, num_particles-1
              this%aero_phase(i_particle*num_phases + curr_phase) = &
                aero_phase_set(j_phase)
              if (i_layer .eq. NUM_LAYERS_) then
                this%aero_is_at_surface_(i_particle*num_phases + curr_phase) = &
                  .true.
                else
                  this%aero_is_at_surface_(i_particle*num_phases + curr_phase) = &
                    .false.
              end if
            end do
            PHASE_STATE_ID_(i_layer,i_phase) = curr_id
            PHASE_MODEL_DATA_ID_(i_layer,i_phase) = j_phase
            curr_id = curr_id + aero_phase_set(j_phase)%val%size()
            curr_phase = curr_phase + 1
            exit
          end if
        end do

        call phases(j_layer)%val_%iter_next()
      end do
    end do
    PARTICLE_STATE_SIZE_ = curr_id - spec_state_id

    ! Initialize the aerosol representation id
    AERO_REP_ID_ = -1

    ! Set the unique names for the chemical species
    this%unique_names_ = this%unique_names( )

  end subroutine initialize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Returns the maximum nunmber of computational particles
  integer(kind=i_kind) function maximum_computational_particles(this)

    !> Aerosol representation data
    class(aero_rep_single_particle_t), intent(in) :: this

    maximum_computational_particles = MAX_PARTICLES_

  end function maximum_computational_particles

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Get the size of the section of the
  !! \c camp_camp_state::camp_state_t::state_var array required for this
  !! aerosol representation.
  !!
  !! For a single particle representation, the size will correspond to the
  !! the sum of the sizes of a single instance of each aerosol phase
  !! provided to \c aero_rep_single_particle::initialize()
  function get_size(this) result (state_size)

    !> Size on the state array
    integer(kind=i_kind) :: state_size
    !> Aerosol representation data
    class(aero_rep_single_particle_t), intent(in) :: this

    state_size = MAX_PARTICLES_ * PARTICLE_STATE_SIZE_

  end function get_size

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Get the number of state variables per-particle
  !!
  !! Calling functions can assume each particle has the same size on the
  !! state array, and that individual particle states are contiguous and
  !! arranged sequentially
  function per_particle_size(this) result(state_size)

    !> Size on the state array per particle
    integer(kind=i_kind) :: state_size
    !> Aerosol representation data
    class(aero_rep_single_particle_t), intent(in) :: this

    state_size = PARTICLE_STATE_SIZE_

  end function per_particle_size

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Get a list of unique names for each element on the
  !! \c camp_camp_state::camp_state_t::state_var array for this aerosol
  !! representation. The list may be restricted to a particular phase and/or
  !! aerosol species by including the phase_name and spec_name arguments.
  !!
  !! For a single particle representation, the unique names will be a 'P'
  !! followed by the computational particle number, a '.', the phase name,
  !! another '.', and the species name.
  function unique_names(this, phase_name, tracer_type, spec_name)

    use camp_util,                      only : integer_to_string
    !> List of unique names
    type(string_t), allocatable :: unique_names(:)
    !> Aerosol representation data
    class(aero_rep_single_particle_t), intent(in) :: this
    !> Aerosol layer name
    !character(len=*), optional, intent(in) :: layer_name
    !> Aerosol phase name
    character(len=*), optional, intent(in) :: phase_name
    !> Aerosol-phase species tracer type
    integer(kind=i_kind), optional, intent(in) :: tracer_type
    !> Aerosol-phase species name
    character(len=*), optional, intent(in) :: spec_name

    integer :: i_particle, i_layer, i_phase, i_spec, j_spec
    integer :: num_spec, curr_tracer_type
    type(string_t), allocatable :: spec_names(:)

    ! copy saved unique names when available and no filters are included
    if (.not. present(phase_name) .and. &
        .not. present(tracer_type) .and. &
        .not. present(spec_name) .and. &
        allocated(this%unique_names_)) then
      unique_names = this%unique_names_
      return
    end if
    
    ! count the number of unique names
    num_spec = 0
    do i_phase = 1, size(this%aero_phase)
      if (present(phase_name)) then
        if(phase_name .ne. this%aero_phase(i_phase)%val%name()) cycle
      end if
      if (present(spec_name) .or. present(tracer_type)) then
        spec_names = this%aero_phase(i_phase)%val%get_species_names()
        do j_spec = 1, size(spec_names)
          if (present(spec_name)) then
            if (spec_name .ne. spec_names(j_spec)%string) cycle
          end if
          if (present(tracer_type)) then
            curr_tracer_type = &
                this%aero_phase(i_phase)%val%get_species_type( &
                    spec_names(j_spec)%string)
            if (tracer_type .ne. curr_tracer_type) cycle
          end if
          num_spec = num_spec + 1
        end do
        deallocate(spec_names)
      else
        num_spec = num_spec + this%aero_phase(i_phase)%val%size()
      end if
    end do            

    ! allocate space for the unique names and assign them
    num_spec = num_spec / MAX_PARTICLES_ ! we need per-particle value for indexing
    allocate(unique_names(num_spec*MAX_PARTICLES_))
    i_spec = 1
    do i_layer = 1, NUM_LAYERS_
      do i_phase = LAYER_PHASE_START_(i_layer), LAYER_PHASE_END_(i_layer)
        if (present(phase_name)) then
          if(phase_name .ne. this%aero_phase(i_phase)%val%name()) cycle
        end if
        spec_names = this%aero_phase(i_phase)%val%get_species_names()
        do j_spec = 1, this%aero_phase(i_phase)%val%size()
          if (present(spec_name)) then
            if (spec_name .ne. spec_names(j_spec)%string) cycle
          end if
          if (present(tracer_type)) then
            curr_tracer_type = &
                this%aero_phase(i_phase)%val%get_species_type( &
                    spec_names(j_spec)%string)
            if (tracer_type .ne. curr_tracer_type) cycle
          end if
          do i_particle = 1, MAX_PARTICLES_
            unique_names((i_particle-1)*num_spec+i_spec)%string = 'P'// &
              trim(integer_to_string(i_particle))//"."// &
              this%layer_names_(i_layer)%string//"."// &
              this%aero_phase(i_phase)%val%name()//"."// &
              spec_names(j_spec)%string
          end do
          i_spec = i_spec + 1
        end do
        deallocate(spec_names)
      end do
    end do
        
  end function unique_names

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Get a species id on the \c camp_camp_state::camp_state_t::state_var
  !! array by its unique name. These are unique ids for each element on the
  !! state array for this \ref camp_aero_rep "aerosol representation" and
  !! are numbered:
  !!
  !!   \f[x_u \in x_f ... (x_f+n-1)\f]
  !!
  !! where \f$x_u\f$ is the id of the element corresponding to the species
  !! with unique name \f$u\f$ on the \c
  !! camp_camp_state::camp_state_t::state_var array, \f$x_f\f$ is the index
  !! of the first element for this aerosol representation on the state array
  !! and \f$n\f$ is the total number of variables on the state array from
  !! this aerosol representation.
  function spec_state_id(this, unique_name) result (spec_id)

    !> Species state id
    integer(kind=i_kind) :: spec_id
    !> Aerosol representation data
    class(aero_rep_single_particle_t), intent(in) :: this
    !> Unique name
    character(len=*), intent(in) :: unique_name

    integer(kind=i_kind) :: i_spec

    spec_id = this%state_id_start
    do i_spec = 1, size(this%unique_names_)
      if (this%unique_names_(i_spec)%string .eq. unique_name) then
        return
      end if
      spec_id = spec_id + 1
    end do
    call die_msg( 449087541, "Cannot find species '"//unique_name//"'" )

  end function spec_state_id

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Get the non-unique name of a species in this aerosol representation by
  !! id.
  function spec_name(this, unique_name)

    !> Chemical species name
    character(len=:), allocatable :: spec_name
    !> Aerosol representation data
    class(aero_rep_single_particle_t), intent(in) :: this
    !> Unique name of the species in this aerosol representation
    character(len=*), intent(in) :: unique_name

    type(string_t) :: l_unique_name
    type(string_t), allocatable :: substrs(:)

    l_unique_name%string = unique_name
    substrs = l_unique_name%split(".")
    call assert(407537518, size( substrs ) .eq. 4 )
    spec_name = substrs(4)%string

  end function spec_name

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Get the number of instances of a specified aerosol phase. In the single
  !! particle representation with layers, a phase can exist in multiple layers
  !! in one particle.  
  integer(kind=i_kind) function num_phase_instances(this, phase_name)
  
    !> Aerosol representation data
    class(aero_rep_single_particle_t), intent(in) :: this
    !> Aerosol phase name
    character(len=*), intent(in) :: phase_name

    integer(kind=i_kind) ::  i_phase, i_layer, phase_index

    num_phase_instances = 0
    phase_index = 0
    do i_layer = 1, NUM_LAYERS_
      do i_phase = LAYER_PHASE_START_(i_layer), LAYER_PHASE_END_(i_layer)
        if (this%aero_phase(i_phase)%val%name() .eq. phase_name) then
          phase_index = phase_index + 1
        end if
      end do
    end do
    num_phase_instances = phase_index * MAX_PARTICLES_

  end function num_phase_instances

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Get the number of Jacobian elements used in calculations of aerosol mass,
  !! volume, number, etc. for a particular phase
  function num_jac_elem(this, phase_id)

    !> Number of Jacobian elements used
    integer(kind=i_kind) :: num_jac_elem
    !> Aerosol respresentation data
    class(aero_rep_single_particle_t), intent(in) :: this
    !> Aerosol phase id
    integer(kind=i_kind), intent(in) :: phase_id

    integer(kind=i_kind) :: i_phase

    call assert_msg(927040495, phase_id .ge. 1 .and. &
                                phase_id .le. size( this%aero_phase ), &
                     "Aerosol phase index out of range. Got "// &
                     trim( integer_to_string( phase_id ) )//", expected 1:"// &
                     trim( integer_to_string( size( this%aero_phase ) ) ) )
    num_jac_elem = 0
    do i_phase = 1, TOTAL_NUM_PHASES_
      num_jac_elem = num_jac_elem + &
                   this%aero_phase(i_phase)%val%num_jac_elem()
    end do

  end function num_jac_elem

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    !> Returns the number of layers
    integer function num_layers(this)

      !> Aerosol representation data
      class(aero_rep_single_particle_t), intent(in) :: this

      num_layers = NUM_LAYERS_

    end function num_layers

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    !> Returns array of booleans indicating if phase is at surface
    function aero_is_at_surface(this) 

      !> Aerosol representation data
      class(aero_rep_single_particle_t), intent(in) :: this

      aero_is_at_surface = aero_is_at_surface_

    end function aero_is_at_surface

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    !> Returns the number of phases in a layer or overall
    integer function num_phases(this, layer)

      !> Aerosol representation data
      class(aero_rep_single_particle_t), intent(in) :: this
      !> Layer id
      integer, optional, intent(in) :: layer

      if (present(layer)) then
        num_phases = LAYER_PHASE_END_(layer) - LAYER_PHASE_START_(layer) + 1
      else
        num_phases = LAYER_PHASE_END_(NUM_LAYERS_) - LAYER_PHASE_START_(1) + 1
      end if

    end function num_phases

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    !> Returns the number of state variables for a layer and phase
    integer function phase_state_size(this, layer, phase)

      use camp_util,                   only : die_msg

      !> Aerosol representation data
      class(aero_rep_single_particle_t), intent(in) :: this
      !> Layer id
      integer, optional, intent(in) :: layer
      !> Phase id
      integer, optional, intent(in) :: phase

      if (present(layer) .and. present(phase)) then
        if (layer .eq. NUM_LAYERS_ .and. phase .eq. NUM_PHASES_(layer)) then
          phase_state_size = PHASE_STATE_ID_(1,1) + PARTICLE_STATE_SIZE_ - &
                             PHASE_STATE_ID_(layer, phase)
        else if (phase .eq. NUM_PHASES_(layer)) then
          phase_state_size = PHASE_STATE_ID_(layer+1, 1) - &
                             PHASE_STATE_ID_(layer, phase)
        else
          phase_state_size = PHASE_STATE_ID_(layer, phase+1) - &
                             PHASE_STATE_ID_(layer, phase)
        end if
      else if (present(layer)) then
        if (layer .eq. NUM_LAYERS_) then
          phase_state_size = PHASE_STATE_ID_(1,1) + PARTICLE_STATE_SIZE_ - &
                             PHASE_STATE_ID_(layer, 1)
        else
          phase_state_size = PHASE_STATE_ID_(layer+1, 1) - &
                             PHASE_STATE_ID_(layer, 1)
        end if
      else if (present(phase)) then
        call die_msg(917793122, "Must specify layer if including phase is "// &
                     "state size request")
      else
          phase_state_size = PARTICLE_STATE_SIZE_
      end if

    end function phase_state_size

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Finalize the aerosol representation
  elemental subroutine finalize(this)

    !> Aerosol representation data
    type(aero_rep_single_particle_t), intent(inout) :: this

    if (allocated(this%rep_name)) deallocate(this%rep_name)
    if (allocated(this%aero_phase)) then
      ! The core will deallocate the aerosol phases
      call this%aero_phase(:)%dereference()
      deallocate(this%aero_phase)
    end if
    if (allocated(this%unique_names_)) deallocate(this%unique_names_)
    if (allocated(this%layer_names_)) deallocate(this%layer_names_)
    if (associated(this%property_set)) deallocate(this%property_set)
    if (allocated(this%condensed_data_real)) &
        deallocate(this%condensed_data_real)
    if (allocated(this%condensed_data_int)) &
        deallocate(this%condensed_data_int)

  end subroutine finalize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Initialize an update data object
  subroutine update_data_init_number(this, update_data, aero_rep_type)

    use camp_rand,                                only : generate_int_id

    !> Aerosol representation to update
    class(aero_rep_single_particle_t), intent(inout) :: this
    !> Update data object
    class(aero_rep_update_data_single_particle_number_t), intent(out) :: &
        update_data
    !> Aerosol representaiton id
    integer(kind=i_kind), intent(in) :: aero_rep_type

    ! If an aerosol representation id has not been generated, do it now
    if (AERO_REP_ID_.eq.-1) then
      AERO_REP_ID_ = generate_int_id()
    end if

    update_data%aero_rep_unique_id = AERO_REP_ID_
    update_data%maximum_computational_particles = &
        this%maximum_computational_particles( )
    update_data%aero_rep_type = int(aero_rep_type, kind=c_int)
    update_data%update_data = &
      aero_rep_single_particle_create_number_update_data()
    update_data%is_malloced = .true.

  end subroutine update_data_init_number

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Order layer array from inner most layer to outermost 
  function ordered_layer_ids(layer_names_unordered, cover_names_unordered)

    !> Layer names in original order
    type(string_t), intent(in) :: layer_names_unordered(:)
    !> Name of "covered" layer for each layer in layer_name_unordered
    type(string_t), intent(in) :: cover_names_unordered(:)
    !> Index of name in layer_name_unordered for each layer after reordering
    integer, allocatable :: ordered_layer_ids(:)

    integer(kind=i_kind) :: i_layer, j_layer, i_cover

    ! Ensure layer names do not repeat 
    do i_layer = 1, size(layer_names_unordered)
      do j_layer = 1, size(layer_names_unordered)
        if (i_layer .eq. j_layer) cycle
        call assert_msg(781626922, layer_names_unordered(i_layer)%string .ne. &
                                   layer_names_unordered(j_layer)%string, &
                        "Duplicate layer name in single particle "// &
                        "representation: '"// &
                        trim(layer_names_unordered(i_layer)%string)//"'")
      end do
    end do

    allocate(ordered_layer_ids(size(layer_names_unordered)))
   
    ! Search for innermost layer with cover set to 'none'
    do i_layer = 1, size(layer_names_unordered)
      if (cover_names_unordered(i_layer)%string == "none") then
        ordered_layer_ids(1) = i_layer
      end if
    end do

    ! Assign each layer working outwards from center of particle
    do i_cover = 2, size(ordered_layer_ids)
      do i_layer = 1, size(layer_names_unordered)
        if (layer_names_unordered(ordered_layer_ids(i_cover-1))%string &
            .eq. cover_names_unordered(i_layer)%string) then
             ordered_layer_ids(i_cover) = i_layer
          exit
        end if
      end do
    end do

  end function ordered_layer_ids

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Set packed update data for particle number (#/m3) for a particular
  !! computational particle.
  subroutine update_data_set_number__n_m3(this, particle_id, number_conc)

    !> Update data
    class(aero_rep_update_data_single_particle_number_t), intent(inout) :: &
            this
    !> Computational particle index
    integer(kind=i_kind), intent(in) :: particle_id
    !> Updated number
    real(kind=dp), intent(in) :: number_conc

    call assert_msg(611967802, this%is_malloced, &
            "Trying to set number of uninitialized update object.")
    call assert_msg(689085496, particle_id .ge. 1 .and. &
                    particle_id .le. this%maximum_computational_particles, &
                    "Invalid computational particle index: "// &
                    trim(integer_to_string(particle_id)))
    call aero_rep_single_particle_set_number_update_data__n_m3( &
            this%get_data(), this%aero_rep_unique_id, particle_id-1, &
            number_conc)

  end subroutine update_data_set_number__n_m3

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Determine the size of a binary required to pack the reaction data
  integer(kind=i_kind) function internal_pack_size_number(this, comm) &
      result(pack_size)

    !> Aerosol representation update data
    class(aero_rep_update_data_single_particle_number_t), intent(in) :: this
    !> MPI communicator
    integer, intent(in) :: comm

    pack_size = &
      camp_mpi_pack_size_logical(this%is_malloced, comm) + &
      camp_mpi_pack_size_integer(this%maximum_computational_particles, comm) + &
      camp_mpi_pack_size_integer(this%aero_rep_unique_id, comm)

  end function internal_pack_size_number

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Pack the given value to the buffer, advancing position
  subroutine internal_bin_pack_number(this, buffer, pos, comm)

    !> Aerosol representation update data
    class(aero_rep_update_data_single_particle_number_t), intent(in) :: this
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
    call camp_mpi_pack_integer(buffer, pos, &
                              this%maximum_computational_particles, comm)
    call camp_mpi_pack_integer(buffer, pos, this%aero_rep_unique_id, comm)
    call assert(411585487, &
         pos - prev_position <= this%pack_size(comm))
#endif

  end subroutine internal_bin_pack_number

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Unpack the given value from the buffer, advancing position
  subroutine internal_bin_unpack_number(this, buffer, pos, comm)

    !> Aerosol representation update data
    class(aero_rep_update_data_single_particle_number_t), intent(inout) :: this
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
    call camp_mpi_unpack_integer(buffer, pos, &
                                this%maximum_computational_particles, comm)
    call camp_mpi_unpack_integer(buffer, pos, this%aero_rep_unique_id, comm)
    call assert(351557153, &
         pos - prev_position <= this%pack_size(comm))
    this%update_data = aero_rep_single_particle_create_number_update_data()
#endif

  end subroutine internal_bin_unpack_number

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Finalize a number update data object
  elemental subroutine update_data_number_finalize(this)

    !> Update data object to free
    type(aero_rep_update_data_single_particle_number_t), intent(inout) :: this

    if (this%is_malloced) call aero_rep_free_update_data(this%update_data)

  end subroutine update_data_number_finalize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end module camp_aero_rep_single_particle
