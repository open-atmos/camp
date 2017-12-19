! Copyright (C) 2017 Matt Dawson
! Licensed under the GNU General Public License version 2 or (at your
! option) any later version. See the file COPYING for details.

!> \file
!> The pmc_aero_phase_data module.

!> The abstract aero_phase_data_t structure and associated subroutines.
module pmc_aero_phase_data

  use pmc_constants,                  only : i_kind, dp
  use pmc_mpi
  use pmc_util,                       only : die_msg, string_t
  use pmc_property
  use pmc_chem_spec_data
#ifdef PMC_USE_MPI
  use mpi
#endif
#ifdef PMC_USE_JSON
  use json_module
#endif

  implicit none
  private

  public :: aero_phase_data_t, aero_phase_data_ptr

  !> Reallocation increment
  integer(kind=i_kind), parameter :: REALLOC_INC = 50

  !> Aerosol phase data type
  !!
  !! Aerosol phase information. The chemistry module uses a set of phases to
  !! model chemical reactions for aerosol species. Chemical reactions can take
  !! place within a phase or across an interface between phases (e.g., for 
  !! condensation/evaporation). Phases may or may not correspond to aerosol
  !! representations directly. For example, a binned aerosol representation
  !! may have one aerosol phase per bin, or an organic and aqueous phase in
  !! each bin, or three concentric aerosol phases (layers).
  type :: aero_phase_data_t
    private
    !> Name of the aerosol phase
    character(len=:), allocatable :: phase_name
    !> Number of species in the phase
    integer(kind=i_kind) :: num_spec = 0
    !> Species names. These are species that are present in the aerosol
    !! phase. These species must exist in the chem_spec_data_t variable
    !! during initialization.
    type(string_t), pointer :: spec_name(:) => null()
    !> Aerosol phase parameters. These will be available during 
    !! initialization, but not during integration. All information required
    !! by functions of the aerosol representation related to a phase must be 
    !! saved by the aero_rep_data_t-exdending type in the condensed data 
    !! arrays.
    type(property_t), pointer :: property_set => null()
  contains
    !> Aerosol representation initialization
    procedure :: initialize => pmc_aero_phase_data_initialize
    !> Get the name of the aerosol phase
    procedure :: name => pmc_aero_phase_data_name
    !> Get property data associated with this phase
    procedure :: get_property_set => pmc_aero_phase_data_property_set
    !> Get the size of the state array required for this phase
    procedure :: state_size => pmc_aero_phase_data_state_size
    !> Get an aerosol species state id
    procedure :: state_id => pmc_aero_phase_data_state_id
    !> Load data from an input file
    procedure :: load => pmc_aero_phase_data_load
    !> Get the number of species in the phase
    procedure :: size => pmc_aero_phase_data_size
    !> Print the aerosol phase data
    procedure :: print => pmc_aero_phase_data_print

    !> Private functions
    !> Add a species
    procedure, private :: add => pmc_aero_phase_data_add
    !> Ensure there is enough room in the species dataset to add a specified
    !! number of species
    procedure, private :: ensure_size => pmc_aero_phase_data_ensure_size
    !> Find a species index by name
    procedure, private :: find => pmc_aero_phase_data_find
  end type aero_phase_data_t

  !> Constructor for aero_phase_data_t
  interface aero_phase_data_t
    procedure :: aero_phase_data_constructor
  end interface aero_phase_data_t

  !> Pointer type for use in building lists of pointers to aero_phase_data_t
  !! instances
  type aero_phase_data_ptr
    class(aero_phase_data_t), pointer :: val
  end type aero_phase_data_ptr

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Constructor for aero_phase_data_t
  function aero_phase_data_constructor(phase_name, init_size) result(new_obj)

    !> A new set of aerosol-phase species
    type(aero_phase_data_t), pointer :: new_obj
    !> Name of the aerosol phase
    character(len=:), allocatable, intent(in), optional :: phase_name
    !> Number of species to allocate space for initially
    integer(kind=i_kind), intent(in), optional :: init_size

    integer(kind=i_kind) :: alloc_size = REALLOC_INC

    if (present(init_size)) alloc_size = init_size
    allocate(new_obj)
    if (present(phase_name)) new_obj%phase_name = phase_name
    allocate(new_obj%spec_name(alloc_size))

  end function aero_phase_data_constructor

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Initialize the aerosol phase data, validating species names.
  subroutine pmc_aero_phase_data_initialize(this, chem_spec_data)

    !> Aerosol phase data
    class(aero_phase_data_t), intent(inout) :: this
    !> Chemical species data
    type(chem_spec_data_t), intent(in) :: chem_spec_data

    integer(kind=i_kind) :: i_spec

    do i_spec = 1, this%num_spec
      if (.not.chem_spec_data%exists(this%spec_name(i_spec)%string)) then
        call die_msg(589987734, "Aerosol phase species "// &
            trim(this%spec_name(i_spec)%string)//" missing in chem_spec_data.")
      end if
    end do

  end subroutine pmc_aero_phase_data_initialize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Get the aerosol phase name
  function pmc_aero_phase_data_name(this) result (phase_name)

    !> The name of the aerosol phase
    character(len=:), allocatable :: phase_name
    !> Aerosol phase data
    class(aero_phase_data_t), intent(in) :: this

    phase_name = this%phase_name

  end function pmc_aero_phase_data_name

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Get the aerosol phase property set
  function pmc_aero_phase_data_property_set(this) result (property_set)

    !> A pointer to the aerosol phase property set
    class(property_t), pointer :: property_set
    !> Aerosol phase data
    class(aero_phase_data_t), intent(in) :: this

    property_set => this%property_set

  end function pmc_aero_phase_data_property_set

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Get the size of the state array required for this phase
  integer(kind=i_kind) function  pmc_aero_phase_data_state_size(this) &
                  result (state_size)

    !> Aerosol phase data
    class(aero_phase_data_t), intent(inout) :: this

    state_size = this%num_spec

  end function pmc_aero_phase_data_state_size

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Get the index of a species in the state array for this phase
  integer(kind=i_kind) function pmc_aero_phase_data_state_id(this, spec_name) &
          result (state_id)

    !> Aerosol phase data
    class(aero_phase_data_t), intent(inout) :: this
    !> Chemical species name
    character(len=:), allocatable, intent(in) :: spec_name

    state_id = this%find(spec_name)

  end function pmc_aero_phase_data_state_id

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Load species from an input file
#ifdef PMC_USE_JSON
  !! j_obj is expected to be a JSON object containing data related to an
  !! aerosol phase. It should be of the form:
  !!
  !! { "pmc-data" : [
  !!   {
  !!     "name" : "my aerosol phase"
  !!     "type" : "AERO_PHASE"
  !!     "species" : [
  !!       "a species",
  !!       "another species",
  !!       ...
  !!     ],
  !!     ...
  !!   },
  !!   ...
  !! ]}
  !!
  !! The key-value pair "name" is required and must contain the unique name
  !! used for this aerosol phase in the mechanism. A single phase may be 
  !! present in several aerosol groups (e.g., an aqueous phase is each bin
  !! of a binned aerosol representation), but the species associated with a
  !! particular phase will be constant throughout the model run.
  !!
  !! The key-value pair "type" is also required and its value must be
  !! AERO_PHASE. A list of species should be included in a key-value pair
  !! named "species" whose value is an array of species names. These names
  !! must correspond to species names loaded into the chem_spec_data_t 
  !! object. All other data is optional and may include any valid JSON value,
  !! including nested objects. Multiple entries with the same aerosol phase 
  !! name will be merged into a single phase, but duplicate property names for
  !! the same phase will cause an error.
  subroutine pmc_aero_phase_data_load(this, json, j_obj)

    !> Aerosol phase data
    class(aero_phase_data_t), intent(inout) :: this
    !> JSON core
    type(json_core), pointer, intent(in) :: json
    !> JSON object
    type(json_value), pointer, intent(in) :: j_obj

    type(json_value), pointer :: child, next, species
    character(kind=json_ck, len=:), allocatable :: key, unicode_str_val
    integer(kind=i_kind) :: var_type

    character(len=:), allocatable :: phase_name, str_val
    type(property_t), pointer :: property_set

    allocate(property_set)
    property_set = property_t()

    next => null()
    call json%get_child(j_obj, child)
    do while (associated(child))
      call json%info(child, name=key, var_type=var_type)
      if (key.eq."name") then
        if (var_type.ne.json_string) call die_msg(429142134, &
                "Received non-string aerosol phase name.")
        call json%get(child, unicode_str_val)
        this%phase_name = unicode_str_val
      else if (key.eq."species") then
        if (var_type.ne.json_array) call die_msg(293312378, &
                "Received non-array list of aerosol phase species: "//&
                to_string(var_type))
        call json%get_child(child, species)
        do while (associated(species))
          call json%info(species, var_type=var_type)
          if (var_type.ne.json_string) call die_msg(669858868, &
                  "Received non-string aerosol phase species name.")
          call json%get(species, unicode_str_val)
          str_val = unicode_str_val
          call this%add(str_val)
          call json%get_next(species, next)
          species => next
        end do
      else if (key.ne."type") then
        call property_set%load(json, child, .false.)
      end if 
      call json%get_next(child, next)
      child => next
    end do
  
    if (associated(this%property_set)) then
      call this%property_set%update(property_set)
    else 
      this%property_set => property_set
    end if
#else
  subroutine pmc_aero_phase_data_load(this)

    !> Aerosol phase data
    class(aero_phase_data_t), intent(in) :: this

    call warn_msg(236665532, "No support for input files.")
#endif
  end subroutine pmc_aero_phase_data_load
   
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Get the number of species in the phase
  integer(kind=i_kind) function pmc_aero_phase_data_size(this) result(num_spec)

    !> Aerosol phase data
    class(aero_phase_data_t), intent(in) :: this

    num_spec = this%num_spec

  end function pmc_aero_phase_data_size

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Print out the aerosol phase data
  subroutine pmc_aero_phase_data_print(this)

    !> Aerosol phase data
    class(aero_phase_data_t), intent(in) :: this

    integer(kind=i_kind) :: i_spec

    write(*,*) "Aerosol phase: ", this%phase_name
    write(*,*) "Number of species: ", this%num_spec
    write(*,*) "Species: ["
    do i_spec = 1, this%num_spec 
      write(*,*) this%spec_name(i_spec)%string
    end do
    write(*,*) "]"
    call this%property_set%print()
    write(*,*) "End aerosol phase: ", this%phase_name

  end subroutine pmc_aero_phase_data_print

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Add a new chemical species to the phase
  subroutine pmc_aero_phase_data_add(this, spec_name)

    !> Aerosol phase data
    class(aero_phase_data_t), intent(inout) :: this
    !> Name of the species to add
    character(len=:), allocatable, intent(in) :: spec_name

    integer(kind=i_kind) :: i_spec

    i_spec = this%find(spec_name)
    if (i_spec.ne.0) then
      call warn_msg(980242449, "Species "//spec_name//&
              " added more than once to phase "//this%name())
      return
    end if
    call this%ensure_size(1)
    this%num_spec = this%num_spec + 1
    this%spec_name(this%num_spec) = string_t(spec_name)

  end subroutine pmc_aero_phase_data_add

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Ensure there is enough room in the species dataset to add a specified 
  !! number of species
  subroutine pmc_aero_phase_data_ensure_size(this, num_spec)

    !> Aerosol phase data
    class(aero_phase_data_t), intent(inout) :: this
    !> Number of new species to ensure space for
    integer(kind=i_kind) :: num_spec

    integer :: new_size
    type(string_t), pointer :: new_name(:)

    if (size(this%spec_name) .ge. this%num_spec + num_spec) return
    new_size = this%num_spec + num_spec + REALLOC_INC
    allocate(new_name(new_size))
    new_name(1:this%num_spec) = this%spec_name(1:this%num_spec)
    deallocate(this%spec_name)
    this%spec_name => new_name

  end subroutine pmc_aero_phase_data_ensure_size

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Get the index of an aerosol-phase species by name. Return 0 if the species
  !! is not found
  integer(kind=i_kind) function pmc_aero_phase_data_find(this, spec_name) &
                  result (spec_id)

    !> Aerosol phase data
    class(aero_phase_data_t), intent(in) :: this
    !> Species name
    character(len=:), allocatable, intent(in) :: spec_name

    integer(kind=i_kind) :: i_spec

    spec_id = 0
    do i_spec = 1, this%num_spec
      if (this%spec_name(i_spec)%string .eq. spec_name) then
        spec_id = i_spec
        return
      end if
    end do

  end function pmc_aero_phase_data_find

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end module pmc_aero_phase_data