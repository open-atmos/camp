! Copyright (C) 2017 Matt Dawson
! Licensed under the GNU General Public License version 2 or (at your
! option) any later version. See the file COPYING for details.

!> \file
!> The pmc_sub_model_data module.

!> \page phlex_sub_model Phlexible Mechanism for Chemistry: Sub Model (general)
!!
!! A sub model is used during solving to calculate parameters based on the
!! current model state for use by reactions.
!!
!! The available sub models are:
!!  - \ref phlex_sub_model_UNIFAC "UNIFAC Activity Coefficients"
!!  - \ref phlex_sub_model_ZSR "Zdanovskii-Stokes-Robinson Aerosol Water"
!!  - \ref phlex_sub_model_PDFITE "PD-FiTE Activity Coefficients"
!!
!! The general input format for a sub-model can be found \ref
!! input_format_sub_model "here".
!!
!! General instructions for adding a new aerosol representation can be found
!! \ref phlex_sub_model_add "here".

!> The abstract sub_model_data_t structure and associated subroutines.
module pmc_sub_model_data

  use pmc_constants,                    only : i_kind, dp
  use pmc_mpi
  use pmc_util,                         only : die_msg, string_t
  use pmc_property
#ifdef PMC_USE_MPI
  use mpi
#endif
#ifdef PMC_USE_JSON
  use json_module
#endif

  implicit none
  private

  public :: sub_model_data_t, sub_model_data_ptr

  !> Abstract sub model data type
  !!
  !! Time-invariant data required by a sub-model to perform calculations
  !! during solving. Derived types extending sub_model_data_t should
  !! be related to specific models or parameterizations. Sub models will
  !! have access to the current model state during solving, but will not
  !! be able to modify state variables or contribute to derivative arrays
  !! directly (this must be done by reactions).
  !!
  !! See \ref phlex_sub_model "Sub Models" for details.
  type, abstract :: sub_model_data_t
    private
    !> Name of the sub model
    character(len=:), allocatable, public :: model_name
    !> Sub model parameters. These will be available during initialization,
    !! but not during integration. All information required by the sub
    !! model during solving must be saved by the extending type to the
    !! condensed data arrays.
    type(property_t), pointer, public :: property_set => null()
    !> Condensed sub model data. These arrays will be available during 
    !! integration, and should contain any information required by the
    !! sub model that cannot be obtained from the 
    !! pmc_phlex_state::phlex_state_t object. (floating point)
    real(kind=dp), allocatable, public :: condensed_data_real(:)
    !> Condensed sub model data. These arrays will be available during 
    !! integration, and should contain any information required by the
    !! sub model that cannot be obtained from the 
    !! pmc_phlex_state::phlex_state_t object. (integer)
    integer(kind=i_kind), allocatable, public :: condensed_data_int(:)
  contains
    !> Initialize the sub model data, validating input parameters and
    !! loading any required information form the \c
    !! sub_model_data_t::property_set. This routine should be called
    !! once for each sub model at the beginning of the model run after all
    !! the input files have been read in. It ensures all data required
    !! during the model run are included in the condensed data arrays.
    procedure(initialize), deferred :: initialize
    !> Get the name of the sub model
    procedure :: name => get_name
    !> Determine the number of bytes required to pack the sub-model data
    procedure :: pack_size
    !> Packs the sub model into the buffer, advancing position
    procedure :: bin_pack
    !> Unpacks the sub model from the buffer, advancing position
    procedure :: bin_unpack
    !> Load data from an input file
    procedure :: load
    !> Print the sub model data
    procedure :: print => do_print
  end type sub_model_data_t

  !> Pointer to sub_model_data_t extending types
  type :: sub_model_data_ptr
    class(sub_model_data_t), pointer :: val
  end type sub_model_data_ptr

interface

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Initialize the sub model data, validating input parameters and
  !! loading any required information form the \c
  !! sub_model_data_t::property_set. This routine should be called
  !! once for each sub model at the beginning of the model run after all
  !! the input files have been read in. It ensures all data required
  !! during the model run are included in the condensed data arrays.
  subroutine initialize(this, aero_rep_set, aero_phase_set, chem_spec_data)
    
    use pmc_chem_spec_data
    use pmc_aero_rep_data
    use pmc_aero_phase_data
    import :: sub_model_data_t

    !> Sub model data
    class(sub_model_data_t), intent(inout) :: this
    !> The set of aerosol representations
    type(aero_rep_data_ptr), pointer, intent(in) :: aero_rep_set(:)
    !> The set of aerosol phases
    type(aero_phase_data_ptr), pointer, intent(in) :: aero_phase_set(:)
    !> Chemical species data
    type(chem_spec_data_t), intent(in) :: chem_spec_data

  end subroutine initialize
 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end interface

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Get the name of the sub model
  function get_name(this)

    !> Sub model name
    character(len=:), allocatable :: get_name
    !> Sub model data
    class(sub_model_data_t), intent(in) :: this

    get_name = this%model_name

  end function get_name

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> \page input_format_sub_model Input JSON Object Format: Sub Model (general)
  !!
  !! A \c json object containing the information required by a \ref
  !! phlex_sub_model "sub model" of the form:
  !! \code{.json}
  !! { "pmc-data" : [
  !!   {
  !!     "type" : "SUB_MODEL_TYPE",
  !!     "some parameter" : 123.34,
  !!     "some other parameter" : true,
  !!     "nested parameters" : {
  !!       "sub param 1" : 12.43,
  !!       "sub param other " : "some text",
  !!       ...
  !!     },
  !!     ...
  !!   },
  !!   ...
  !! ]}
  !! \endcode
  !! Sum models must have a unique \b type that corresponds to a valid
  !! sub model type. These include:
  !!
  !!   - \subpage phlex_sub_model_UNIFAC_activity "UNIFAC Activity"
  !!
  !! All remaining data are optional and may include and valid \b json value,
  !! including nested objects. However, extending types will have specific
  !! requirements for the remaining data.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Load a sub model from an input file
#ifdef PMC_USE_JSON
  subroutine load(this, json, j_obj)

    !> Sub model data
    class(sub_model_data_t), intent(inout) :: this
    !> JSON core
    type(json_core), pointer, intent(in) :: json
    !> JSON object
    type(json_value), pointer, intent(in) :: j_obj

    type(json_value), pointer :: child, next
    character(kind=json_ck, len=:), allocatable :: key, unicode_str_val
    integer(kind=json_ik) :: var_type
    logical :: found_name

    this%property_set => property_t()

    found_name = .false.

    next => null()
    call json%get_child(j_obj, child)
    do while (associated(child))
      call json%info(child, name=key, var_type=var_type)
      if (key.eq."name") then
        call assert_msg(241525122, var_type.eq.json_string, &
                "Received non-string value for sub model name")
        call json%get(child, unicode_str_val)
        this%model_name = unicode_str_val
        found_name = .true.
      else if (key.ne."type") then
        call this%property_set%load(json, child, .false.)
      end if
      call json%get_next(child, next)
      child => next
    end do
    call assert_msg(281116577, found_name, &
            "Received unnamed sub model.")
#else
  subroutine load(this)

    !> Sub model data
    class(sub_model_data_t), intent(inout) :: this

    call warn_msg(391981683, "No support for input files.")
#endif

  end subroutine load

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Determine the size of a binary required to pack the reaction data
  integer(kind=i_kind) function pack_size(this)

    !> Sub model data
    class(sub_model_data_t), intent(in) :: this
    
    pack_size = &
            pmc_mpi_pack_size_real_array(this%condensed_data_real) + &
            pmc_mpi_pack_size_integer_array(this%condensed_data_int)

  end function pack_size

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Pack the given value to the buffer, advancing position
  subroutine bin_pack(this, buffer, pos)

    !> Sub model data
    class(sub_model_data_t), intent(in) :: this
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos

#ifdef PMC_USE_MPI
    integer :: prev_position

    prev_position = pos
    call pmc_mpi_pack_real_array(buffer, pos, this%condensed_data_real)
    call pmc_mpi_pack_integer_array(buffer, pos, this%condensed_data_int)
    call assert(924075845, &
         pos - prev_position <= this%pack_size())
#endif

  end subroutine bin_pack

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Unpack the given value from the buffer, advancing position
  subroutine bin_unpack(this, buffer, pos)

    !> Sub model data
    class(sub_model_data_t), intent(out) :: this
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos


#ifdef PMC_USE_MPI
    integer :: prev_position

    prev_position = pos
    call pmc_mpi_unpack_real_array(buffer, pos, this%condensed_data_real)
    call pmc_mpi_unpack_integer_array(buffer, pos, this%condensed_data_int)
    call assert(299381254, &
         pos - prev_position <= this%pack_size())
#endif

  end subroutine bin_unpack

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Print the sub model data
  subroutine do_print(this, file_unit)

    !> Sub model data
    class(sub_model_data_t), intent(in) :: this
    !> File unit for output
    integer(kind=i_kind), optional :: file_unit

    integer(kind=i_kind) :: f_unit = 6

    if (present(file_unit)) f_unit = file_unit
    write(f_unit,*) "*** Sub Model ***"
    if (associated(this%property_set)) call this%property_set%print(f_unit)
    if (allocated(this%condensed_data_int)) &
      write(f_unit,*) "  *** condensed data int: ", this%condensed_data_int(:)
    if (allocated(this%condensed_data_real)) &
      write(f_unit,*) "  *** condensed data real: ", this%condensed_data_real(:)

  end subroutine do_print

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end module pmc_sub_model_data