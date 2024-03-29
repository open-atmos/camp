! Copyright (C) 2021 Barcelona Supercomputing Center and University of
! Illinois at Urbana-Champaign
! SPDX-License-Identifier: MIT

!> \file
!> The camp_sub_model_factory module.

!> \page camp_sub_model_add CAMP: Adding a Sub Model
!!
!! TODO write instructions
!!

!> The sub_model_factory_t type and associated subroutines
module camp_sub_model_factory

#ifdef CAMP_USE_JSON
  use json_module
#endif
#ifdef CAMP_USE_MPI
  use mpi
#endif
  use camp_constants,                    only : i_kind, dp
  use camp_mpi
  use camp_sub_model_data
  use camp_util,                         only : die_msg, string_t, assert_msg, &
                                               warn_msg

  ! Use all sub-models
  use camp_sub_model_PDFiTE
  use camp_sub_model_UNIFAC
  use camp_sub_model_ZSR_aerosol_water

  implicit none
  private

  public :: sub_model_factory_t

  !> Identifiers for sub-models - used by binary packing/unpacking functions
  integer(kind=i_kind), parameter, public :: SUB_MODEL_UNIFAC = 1
  integer(kind=i_kind), parameter, public :: SUB_MODEL_ZSR_AEROSOL_WATER = 2
  integer(kind=i_kind), parameter, public :: SUB_MODEL_PDFITE = 3

  !> Factory type for sub-models
  !!
  !! Provides new instances of type extending sub_model_data_t by name or
  !! from input file data
  type :: sub_model_factory_t
  contains
    !> Create a new sub-model by type name
    procedure :: create
    !> Create a new aerosol representation from input data
    procedure :: load
    !> Get the aerosol representation type
    procedure :: get_type
    !> Get a new update data object
    procedure :: initialize_update_data
    !> Determine the number of bytes required to pack a given sub-model
    procedure :: pack_size
    !> Pack a given sub-model onto the buffer, advancing position
    procedure :: bin_pack
    !> Unpack a sub-model from the buffer, advancing position
    procedure :: bin_unpack
  end type sub_model_factory_t

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Create a new sub-model by type name
  function create(this, type_name) result (new_obj)

    !> A new sub-model
    class(sub_model_data_t), pointer :: new_obj
    !> Sub-model factory
    class(sub_model_factory_t), intent(in) :: this
    !> Type of the sub-model
    character(len=*), intent(in) :: type_name

    new_obj => null()

    select case (type_name)
      case ("SUB_MODEL_PDFITE")
        new_obj => sub_model_PDFiTE_t()
      case ("SUB_MODEL_UNIFAC")
        new_obj => sub_model_UNIFAC_t()
      case ("SUB_MODEL_ZSR_AEROSOL_WATER")
        new_obj => sub_model_ZSR_aerosol_water_t()
      case default
        call die_msg(293855421, "Unknown sub-model type: "//type_name)
    end select

  end function create

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Load a sub-model based on its type
#ifdef CAMP_USE_JSON
  function load(this, json, j_obj) result (new_obj)

    !> A new sub-model
    class(sub_model_data_t), pointer :: new_obj
    !> Sub-model factory
    class(sub_model_factory_t), intent(in) :: this
    !> JSON core
    type(json_core), pointer, intent(in) :: json
    !> JSON object
    type(json_value), pointer, intent(in) :: j_obj

    character(kind=json_ck, len=:), allocatable :: unicode_type_name
    character(len=:), allocatable :: type_name
    logical(kind=json_lk) :: found

    new_obj => null()

    ! Get the sub-model type
    call json%get(j_obj, "type", unicode_type_name, found)
    call assert_msg(447218460, found, 'Missing sub-model type.')
    type_name = unicode_type_name

    ! Create a new sub-model instance of the type specified
    new_obj => this%create(type_name)

    ! Load sub-model parameters from the json object
    call new_obj%load(json, j_obj)

#else
  !> Generic warning function when no input file support exists
  function load(this) result (new_obj)

    !> A new sub-model
    class(sub_model_data_t), pointer :: new_obj
    !> Sub-model factory
    class(sub_model_factory_t), intent(in) :: this

    new_obj => null()

    call warn_msg(545649418, "No support for input files.")
#endif
  end function load

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Get the sub-model type as a constant
  integer(kind=i_kind) function get_type(this, sub_model) &
            result (sub_model_data_type)

    !> Sub-model factory
    class(sub_model_factory_t), intent(in) :: this
    !> Sub-model to get type of
    class(sub_model_data_t), intent(in) :: sub_model

    select type (sub_model)
      type is (sub_model_PDFiTE_t)
        sub_model_data_type = SUB_MODEL_PDFITE
      type is (sub_model_UNIFAC_t)
        sub_model_data_type = SUB_MODEL_UNIFAC
      type is (sub_model_ZSR_aerosol_water_t)
        sub_model_data_type = SUB_MODEL_ZSR_AEROSOL_WATER
      class default
        call die_msg(695653684, "Unknown sub-model type")
    end select

  end function get_type

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Initialize an update data object
  subroutine initialize_update_data(this, sub_model, update_data)

    !> Sub-model factory
    class(sub_model_factory_t), intent(in) :: this
    !> Sub-model to be updated
    class(sub_model_data_t), intent(inout) :: sub_model
    !> Update data object
    class(sub_model_update_data_t), intent(out) :: update_data

    select type (update_data)
      class default
        call die_msg(245232793, "Internal error - update data type missing")
    end select

  end subroutine initialize_update_data

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Determine the size of a binary required to pack a sub-model
  integer(kind=i_kind) function pack_size(this, sub_model, comm)

    !> Sub-model factory
    class(sub_model_factory_t) :: this
    !> Sub-model to pack
    class(sub_model_data_t), intent(in) :: sub_model
    !> MPI communicator
    integer, intent(in) :: comm

    pack_size =  camp_mpi_pack_size_integer(int(1, kind=i_kind), comm) + &
                 sub_model%pack_size(comm)

  end function pack_size

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Pack the given value to the buffer, advancing position
  subroutine bin_pack(this, sub_model, buffer, pos, comm)

    !> Sub-model factory
    class(sub_model_factory_t), intent(in) :: this
    !> Sub-model to pack
    class(sub_model_data_t), intent(in) :: sub_model
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos
    !> MPI communicator
    integer, intent(in) :: comm

#ifdef CAMP_USE_MPI
    integer :: sub_model_data_type, i_sub_model, prev_position

    prev_position = pos
    select type (sub_model)
      type is (sub_model_PDFiTE_t)
        sub_model_data_type = SUB_MODEL_PDFITE
      type is (sub_model_UNIFAC_t)
        sub_model_data_type = SUB_MODEL_UNIFAC
      type is (sub_model_ZSR_aerosol_water_t)
        sub_model_data_type = SUB_MODEL_ZSR_AEROSOL_WATER
      class default
        call die_msg(850922257, "Trying to pack sub-model of unknown type.")
    end select
    call camp_mpi_pack_integer(buffer, pos, sub_model_data_type, comm)
    call sub_model%bin_pack(buffer, pos, comm)
    call assert(340451545, &
         pos - prev_position <= this%pack_size(sub_model, comm))
#endif

  end subroutine bin_pack

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Unpack the given value to the buffer, advancing position
  function bin_unpack(this, buffer, pos, comm) result (sub_model)

    !> Unpacked sub-model
    class(sub_model_data_t), pointer :: sub_model
    !> Sub-model factory
    class(sub_model_factory_t), intent(in) :: this
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos
    !> MPI communicator
    integer, intent(in) :: comm

#ifdef CAMP_USE_MPI
    integer :: sub_model_data_type, i_sub_model, prev_position

    prev_position = pos
    call camp_mpi_unpack_integer(buffer, pos, sub_model_data_type, comm)
    select case (sub_model_data_type)
      case (SUB_MODEL_PDFITE)
        sub_model => sub_model_PDFiTE_t()
      case (SUB_MODEL_UNIFAC)
        sub_model => sub_model_UNIFAC_t()
      case (SUB_MODEL_ZSR_AEROSOL_WATER)
        sub_model => sub_model_ZSR_aerosol_water_t()
      case default
        call die_msg(786366152, "Trying to unpack sub-model of unknown "// &
                "type: "//trim(to_string(sub_model_data_type)))
    end select
    call sub_model%bin_unpack(buffer, pos, comm)
    call assert(209433801, &
         pos - prev_position <= this%pack_size(sub_model, comm))
#endif

  end function bin_unpack

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end module camp_sub_model_factory
