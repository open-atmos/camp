!> this module is necessary to work around cyclic dependencies between the main constraints module
!! and the specific constraint modules
module boxmodel_constraints_utils
  use camp_constants, only: i_kind
  use camp_util, only: die_msg, assert_msg
  use boxmodel_constraints, only: constraint_ptr, CONSTANT_CONSTRAINT, FILE_CONSTRAINT, MONARCH_CONSTRAINT
  use boxmodel_constant_constraint, only: constant_constraint_t
  use boxmodel_file_constraint, only: file_constraint_t
  use boxmodel_monarch_constraint, only: empty_monarch_constraint

  implicit none
  private
  public :: constraint_from_type_id

  ! interface
  !   subroutine constraint_from_type_id(constraint, constraint_type_id)
  !     use boxmodel_constraints, only: constraint_ptr
  !     use camp_constants, only: i_kind
  !     class(constraint_ptr), intent(inout) :: constraint
  !     integer(kind=i_kind), intent(in)     :: constraint_type_id
  !   end subroutine constraint_from_type_id
  ! end interface

contains
  !> allocate constraint from type id
  subroutine constraint_from_type_id(constraint, constraint_type_id)
    class(constraint_ptr), intent(inout) :: constraint
    integer(kind=i_kind), intent(in)     :: constraint_type_id

    call assert_msg(376413702,.not. associated(constraint%val), &
                    "trying to allocate an already allocated constraint")

    select case (constraint_type_id)
    case (CONSTANT_CONSTRAINT)
      constraint%val => constant_constraint_t()

    case (FILE_CONSTRAINT)
      constraint%val => file_constraint_t()

    case (MONARCH_CONSTRAINT)
      constraint%val => empty_monarch_constraint()

    case DEFAULT
      call die_msg(683213892, "unknown constraint type, needs to be assigned a type_id in boxmodel_constraints.F90")
    end select

  end subroutine constraint_from_type_id
end module boxmodel_constraints_utils
