
!> this module is necessary to work around cyclic dependencies between the main montecarlo module
!! and the specific montecarlo modules
module boxmodel_montecarlo_utils
  use camp_constants, only: i_kind
  use camp_util, only: die_msg, assert_msg, to_string
  use boxmodel_montecarlo, only: montecarlo_ptr, MONTECARLO_UNIFORM, MONTECARLO_GAUSSIAN
  use boxmodel_montecarlo_uniform, only: montecarlo_uniform_t
  use boxmodel_montecarlo_gaussian, only: montecarlo_gaussian_t

contains
  !> allocate montecarlo from type id
  subroutine montecarlo_from_typeid(montecarlo, montecarlo_type_id)
    class(montecarlo_ptr), intent(inout) :: montecarlo
    integer(kind=i_kind), intent(in) :: montecarlo_type_id
    
    call assert_msg(420475357,.not. associated(montecarlo%val), &
                    "trying to allocate an already allocated montecarlo object")

    select case (montecarlo_type_id)
    case (MONTECARLO_UNIFORM)
      montecarlo%val => montecarlo_uniform_t()
    case (MONTECARLO_GAUSSIAN)
      montecarlo%val => montecarlo_gaussian_t()
    case default
      call die_msg(1349266877, "unknown montecarlo type id: "//to_string(montecarlo_type_id))
    end select
  end subroutine montecarlo_from_typeid

end module boxmodel_montecarlo_utils
