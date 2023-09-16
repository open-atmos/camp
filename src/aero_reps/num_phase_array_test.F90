program order_num_phase_array
  implicit none

  !> Order num_phase array from inner most layer to outermost 
  !> Layer_names
  character(len=20), dimension(4) :: ordered_layer_name
  !> Cover names 
  character(len=20), dimension(4) :: cover_name
  !> Num phase array input
  integer, dimension(4) :: num_phase_array
  !> Ordered num phase array output
  integer, allocatable :: ordered_num_phase(:)

  integer :: i_layer, i_cover

  ordered_layer_name = [character(len=20) :: 'bottom bread','almond butter','jam','top bread']
  cover_name = [character(len=20) :: 'bottom bread','jam','almond butter','none']
  num_phase_array = (/ 2,2,4,3 /)
 
  allocate(ordered_num_phase(size(num_phase_array)))

  ! Search for innermost layer
  do i_layer = 1, size(num_phase_array)
    if (cover_name(i_layer) == "none") then
      ordered_num_phase(1) = num_phase_array(i_layer)
    end if
  end do

  do i_cover = 2, size(num_phase_array)
    do i_layer = 1, size(num_phase_array)
      if (ordered_layer_name(i_cover-1).eq.cover_name(i_layer)) then
        ordered_num_phase(i_cover) = num_phase_array(i_layer)
      end if
    end do
  end do

print *, ordered_num_phase

end program order_num_phase_array
