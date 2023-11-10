program construct_layer_state_id
  implicit none 

  !> Number of phases in each layer
  integer, dimension(4) :: num_phase_array
  !> Layer_state_id variable
  integer, allocatable :: layer_state_id_array(:)

  integer :: i_layer

  num_phase_array = (/ 2,2,4,3 /)

  ! Allocate space in layer_state_id
  allocate(layer_state_id_array(size(num_phase_array)+1))

  layer_state_id_array(1) = 1
  do i_layer = 1, size(num_phase_array)
    layer_state_id_array(i_layer+1) = layer_state_id_array(i_layer) + &
      num_phase_array(i_layer)
  end do
  
  print *, layer_state_id_array
end program construct_layer_state_id
