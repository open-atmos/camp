program aero_rep_array
  implicit none 
  
  integer :: num_particles, i_particle, i_phase 
  integer :: TOTAL_NUM_PHASES_, TOTAL_NUM_LAYERS_
  integer, dimension(10) :: aero_layer_phase_set
  integer, allocatable :: aero_phase(:)
 
  aero_layer_phase_set = (/ 1,6,5,7,3,4,9,8,2,0 /)
  num_particles = 10
  TOTAL_NUM_PHASES_ = 5
  TOTAL_NUM_LAYERS_ = 2

  allocate(aero_phase(size(aero_layer_phase_set)*num_particles))
  do i_particle = 1, num_particles
    do i_phase = 1, size(aero_layer_phase_set)
      aero_phase((i_particle-1)*TOTAL_NUM_PHASES_*TOTAL_NUM_LAYERS_+i_phase) = &
          aero_layer_phase_set(i_phase)
    end do
  end do

  print *, aero_phase

end program aero_rep_array
