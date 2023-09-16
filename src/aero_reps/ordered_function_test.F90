program order_array
    implicit none

    !> Layer_names
    character(len=20), dimension(4) :: layer_name
    character(len=20), dimension(4) :: ordered_layer_name
    !> Cover names 
    character(len=20), dimension(4) :: cover_name
    !> Phase names
    character(len=20), dimension(11) :: phase_name
    character(len=20), dimension(11) :: ordered_phase_name 
    !> Ordered and unordered layer state id
    integer, dimension(5) :: layer_state_id
    integer, dimension(5) :: unordered_layer_state_id

!    layer_name = [character(len=10) :: 'bread','cheese','bread','butter']
!    cover_name = [character(len=10) :: 'cheese','butter','none','bread']
!    phase_name = [character(len=10) :: 'wheat','milk','salt','wheat','water', 'filler', & 
!                                       'cream','salt','enzyme','man power']
!    unordered_layer_state_id = (/ 1,2,4,7,11 /)
!    layer_state_id = (/ 1,4,8,10,11 /)

    layer_name = [character(len=20) :: 'almond butter','top bread','jam','bottom bread']
    cover_name = [character(len=20) :: 'bottom bread','jam','almond butter','none']
    phase_name = [character(len=11) :: 'almonds','salt','wheat','water','rasberry', & 
                                       'honey','sugar','lemon','wheat', 'water','salt']

    unordered_layer_state_id = (/ 1,3,5,9,12 /)
    layer_state_id = (/ 1,4,6,10,12 /)

    ordered_layer_name = ordered_layer_array(cover_name,layer_name)
    ordered_phase_name = ordered_phase_array(phase_name, layer_state_id, unordered_layer_state_id, &
                              cover_name,layer_name,ordered_layer_name)
    print *, 'ordered layer names : ', ordered_layer_name
    print *, 'ordered phase names : ', ordered_phase_name 
contains
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> Order layer array from inner most layer to outermost 
  function ordered_layer_array(cover_name_array,layer_name_array) result(ordered_layer_name) 
    implicit none
    !> Layer names in order (only used during initialization)
    character(len=20), dimension(4) :: ordered_layer_name
    !> Layer names 
    character(len=*), intent(in) :: layer_name_array(:)
    !> Cover names 
    character(len=*), intent(in) :: cover_name_array(:)

    integer :: i_layer, i_cover

    do i_layer = 1, 4
      if (cover_name_array(i_layer) == "none") then
        ordered_layer_name(1) = layer_name_array(i_layer)
      end if
    end do
    

    do i_cover = 2, 4
      do i_layer = 1, 4
        if (ordered_layer_name(i_cover-1).eq.cover_name_array(i_layer)) then
          ordered_layer_name(i_cover) = layer_name_array(i_layer)
        end if
      end do
    end do
    !print *, ordered_layer_name

  end function ordered_layer_array

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Order layer array from inner most layer to outermost 
  function ordered_phase_array(phase_name_array, &
                              ordered_layer_state_id_array, unordered_layer_state_id_array, &
                              cover_name_array,layer_name_array,ordered_layer_name_array) &
                              result(ordered_phase_name)
    implicit none
    !> Layer names 
    character(len=*), intent(in) :: layer_name_array(:)
    character(len=*), intent(in) :: ordered_layer_name_array(:)
    !> Cover names 
    character(len=*), intent(in) :: cover_name_array(:)
    !> Phase names 
    character(len=*), intent(in) :: phase_name_array(:)
    !> Layer_state_id unordered and ordered
    integer, intent(in) :: ordered_layer_state_id_array(:)
    integer, intent(in) :: unordered_layer_state_id_array(:)

    !> Ordered phase array
    character(len=20), dimension(11) :: ordered_phase_name

    integer :: i_layer, i_cover
 
    ! Search for innermost layer

    do i_layer = 1,4
      i_cover = 1
      if (cover_name_array(i_layer) == "none") then
        !print *, ordered_layer_state_id_array(i_cover)
        !print *, unordered_layer_state_id_array(i_layer)
        ordered_phase_name(ordered_layer_state_id_array(i_cover):ordered_layer_state_id_array(i_cover+1)-1) = &
        phase_name_array(unordered_layer_state_id_array(i_layer):unordered_layer_state_id_array(i_layer+1)-1)
      end if
    end do

    do i_cover = 2, 4
      do i_layer = 1, 4
        !print *, i_cover
        !print *, i_layer
        !print *, cover_name_array(i_layer)
        !print *, ordered_layer_name_array(i_cover-1)
        if (ordered_layer_name_array(i_cover-1).eq.cover_name_array(i_layer)) then
          ordered_phase_name(ordered_layer_state_id_array(i_cover):ordered_layer_state_id_array(i_cover+1)-1) = &
          phase_name_array(unordered_layer_state_id_array(i_layer):unordered_layer_state_id_array(i_layer+1)-1)
        end if
      end do
    end do
    !print *, ordered_phase_name 
  end function ordered_phase_array
 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end program order_array
