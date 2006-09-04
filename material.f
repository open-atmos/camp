! -*- mode: f90;-*-

module mod_material

  type material
     integer n_spec
     integer i_water ! water species number
     real*8, dimension(:), pointer ::  rho ! densities (kg m^{-3})
     integer, dimension(:), pointer :: nu ! number of ions in the solute
     real*8, dimension(:), pointer :: eps ! solubilities (1)
     real*8, dimension(:), pointer :: M_w ! molecular weights (kg mole^{-1})
  end type material

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  real*8 function average_solute_quantity(V, mat, quantity)

    ! returns the volume-average of the non-water elements of quantity

    real*8, dimension(:), intent(in) :: V         ! species volumes (m^3)
    type(material), intent(in) :: mat             ! material properties
    real*8, dimension(:), intent(in) :: quantity  ! quantity to average

    real*8 :: ones(mat%n_spec)

    ones = 1d0
    average_solute_quantity = total_solute_quantity(V, mat, quantity) &
         / total_solute_quantity(V, mat, ones)

  end function average_solute_quantity

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  real*8 function total_solute_quantity(V, mat, quantity)

    ! returns the volume-total of the non-water elements of quantity

    real*8, dimension(:), intent(in) :: V         ! species volumes (m^3)
    type(material), intent(in) :: mat             ! material properties
    real*8, dimension(:), intent(in) :: quantity  ! quantity to total

    real*8 total
    integer i

    total = 0d0
    do i = 1,mat%n_spec
       if (i .ne. mat%i_water) then
          total = total + V(i) * quantity(i)
       end if
    end do
    total_solute_quantity = total

  end function total_solute_quantity

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  real*8 function average_water_quantity(V, mat, quantity)

    ! returns the water element of quantity

    real*8, dimension(:), intent(in) :: V         ! species volumes (m^3)
    type(material), intent(in) :: mat             ! material properties
    real*8, dimension(:), intent(in) :: quantity  ! quantity to average

    average_water_quantity = quantity(mat%i_water)

  end function average_water_quantity

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  real*8 function total_water_quantity(V, mat, quantity)

    ! returns the volume-total of the water element of quantity

    real*8, dimension(:), intent(in) :: V         ! species volumes (m^3)
    type(material), intent(in) :: mat             ! material properties
    real*8, dimension(:), intent(in) :: quantity  ! quantity to total

    total_water_quantity = V(mat%i_water) * quantity(mat%i_water)

  end function total_water_quantity

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end module mod_material
