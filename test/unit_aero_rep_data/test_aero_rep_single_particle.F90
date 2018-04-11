! Copyright (C) 2017 Matt Dawson
! Licensed under the GNU General Public License version 2 or (at your
! option) any later version. See the file COPYING for details.

!> \file
!> The pmc_test_aero_rep_data program

!> Test class for the aero_rep_data_t extending types
program pmc_test_aero_rep_data

  use pmc_util,                         only: i_kind, dp, assert, &
                                              almost_equal
  use pmc_property
  use pmc_phlex_core
  use pmc_phlex_state
  use pmc_aero_rep_data
  use pmc_aero_rep_single_particle
#ifdef PMC_USE_JSON
  use json_module
#endif
  use pmc_mpi

  implicit none

  ! New-line character
  character(len=*), parameter :: new_line = char(10)

  !> initialize mpi
  call pmc_mpi_init()

  if (run_pmc_aero_rep_data_tests()) then
    write(*,*) "Aerosol rep data tests - PASS"
  else
    write(*,*) "Aerosol rep data tests - FAIL"
  end if

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Run all pmc_aero_rep_data tests
  logical function run_pmc_aero_rep_data_tests() result(passed)

    use pmc_phlex_solver_data

    type(phlex_solver_data_t), pointer :: phlex_solver_data

    phlex_solver_data => phlex_solver_data_t()

    if (phlex_solver_data%is_solver_available()) then
      passed = build_aero_rep_data_set_test()
    else
      call warn_msg(594028423, "No solver available")
      passed = .true.
    end if

  end function run_pmc_aero_rep_data_tests

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Build aero_rep_data set
  logical function build_aero_rep_data_set_test()

    type(phlex_core_t), pointer :: phlex_core
    type(phlex_state_t), pointer :: phlex_state
    class(aero_rep_data_t), pointer :: aero_rep

#ifdef PMC_USE_JSON

    integer(kind=i_kind) :: i_rep, i_spec, j_spec, rep_id, i_phase
    type(string_t), allocatable :: rep_names(:)
    character(len=:), allocatable :: rep_name, spec_name, phase_name
    type(string_t), allocatable :: file_list(:), unique_names(:)

    character, allocatable :: buffer(:)
    integer(kind=i_kind) :: pos, pack_size

    build_aero_rep_data_set_test = .false.

    phlex_core => phlex_core_t()

    allocate(file_list(1))
    file_list(1)%string = 'test_run/unit_aero_rep_data/test_aero_rep_single_particle.json'

    call phlex_core%load(file_list)
    call phlex_core%initialize()
    phlex_state => phlex_core%new_state()

    ! Set up the list of aerosol representation names
    ! !!! Add new aero_rep_data_t extending types here !!!
    allocate(rep_names(1))
    rep_names(1)%string = "AERO_REP_SINGLE_PARTICLE"

    ! Check the number of aerosol representations in the core
    call assert(154970920, size(phlex_core%aero_rep) .eq. size(rep_names))

    ! Loop through all the aerosol representations
    do i_rep = 1, size(rep_names)
    
      ! Check the aerosol representation getter functions
      rep_name = rep_names(i_rep)%string
      call assert_msg(253854173, phlex_core%find_aero_rep(rep_name, rep_id), rep_name)
      call assert_msg(362813745, rep_id .gt. 0, rep_name)
      call assert_msg(589355969, phlex_core%find_aero_rep(rep_name, aero_rep), rep_name)
      call assert_msg(191203602, associated(aero_rep), rep_name)
      select type (aero_rep)
        type is (aero_rep_single_particle_t)
        class default
          call die_msg(519535557, rep_name)
      end select
      aero_rep => phlex_core%aero_rep(rep_id)%val
      call assert_msg(240871376, associated(aero_rep), rep_name)
      select type (aero_rep)
        type is (aero_rep_single_particle_t)
        class default
          call die_msg(625136356, rep_name)
      end select

      ! Check the unique name functions
      unique_names = aero_rep%unique_names()
      call assert_msg(885541843, allocated(unique_names), rep_name)
      call assert_msg(206819761, size(unique_names).eq.8, rep_name)
      do i_spec = 1, size(unique_names)
        call assert_msg(142263656, aero_rep%spec_state_id(&
                unique_names(i_spec)%string).gt.0, rep_name)
        do j_spec = 1, size(unique_names)
          if (i_spec.eq.j_spec) cycle
          call assert_msg(414662586, aero_rep%spec_state_id(&
                  unique_names(i_spec)%string) .ne. aero_rep%spec_state_id(&
                  unique_names(j_spec)%string), rep_name)
        end do
      end do

      ! Set the species concentrations
      phase_name = "my test phase one"
      spec_name = "species a"
      unique_names = aero_rep%unique_names(phase_name = phase_name, spec_name = spec_name)
      i_spec = aero_rep%spec_state_id(unique_names(1)%string)
      call assert_msg(258227897, i_spec.gt.0, rep_name)
      phlex_state%state_var(i_spec) = 1.5
      spec_name = "species b"
      unique_names = aero_rep%unique_names(phase_name = phase_name, spec_name = spec_name)
      i_spec = aero_rep%spec_state_id(unique_names(1)%string)
      call assert_msg(418308482, i_spec.gt.0, rep_name)
      phlex_state%state_var(i_spec) = 2.5
      spec_name = "species c"
      unique_names = aero_rep%unique_names(phase_name = phase_name, spec_name = spec_name)
      i_spec = aero_rep%spec_state_id(unique_names(1)%string)
      call assert_msg(420214016, i_spec.gt.0, rep_name)
      phlex_state%state_var(i_spec) = 3.5
      phase_name = "my test phase two"
      spec_name = "species c"
      unique_names = aero_rep%unique_names(phase_name = phase_name, spec_name = spec_name)
      i_spec = aero_rep%spec_state_id(unique_names(1)%string)
      call assert_msg(416855243, i_spec.gt.0, rep_name)
      phlex_state%state_var(i_spec) = 4.5
      spec_name = "species d"
      unique_names = aero_rep%unique_names(phase_name = phase_name, spec_name = spec_name)
      i_spec = aero_rep%spec_state_id(unique_names(1)%string)
      call assert_msg(578389067, i_spec.gt.0, rep_name)
      phlex_state%state_var(i_spec) = 5.5
      spec_name = "species e"
      unique_names = aero_rep%unique_names(phase_name = phase_name, spec_name = spec_name)
      i_spec = aero_rep%spec_state_id(unique_names(1)%string)
      call assert_msg(147314014, i_spec.gt.0, rep_name)
      phlex_state%state_var(i_spec) = 6.5
      phase_name = "my last test phase"
      spec_name = "species b"
      unique_names = aero_rep%unique_names(phase_name = phase_name, spec_name = spec_name)
      i_spec = aero_rep%spec_state_id(unique_names(1)%string)
      call assert_msg(401514617, i_spec.gt.0, rep_name)
      phlex_state%state_var(i_spec) = 7.5
      spec_name = "species e"
      unique_names = aero_rep%unique_names(phase_name = phase_name, spec_name = spec_name)
      i_spec = aero_rep%spec_state_id(unique_names(1)%string)
      call assert_msg(291101806, i_spec.gt.0, rep_name)
      phlex_state%state_var(i_spec) = 8.5

    end do

    rep_name = "AERO_REP_BAD_NAME"
    call assert(676257369, .not.phlex_core%find_aero_rep(rep_name, rep_id))
    call assert(453526213, rep_id .eq. 0)
    call assert(848319807, .not.phlex_core%find_aero_rep(rep_name, aero_rep))
    call assert(343113402, .not.associated(aero_rep))



#ifdef PMC_USE_MPI
#endif

    ! If condensed data arrays are used for aerosol phases in the future, put
    ! tests for passed info here

#endif  
    build_aero_rep_data_set_test = .true.

  end function build_aero_rep_data_set_test

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end program pmc_test_aero_rep_data