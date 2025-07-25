! Copyright (C) 2021 Barcelona Supercomputing Center and University of
! Illinois at Urbana-Champaign
! SPDX-License-Identifier: MIT

!> \file
!> The camp_test_aero_rep_data program

!> Test class for the aero_rep_data_t extending types
program camp_test_aero_rep_data

  use camp_util,                         only: i_kind, dp, assert, &
                                              almost_equal
  use camp_property
  use camp_camp_core
  use camp_camp_state
  use camp_aero_rep_data
  use camp_aero_rep_factory
  use camp_aero_rep_modal_binned_mass
#ifdef CAMP_USE_JSON
  use json_module
#endif
#ifdef CAMP_USE_MPI
  use mpi
#endif
  use camp_mpi

  use iso_c_binding
  implicit none

  ! Test Aerosol phases
  integer(kind=i_kind), parameter :: AERO_PHASE_IDX   = 10
  integer(kind=i_kind), parameter :: AERO_PHASE_IDX_2 =  2

  !> Expected Jacobian elements for test phases
  integer(kind=i_kind), parameter :: N_JAC_ELEM   = 5
  integer(kind=i_kind), parameter :: N_JAC_ELEM_2 = 6

  !> Interface to c ODE solver and test functions
  interface
    !> Run the c function tests
    integer(kind=c_int) function run_aero_rep_modal_c_tests(solver_data, &
        state, env)  bind (c)
      use iso_c_binding
      !> Pointer to the initialized solver data
      type(c_ptr), value :: solver_data
      !> Pointer to the state array
      type(c_ptr), value :: state
      !> Pointer to the environmental state array
      type(c_ptr), value :: env
    end function run_aero_rep_modal_c_tests
  end interface

  ! New-line character
  character(len=*), parameter :: new_line = char(10)

  !> initialize mpi
  call camp_mpi_init()

  if (run_camp_aero_rep_data_tests()) then
    if (camp_mpi_rank().eq.0) write(*,*) "Aerosol representation tests - PASS"
  else
    if (camp_mpi_rank().eq.0) write(*,*) "Aerosol representation tests - FAIL"
    stop 3
  end if

  !> finalize mpi
  call camp_mpi_finalize()

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Run all camp_aero_rep_data tests
  logical function run_camp_aero_rep_data_tests() result(passed)

    use camp_camp_solver_data

    type(camp_solver_data_t), pointer :: camp_solver_data

    camp_solver_data => camp_solver_data_t()

    if (camp_solver_data%is_solver_available()) then
      ! The MPI tests only involve packing and unpacking the aero rep
      ! from a buffer on the primary task
      if (camp_mpi_rank().eq.0) then
         passed = build_aero_rep_data_set_test()
      else
         passed = .true.
       end if
    else
      call warn_msg(770499541, "No solver available")
      passed = .true.
    end if

    deallocate(camp_solver_data)

  end function run_camp_aero_rep_data_tests

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Build aero_rep_data set
  logical function build_aero_rep_data_set_test()

    type(camp_core_t), pointer :: camp_core
    type(camp_state_t), pointer :: camp_state
    class(aero_rep_data_t), pointer :: aero_rep

#ifdef CAMP_USE_JSON

    integer(kind=i_kind) :: i_spec, j_spec, i_phase, spec_id
    character(len=:), allocatable :: rep_name, spec_name, phase_name
    type(string_t), allocatable :: file_list(:), unique_names(:)
    character(len=:), allocatable :: phase_name_test
    integer, allocatable :: last_phase_id_correct(:)
    integer, dimension(9) :: last_phase_id
    type(index_pair_t), allocatable :: adjacent_phases(:)
    character(len=:), allocatable :: phase_name_first, phase_name_second
#ifdef CAMP_USE_MPI
    type(string_t), allocatable :: rep_names(:)
    type(aero_rep_factory_t) :: aero_rep_factory
    type(aero_rep_data_ptr), allocatable :: aero_rep_passed_data_set(:)
    character, allocatable :: buffer(:)
    integer(kind=i_kind) :: pos, pack_size, i_prop, i_rep
#endif

    build_aero_rep_data_set_test = .true.

    camp_core => camp_core_t()

    allocate(file_list(1))
    file_list(1)%string = 'test_run/unit_aero_rep_data/test_aero_rep_modal_binned_mass.json'

    call camp_core%load(file_list)
    call camp_core%initialize()
    camp_state => camp_core%new_state()

    ! Check the aerosol representation getter functions
    rep_name = "my modal/binned mass aerosol rep"
    call assert_msg(745534891, camp_core%get_aero_rep(rep_name, aero_rep), rep_name)
    call assert_msg(575377987, associated(aero_rep), rep_name)
    select type (aero_rep)
      type is (aero_rep_modal_binned_mass_t)
        !> Test the adjacent phases function
        phase_name_first = "my test phase one"
        phase_name_second = "my test phase two"
        adjacent_phases = aero_rep%adjacent_phases(phase_name_first,phase_name_second)
        call assert(919038338, size(adjacent_phases) .eq. 0)

      class default
        call die_msg(570113680, rep_name)
    end select

    ! Check the unique name functions
    unique_names = aero_rep%unique_names()
    call assert_msg(282826419, allocated(unique_names), rep_name)
    call assert_msg(114575049, size(unique_names).eq.48, rep_name)
    do i_spec = 1, size(unique_names)
      spec_id = aero_rep%spec_state_id(unique_names(i_spec)%string)
      call assert_msg(339211739, spec_id.gt.0, rep_name)
      do j_spec = 1, size(unique_names)
        if (i_spec.eq.j_spec) cycle
        call assert_msg(104046435, aero_rep%spec_state_id(&
                unique_names(i_spec)%string) .ne. aero_rep%spec_state_id(&
                unique_names(j_spec)%string), rep_name)
      end do
    end do
    call assert(314287517, unique_names(1)%string.eq."mixed mode.my test phase one.species a")
    call assert(949345550, unique_names(5)%string.eq."mixed mode.my test phase two.species d")
    call assert(444139145, unique_names(8)%string.eq."single phase mode.my last test phase.species e")
    call assert(208973841, unique_names(25)%string.eq."binned aerosol.6.my test phase one.species b")
    call assert(386300586, unique_names(47)%string.eq."binned aerosol.8.my last test phase.species b")

    phase_name_test = "my last test phase"
    last_phase_id(1) = 3
    last_phase_id(2) = 12
    last_phase_id(3) = 13
    last_phase_id(4) = 14
    last_phase_id(5) = 15
    last_phase_id(6) = 16
    last_phase_id(7) = 17
    last_phase_id(8) = 18
    last_phase_id(9) = 19
    !check values
    last_phase_id_correct = aero_rep%phase_ids(phase_name_test, is_at_surface = .true.)
    call assert(087831392, last_phase_id(1) .eq. last_phase_id_correct(1))
    call assert(597703093, last_phase_id(2) .eq. last_phase_id_correct(2))
    call assert(803195649, last_phase_id(3) .eq. last_phase_id_correct(3))
    call assert(516324200, last_phase_id(4) .eq. last_phase_id_correct(4))
    call assert(160854212, last_phase_id(5) .eq. last_phase_id_correct(5))
    call assert(626595159, last_phase_id(6) .eq. last_phase_id_correct(6))
    call assert(886729286, last_phase_id(7) .eq. last_phase_id_correct(7))
    call assert(787382607, last_phase_id(8) .eq. last_phase_id_correct(8))
    call assert(132691589, last_phase_id(9) .eq. last_phase_id_correct(9))

    ! Set the species concentrations
    phase_name = "my test phase one"
    spec_name = "species a"
    unique_names = aero_rep%unique_names(phase_name = phase_name, spec_name = spec_name)
    i_spec = aero_rep%spec_state_id(unique_names(1)%string)
    call assert_msg(551866576, i_spec.gt.0, rep_name)
    camp_state%state_var(i_spec) = 1.5
    spec_name = "species b"
    unique_names = aero_rep%unique_names(phase_name = phase_name, spec_name = spec_name)
    i_spec = aero_rep%spec_state_id(unique_names(1)%string)
    call assert_msg(776503266, i_spec.gt.0, rep_name)
    camp_state%state_var(i_spec) = 2.5
    spec_name = "species c"
    unique_names = aero_rep%unique_names(phase_name = phase_name, spec_name = spec_name)
    i_spec = aero_rep%spec_state_id(unique_names(1)%string)
    call assert_msg(771238959, i_spec.gt.0, rep_name)
    camp_state%state_var(i_spec) = 3.5
    phase_name = "my test phase two"
    spec_name = "species c"
    unique_names = aero_rep%unique_names(phase_name = phase_name, spec_name = spec_name)
    i_spec = aero_rep%spec_state_id(unique_names(1)%string)
    call assert_msg(660826148, i_spec.gt.0, rep_name)
    camp_state%state_var(i_spec) = 4.5
    spec_name = "species d"
    unique_names = aero_rep%unique_names(phase_name = phase_name, spec_name = spec_name)
    i_spec = aero_rep%spec_state_id(unique_names(1)%string)
    call assert_msg(373086592, i_spec.gt.0, rep_name)
    camp_state%state_var(i_spec) = 5.5
    spec_name = "species e"
    unique_names = aero_rep%unique_names(phase_name = phase_name, spec_name = spec_name)
    i_spec = aero_rep%spec_state_id(unique_names(1)%string)
    call assert_msg(432830685, i_spec.gt.0, rep_name)
    camp_state%state_var(i_spec) = 6.5
    phase_name = "my last test phase"
    spec_name = "species b"
    unique_names = aero_rep%unique_names(phase_name = phase_name, spec_name = spec_name)
    i_spec = aero_rep%spec_state_id(unique_names(1)%string)
    call assert_msg(145091129, i_spec.gt.0, rep_name)
    camp_state%state_var(i_spec) = 7.5
    spec_name = "species e"
    unique_names = aero_rep%unique_names(phase_name = phase_name, spec_name = spec_name)
    i_spec = aero_rep%spec_state_id(unique_names(1)%string)
    call assert_msg(539884723, i_spec.gt.0, rep_name)
    camp_state%state_var(i_spec) = 8.5

    ! Check jacobian sizes
    call assert(734627842, aero_rep%num_jac_elem( 1 ) .eq. 6 )
    call assert(564470938, aero_rep%num_jac_elem( 2 ) .eq. 6 )
    call assert(676789283, aero_rep%num_jac_elem( 3 ) .eq. 2 )
    do i_phase = 4, 19
      call assert(239497756, aero_rep%num_jac_elem( i_phase ) .eq. 5 )
    end do

    aero_rep => null()

    rep_name = "AERO_REP_BAD_NAME"
    call assert(654108602, .not.camp_core%get_aero_rep(rep_name, aero_rep))
    call assert(366369046, .not.associated(aero_rep))


#ifdef CAMP_USE_MPI
    allocate(rep_names(1))
    rep_names(1)%string = "my modal/binned mass aerosol rep"
    pack_size = 0
    do i_rep = 1, size(rep_names)
      call assert(778520709, &
              camp_core%get_aero_rep(rep_names(i_rep)%string, aero_rep))
      pack_size = pack_size + aero_rep_factory%pack_size(aero_rep, MPI_COMM_WORLD)
    end do
    allocate(buffer(pack_size))
    pos = 0
    do i_rep = 1, size(rep_names)
      call assert(543807700, &
              camp_core%get_aero_rep(rep_names(i_rep)%string, aero_rep))
      call aero_rep_factory%bin_pack(aero_rep, buffer, pos, MPI_COMM_WORLD)
    end do
    allocate(aero_rep_passed_data_set(size(rep_names)))
    pos = 0
    do i_rep = 1, size(rep_names)
      aero_rep_passed_data_set(i_rep)%val => &
              aero_rep_factory%bin_unpack(buffer, pos, MPI_COMM_WORLD)
    end do
    do i_rep = 1, size(rep_names)
      associate (passed_aero_rep => aero_rep_passed_data_set(i_rep)%val)
        call assert(744932979, &
                camp_core%get_aero_rep(rep_names(i_rep)%string, aero_rep))
        call assert(860610097, size(aero_rep%condensed_data_real) .eq. &
                size(passed_aero_rep%condensed_data_real))
        do i_prop = 1, size(aero_rep%condensed_data_real)
          call assert(205686549, aero_rep%condensed_data_real(i_prop).eq. &
                  passed_aero_rep%condensed_data_real(i_prop))
        end do
        call assert(823663594, size(aero_rep%condensed_data_int) .eq. &
                size(passed_aero_rep%condensed_data_int))
        do i_prop = 1, size(aero_rep%condensed_data_int)
          call assert(318457189, aero_rep%condensed_data_int(i_prop).eq. &
                  passed_aero_rep%condensed_data_int(i_prop))
        end do
      end associate
    end do

    aero_rep => null()

    deallocate(buffer)
    deallocate(rep_names)
    deallocate(aero_rep_passed_data_set)
#endif

    ! Evaluate the aerosol representation c functions
    build_aero_rep_data_set_test = eval_c_func(camp_core)

    deallocate(file_list)
    deallocate(camp_state)
    deallocate(camp_core)

#endif

  end function build_aero_rep_data_set_test

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Evaluate aerosol representation c functions
  logical function eval_c_func(camp_core) result(passed)

    !> CAMP-core
    type(camp_core_t), intent(inout) :: camp_core

    class(aero_rep_data_t), pointer :: aero_rep
    type(camp_state_t), pointer :: camp_state
    integer(kind=i_kind), allocatable :: phase_ids(:)
    character(len=:), allocatable :: rep_name, phase_name

    character(len=:), allocatable :: section_name
    integer(kind=i_kind) :: i_sect_mixed, i_sect_single
    type(aero_rep_factory_t) :: aero_rep_factory
    type(aero_rep_update_data_modal_binned_mass_GMD_t) :: update_data_GMD
    type(aero_rep_update_data_modal_binned_mass_GSD_t) :: update_data_GSD

    rep_name = "my modal/binned mass aerosol rep"
    call assert_msg(940125461, camp_core%get_aero_rep(rep_name, aero_rep),  &
                    rep_name)
    call assert_msg(636914093, associated(aero_rep), rep_name)

    ! Set the aerosol representation id
    select type (aero_rep)
      type is (aero_rep_modal_binned_mass_t)
        call camp_core%initialize_update_object(aero_rep, update_data_GMD)
        call camp_core%initialize_update_object(aero_rep, update_data_GSD)
      class default
        call die_msg(587271916, rep_name)
    end select

    ! Initialize the solver
    call camp_core%solver_initialize()

    camp_state => camp_core%new_state()

    ! Update the GMD and GSD for the two modes
    select type (aero_rep)
      type is (aero_rep_modal_binned_mass_t)
        section_name = "mixed mode"
        call assert_msg(207793351, &
                        aero_rep%get_section_id(section_name, i_sect_mixed), &
                        "Could not get section id for the mixed mode")
        call update_data_GMD%set_GMD(i_sect_mixed, 1.2d-6)
        call update_data_GSD%set_GSD(i_sect_mixed, 1.2d0)
        call camp_core%update_data(update_data_GMD)
        call camp_core%update_data(update_data_GSD)
        call assert_msg(937636446, &
                        aero_rep%get_section_id("single phase mode", &
                                                 i_sect_single), &
                        "Could not get section id for the single phase mode")
        call update_data_GMD%set_GMD(i_sect_single, 9.3d-7)
        call update_data_GSD%set_GSD(i_sect_single, 0.9d0)
        call camp_core%update_data(update_data_GMD)
        call camp_core%update_data(update_data_GSD)
      class default
        call die_msg(570113680, rep_name)
    end select

    ! Tests will use bin 4 phase one
    phase_name = "my test phase one"
    phase_ids = aero_rep%phase_ids(phase_name)

    ! Check the number of Jacobian elements for the test phases
    call assert_msg(344638868, &
                    aero_rep%num_jac_elem(AERO_PHASE_IDX) .eq. N_JAC_ELEM, &
                    "Test phase 1 number of Jacobian element mismatch")
    call assert_msg(225602977, &
                    aero_rep%num_jac_elem(AERO_PHASE_IDX_2) &
                        .eq. N_JAC_ELEM_2, &
                    "Test phase 2 number of Jacobian element mismatch")
    camp_state%state_var(:) = 0.0;
    call camp_state%env_states(1)%set_temperature_K(  298.0d0 )
    call camp_state%env_states(1)%set_pressure_Pa( 101325.0d0 )

    passed = run_aero_rep_modal_c_tests(                              &
                         camp_core%solver_data_gas_aero%solver_c_ptr, &
                         c_loc(camp_state%state_var),                 &
                         c_loc(camp_state%env_var)                    &
                        ) .eq. 0

    deallocate(camp_state)
    deallocate(phase_ids)

  end function eval_c_func

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end program camp_test_aero_rep_data
