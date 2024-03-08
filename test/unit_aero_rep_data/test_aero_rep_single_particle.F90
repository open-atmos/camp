! Copyright (C) 2021 Barcelona Supercomputing Center and University of
! Illinois at Urbana-Champaign
! SPDX-License-Identifier: MIT

!> \file
!> The camp_test_aero_rep_data program

!> Test class for the aero_rep_data_t extending types
program camp_test_aero_rep_data

#ifdef CAMP_USE_JSON
  use json_module
#endif
#ifdef CAMP_USE_MPI
  use mpi
#endif
  use camp_aero_rep_data
  use camp_aero_rep_factory
  use camp_aero_rep_single_particle
  use camp_mpi
  use camp_camp_core
  use camp_camp_state
  use camp_property
  use camp_util,                         only: i_kind, dp, assert, &
                                              almost_equal

  use iso_c_binding
  implicit none

  ! Test computational particle
  integer(kind=i_kind), parameter :: TEST_PARTICLE = 2

  ! Total computational particles
  integer(kind=i_kind), parameter :: NUM_COMP_PARTICLES = 3

  ! Number of aerosol phases per particle
  integer(kind=i_kind), parameter :: NUM_AERO_PHASE = 3

  ! Index for the test phase (test-particle phase 2)
  integer(kind=i_kind), parameter :: AERO_PHASE_IDX = ((TEST_PARTICLE-1)*NUM_AERO_PHASE+2)

  ! Number of expected Jacobian elements for the test phase
  integer(kind=i_kind), parameter :: NUM_JAC_ELEM = 8

  ! Externally set properties
  real(kind=dp), parameter :: PART_NUM_CONC = 1.23e3
  real(kind=dp), parameter :: PART_RADIUS   = 2.43e-7

  !> Interface to c ODE solver and test functions
  interface
    !> Run the c function tests
    integer(kind=c_int) function run_aero_rep_single_particle_c_tests(solver_data, &
        state, env)  bind (c)
      use iso_c_binding
      !> Pointer to the initialized solver data
      type(c_ptr), value :: solver_data
      !> Pointer to the state array
      type(c_ptr), value :: state
      !> Pointer to the environmental state array
      type(c_ptr), value :: env
    end function run_aero_rep_single_particle_c_tests
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
      call warn_msg(594028423, "No solver available")
      passed = .true.
    end if

    deallocate(camp_solver_data)

    call test_ordered_layer_ids()
    call test_config_read()

  end function run_camp_aero_rep_data_tests

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  ! Tests the ordered layer array function
  subroutine test_ordered_layer_ids()

    use camp_aero_rep_single_particle, only : ordered_layer_ids

    type(string_t), dimension(4) :: layer_names, correct_layer_names
    type(string_t), dimension(4) :: cover_names
    !type(string_t), allocatable :: layer_names(:), correct_layer_names(:)
    !type(string_t), allocatable :: cover_names(:)
    integer, allocatable :: ordered_ids(:)

    ! Assemble input arguments
    layer_names(1)%string = 'almond butter'
    layer_names(2)%string = 'top bread'
    layer_names(3)%string = 'jam'
    layer_names(4)%string = 'bottom bread'
    cover_names(1)%string = 'bottom bread'
    cover_names(2)%string = 'jam'
    cover_names(3)%string = 'almond butter'
    cover_names(4)%string = 'none'
    correct_layer_names(1)%string = 'bottom bread'
    correct_layer_names(2)%string = 'almond butter'
    correct_layer_names(3)%string = 'jam'
    correct_layer_names(4)%string = 'top bread'

    !layer_names(1)%string = 'one layer'
    !correct_layer_names(1)%string = 'one layer'
    ! Call the function and enter inputs 
    ordered_ids = ordered_layer_ids(layer_names, cover_names)
    print *, size(ordered_ids)
    ! check individual values:
    call assert(476179048, size(ordered_ids) .eq. 4)
    call assert(903386486, layer_names(ordered_ids(1))%string .eq. correct_layer_names(1)%string)
    call assert(468777371, layer_names(ordered_ids(2))%string .eq. correct_layer_names(2)%string)
    call assert(487966491, layer_names(ordered_ids(3))%string .eq. correct_layer_names(3)%string)
    call assert(721784428, layer_names(ordered_ids(4))%string .eq. correct_layer_names(4)%string)
  end subroutine test_ordered_layer_ids

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> Test that a configuration ends up generating unique names in the correct
  !! order
  subroutine test_config_read()
#ifdef CAMP_USE_JSON

    type(camp_core_t), pointer :: camp_core
    class(aero_rep_data_t), pointer :: aero_rep

    type(string_t), allocatable :: file_list(:), unique_names(:)
    character(len=:), allocatable :: rep_name, phase_name_test
    integer :: i_name, num_bread, max_part, bread_phase_instance

    ! create the camp core from test data
    allocate(file_list(1))
    file_list(1)%string = &
            'test_run/unit_aero_rep_data/test_aero_rep_single_particle.json'
    camp_core => camp_core_t()
    call camp_core%load(file_list)
    call camp_core%initialize()

    ! get the aerosol representation to test
    rep_name = "single-particle with layers"
    call assert_msg(773918149, &
              camp_core%get_aero_rep(rep_name, aero_rep), rep_name)
    call assert_msg(372407009, associated(aero_rep), rep_name)
    select type (aero_rep)
      type is (aero_rep_single_particle_t)

        ! check dimensions of the system
        call assert(118129615, aero_rep%num_layers() .eq. 3)
        call assert(792039685, aero_rep%num_phases() .eq. 4)
        call assert(958837816, aero_rep%num_phases(1) .eq. 1)
        call assert(230900255, aero_rep%num_phases(2) .eq. 2)
        call assert(113317603, aero_rep%num_phases(3) .eq. 1)
        call assert(902904791, aero_rep%phase_state_size() .eq. 12)
        call assert(154362297, aero_rep%phase_state_size(layer=1) .eq. 3)
        call assert(266680642, aero_rep%phase_state_size(layer=2) .eq. 6)
        call assert(714048488, aero_rep%phase_state_size(layer=3) .eq. 3)
        call assert(326424735, aero_rep%phase_state_size(layer=1,phase=1) .eq. 3)
        call assert(491317332, aero_rep%phase_state_size(layer=2,phase=1) .eq. 4)
        call assert(603635677, aero_rep%phase_state_size(layer=2,phase=2) .eq. 2)
        call assert(768528274, aero_rep%phase_state_size(layer=3,phase=1) .eq. 3)

        ! test num_phase_instances funtion
        phase_name_test = "bread"
        num_bread = 2
        max_part = aero_rep%maximum_computational_particles()        
        bread_phase_instance = num_bread * max_part
        !check value
        call assert(417730478, 3 .eq. max_part)
        call assert(734138496, bread_phase_instance .eq. aero_rep%num_phase_instances(phase_name_test))
        
      class default
        call die_msg(519535557, rep_name)
    end select
    
    unique_names = aero_rep%unique_names()
    call assert(551749649, size( unique_names ) .eq. 36)
#if 0
    call assert(112328243, unique_names(1)%string .eq. "P1.one layer.my test phase one.species a")
    call assert(124534442, unique_names(2)%string .eq. "P1.one layer.my test phase one.species b")
    call assert(519328036, unique_names(3)%string .eq. "P1.one layer.my test phase one.species c")
    call assert(349171133, unique_names(4)%string .eq. "P1.one layer.my test phase two.species c")
    call assert(179014228, unique_names(5)%string .eq. "P1.one layer.my test phase two.species d")
    call assert(626382071, unique_names(6)%string .eq. "P1.one layer.my test phase two.species e")
    call assert(173749920, unique_names(7)%string .eq. "P1.one layer.my last test phase.species b")
    call assert(286068262, unique_names(8)%string .eq. "P1.one layer.my last test phase.species e")
    call assert(112328256, unique_names(9)%string .eq. "P2.one layer.my test phase one.species a")
    call assert(124534432, unique_names(10)%string .eq. "P2.one layer.my test phase one.species b")
    call assert(519328055, unique_names(11)%string .eq. "P2.one layer.my test phase one.species c")
    call assert(349171155, unique_names(12)%string .eq. "P2.one layer.my test phase two.species c")
    call assert(179014255, unique_names(13)%string .eq. "P2.one layer.my test phase two.species d")
    call assert(626382000, unique_names(14)%string .eq. "P2.one layer.my test phase two.species e")
    call assert(173749998, unique_names(15)%string .eq. "P2.one layer.my last test phase.species b")
    call assert(286068243, unique_names(16)%string .eq. "P2.one layer.my last test phase.species e")
    call assert(112328200, unique_names(17)%string .eq. "P3.one layer.my test phase one.species a")
    call assert(124534411, unique_names(18)%string .eq. "P3.one layer.my test phase one.species b")
    call assert(519328088, unique_names(19)%string .eq. "P3.one layer.my test phase one.species c")
    call assert(349171189, unique_names(20)%string .eq. "P3.one layer.my test phase two.species c")
    call assert(179014263, unique_names(21)%string .eq. "P3.one layer.my test phase two.species d")
    call assert(626382009, unique_names(22)%string .eq. "P3.one layer.my test phase two.species e")
    call assert(173749900, unique_names(23)%string .eq. "P3.one layer.my last test phase.species b")
    call assert(286068230, unique_names(24)%string .eq. "P3.one layer.my last test phase.species e")
#endif
    call assert(112328249, unique_names(1)%string .eq. "P1.bottom bread.bread.wheat")
    call assert(124534441, unique_names(2)%string .eq. "P1.bottom bread.bread.water")
    call assert(519328035, unique_names(3)%string .eq. "P1.bottom bread.bread.salt")
    call assert(349171131, unique_names(4)%string .eq. "P1.filling.jam.rasberry")
    call assert(179014227, unique_names(5)%string .eq. "P1.filling.jam.honey")
    call assert(626382073, unique_names(6)%string .eq. "P1.filling.jam.sugar")
    call assert(173749920, unique_names(7)%string .eq. "P1.filling.jam.lemon")
    call assert(286068265, unique_names(8)%string .eq. "P1.filling.almond butter.almonds")
    call assert(680861859, unique_names(9)%string .eq. "P1.filling.almond butter.sugar")
    call assert(293238106, unique_names(10)%string .eq. "P1.top bread.bread.wheat")
    call assert(405556451, unique_names(11)%string .eq. "P1.top bread.bread.water")
    call assert(235399547, unique_names(12)%string .eq. "P1.top bread.bread.salt")
    call assert(347717892, unique_names(13)%string .eq. "P2.bottom bread.bread.wheat")
    call assert(512610489, unique_names(14)%string .eq. "P2.bottom bread.bread.water")
    call assert(342453585, unique_names(15)%string .eq. "P2.bottom bread.bread.salt")
    call assert(737247179, unique_names(16)%string .eq. "P2.filling.jam.rasberry")
    call assert(567090275, unique_names(17)%string .eq. "P2.filling.jam.honey")
    call assert(114458122, unique_names(18)%string .eq. "P2.filling.jam.sugar")
    call assert(844301217, unique_names(19)%string .eq. "P2.filling.jam.lemon")
    call assert(109193815, unique_names(20)%string .eq. "P2.filling.almond butter.almonds")
    call assert(339094812, unique_names(21)%string .eq. "P2.filling.almond butter.sugar")
    call assert(168937908, unique_names(22)%string .eq. "P2.top bread.bread.wheat")
    call assert(616305754, unique_names(23)%string .eq. "P2.top bread.bread.water")
    call assert(676049847, unique_names(24)%string .eq. "P2.top bread.bread.salt")
    call assert(505892943, unique_names(25)%string .eq. "P3.bottom bread.bread.wheat")
    call assert(670785540, unique_names(26)%string .eq. "P3.bottom bread.bread.water")
    call assert(900686537, unique_names(27)%string .eq. "P3.bottom bread.bread.salt")
    call assert(730529633, unique_names(28)%string .eq. "P3.filling.jam.rasberry")
    call assert(612946981, unique_names(29)%string .eq. "P3.filling.jam.honey")
    call assert(225323228, unique_names(30)%string .eq. "P3.filling.jam.sugar")
    call assert(672691074, unique_names(31)%string .eq. "P3.filling.jam.lemon")
    call assert(837583671, unique_names(32)%string .eq. "P3.filling.almond butter.almonds")
    call assert(102476269, unique_names(33)%string .eq. "P3.filling.almond butter.sugar")
    call assert(614852515, unique_names(34)%string .eq. "P3.top bread.bread.wheat")
    call assert(779745112, unique_names(35)%string .eq. "P3.top bread.bread.water")
    call assert(109646110, unique_names(36)%string .eq. "P3.top bread.bread.salt")
#endif
  end subroutine test_config_read

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Build aero_rep_data set
  logical function build_aero_rep_data_set_test()
    type(camp_core_t), pointer :: camp_core
    type(camp_state_t), pointer :: camp_state
    class(aero_rep_data_t), pointer :: aero_rep
#if 0
#ifdef CAMP_USE_JSON

    integer(kind=i_kind) :: i_spec, j_spec, i_rep, i_phase
    type(string_t), allocatable :: rep_names(:)
    character(len=:), allocatable :: rep_name, spec_name, phase_name
    type(string_t), allocatable :: file_list(:), unique_names(:)
#ifdef CAMP_USE_MPI
    type(aero_rep_factory_t) :: aero_rep_factory
    type(aero_rep_data_ptr), allocatable :: aero_rep_passed_data_set(:)
    character, allocatable :: buffer(:)
    integer(kind=i_kind) :: pos, pack_size, i_prop
#endif

    build_aero_rep_data_set_test = .false.


    allocate(file_list(1))
    file_list(1)%string = &
            'test_run/unit_aero_rep_data/test_aero_rep_single_particle.json'
    camp_core => camp_core_t()
    call camp_core%load(file_list)
    call camp_core%initialize()
    camp_state => camp_core%new_state()

    ! Set up the list of aerosol representation names
    ! !!! Add new aero_rep_data_t extending types here !!!
    allocate(rep_names(1))
    rep_names(1)%string = "single-particle with layers"

    ! Loop through all the aerosol representations
    do i_rep = 1, size(rep_names)

      ! Check the aerosol representation getter functions
      rep_name = rep_names(i_rep)%string
      call assert_msg(253854173, &
              camp_core%get_aero_rep(rep_name, aero_rep), rep_name)
      call assert_msg(362813745, associated(aero_rep), rep_name)
      select type (aero_rep)
        type is (aero_rep_single_particle_t)
        class default
          call die_msg(519535557, rep_name)
      end select

      ! Check the unique name functions
      unique_names = aero_rep%unique_names()
      call assert_msg(885541843, allocated(unique_names), rep_name)
      !call assert_msg(206819761, size(unique_names).eq.NUM_COMP_PARTICLES*8, &
      !                rep_name)
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
      unique_names = aero_rep%unique_names(phase_name = phase_name, &
              spec_name = spec_name)
      i_spec = aero_rep%spec_state_id(unique_names(1)%string)
      call assert_msg(258227897, i_spec.gt.0, rep_name)
      camp_state%state_var(i_spec) = 1.5
      spec_name = "species b"
      unique_names = aero_rep%unique_names(phase_name = phase_name, &
              spec_name = spec_name)
      i_spec = aero_rep%spec_state_id(unique_names(1)%string)
      call assert_msg(418308482, i_spec.gt.0, rep_name)
      camp_state%state_var(i_spec) = 2.5
      spec_name = "species c"
      unique_names = aero_rep%unique_names(phase_name = phase_name, &
              spec_name = spec_name)
      i_spec = aero_rep%spec_state_id(unique_names(1)%string)
      call assert_msg(420214016, i_spec.gt.0, rep_name)
      camp_state%state_var(i_spec) = 3.5
      phase_name = "my test phase two"
      spec_name = "species c"
      unique_names = aero_rep%unique_names(phase_name = phase_name, &
              spec_name = spec_name)
      i_spec = aero_rep%spec_state_id(unique_names(1)%string)
      call assert_msg(416855243, i_spec.gt.0, rep_name)
      camp_state%state_var(i_spec) = 4.5
      spec_name = "species d"
      unique_names = aero_rep%unique_names(phase_name = phase_name, &
              spec_name = spec_name)
      i_spec = aero_rep%spec_state_id(unique_names(1)%string)
      call assert_msg(578389067, i_spec.gt.0, rep_name)
      camp_state%state_var(i_spec) = 5.5
      spec_name = "species e"
      unique_names = aero_rep%unique_names(phase_name = phase_name, &
              spec_name = spec_name)
      i_spec = aero_rep%spec_state_id(unique_names(1)%string)
      call assert_msg(147314014, i_spec.gt.0, rep_name)
      camp_state%state_var(i_spec) = 6.5
      phase_name = "my last test phase"
      spec_name = "species b"
      unique_names = aero_rep%unique_names(phase_name = phase_name, &
              spec_name = spec_name)
      i_spec = aero_rep%spec_state_id(unique_names(1)%string)
      call assert_msg(401514617, i_spec.gt.0, rep_name)
      camp_state%state_var(i_spec) = 7.5
      spec_name = "species e"
      unique_names = aero_rep%unique_names(phase_name = phase_name, &
              spec_name = spec_name)
      i_spec = aero_rep%spec_state_id(unique_names(1)%string)
      call assert_msg(291101806, i_spec.gt.0, rep_name)
      camp_state%state_var(i_spec) = 8.5

    end do

    rep_name = "AERO_REP_BAD_NAME"
    call assert(676257369, .not.camp_core%get_aero_rep(rep_name, aero_rep))
    call assert(453526213, .not.associated(aero_rep))
#ifdef CAMP_USE_MPI
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
    deallocate(aero_rep_passed_data_set)

#endif

    ! Evaluate the aerosol representation c functions
    !build_aero_rep_data_set_test = eval_c_func(camp_core)

    deallocate(camp_state)
    deallocate(camp_core)
#endif
#endif
  end function build_aero_rep_data_set_test

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Evaluate the aerosol representation c functions
  logical function eval_c_func(camp_core) result(passed)
    !> CAMP-core
    type(camp_core_t), intent(inout) :: camp_core
#if 0
    class(aero_rep_data_t), pointer :: aero_rep
    type(camp_state_t), pointer :: camp_state
    integer(kind=i_kind), allocatable :: phase_ids(:)
    character(len=:), allocatable :: rep_name, phase_name
    type(aero_rep_factory_t) :: aero_rep_factory
    type(aero_rep_update_data_single_particle_number_t) :: update_number

    rep_name = "single-particle with layers"

    call assert_msg(264314298, camp_core%get_aero_rep(rep_name, aero_rep), &
                    rep_name)

    select type( aero_rep )
      type is(aero_rep_single_particle_t)
        call camp_core%initialize_update_object( aero_rep, update_number )
      class default
        call die_msg(766425873, "Wrong aero rep type")
    end select

    call camp_core%solver_initialize()

    camp_state => camp_core%new_state()

    camp_state%state_var(:) = 0.0
    call camp_state%env_states(1)%set_temperature_K(  298.0d0 )
    call camp_state%env_states(1)%set_pressure_Pa( 101325.0d0 )

    ! Check the number of Jacobian elements for the test phase
    call assert_msg(153137613, &
                    aero_rep%num_jac_elem(AERO_PHASE_IDX) .eq. NUM_JAC_ELEM, &
                    rep_name)

    ! Update external properties
    call update_number%set_number__n_m3( TEST_PARTICLE, 12.3d0 )
    call camp_core%update_data( update_number )

    ! Test re-setting number concentration
    call update_number%set_number__n_m3( TEST_PARTICLE, PART_NUM_CONC )
    call camp_core%update_data( update_number )

    passed = run_aero_rep_single_particle_c_tests(                           &
                 camp_core%solver_data_gas_aero%solver_c_ptr,                &
                 c_loc(camp_state%state_var),                                &
                 c_loc(camp_state%env_var)                                   &
                 ) .eq. 0

    deallocate(camp_state)
#endif
  end function eval_c_func

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
end program camp_test_aero_rep_data
