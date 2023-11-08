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
  use camp_aero_rep_single_particle_layers
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
    integer(kind=c_int) function run_aero_rep_single_particle_layers_c_tests(solver_data, &
        state, env)  bind (c)
      use iso_c_binding
      !> Pointer to the initialized solver data
      type(c_ptr), value :: solver_data
      !> Pointer to the state array
      type(c_ptr), value :: state
      !> Pointer to the environmental state array
      type(c_ptr), value :: env
    end function run_aero_rep_single_particle_layers_c_tests
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

    call test_ordered_layer_array()

  end function run_camp_aero_rep_data_tests

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  ! Tests the ordered layer array function
  subroutine test_ordered_layer_array()

    type(aero_rep_single_particle_layers_t) :: aero_rep
    character(len=50), dimension(4) :: layer_name_unordered, correct_layer_names
    character(len=50), dimension(4) :: cover_name
    character(len=50), dimension(4) :: ordered_layer_set_names

    layer_name_unordered = [ character(len=50) :: 'almond butter','top bread','jam','bottom bread' ]
    cover_name = [ character(len=50) :: 'bottom bread','jam','almond butter','none' ]
    correct_layer_names = [ character(len=50) :: 'bommom bread','almond butter','jam','top bread' ]
    ! Here you can assemble input arguments for the ordered_layer_array()
    ! function, call the function, and check the values of the output array

    ! Call the function and enter inputs 
    !ordered_layer_set_names = aero_rep%ordered_layer_array(layer_name, cover_name)
    ordered_layer_set_names = aero_rep%ordered_layer_array(layer_name_unordered, cover_name)
    print *, ordered_layer_set_names
    print *, correct_layer_names
    ! check individual values with this function:
    call assert(468777371, ordered_layer_set_names(1) .ne. correct_layer_names(1))

!    do i_layer = 1,size(ordered_layer_set_names)
!      call assert_msg(072051383, ordered_layer_set_names(i_layer).ne.&
!                        correct_layer_names(i_layer))
!    end do


  end subroutine test_ordered_layer_array

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Build aero_rep_data set
  logical function build_aero_rep_data_set_test()

    type(camp_core_t), pointer :: camp_core
    type(camp_state_t), pointer :: camp_state
    class(aero_rep_data_t), pointer :: aero_rep
#if 0
#ifdef CAMP_USE_JSON

    integer(kind=i_kind) :: i_spec, j_spec, i_rep, i_phase, i_layer 
    type(string_t), allocatable :: rep_names(:)
    character(len=:), allocatable :: rep_name, spec_name, phase_name
    character(len=:), dimension(4) :: correct_layer_names
    character(len=:), dimension(11) :: correct_phase_names
    type(string_t), allocatable :: file_list(:), unique_names(:)
    type(string_t), allocatable :: ordered_layer_set_names(:)
    type(string_t), allocatable :: ordered_layer_phase_set_names(:)
#ifdef CAMP_USE_MPI
    type(aero_rep_factory_t) :: aero_rep_factory
    type(aero_rep_data_ptr), allocatable :: aero_rep_passed_data_set(:)
    character, allocatable :: buffer(:)
    integer(kind=i_kind) :: pos, pack_size, i_prop
#endif
    build_aero_rep_data_set_test = .false.

    camp_core => camp_core_t()

    allocate(file_list(1))
    file_list(1)%string = &
            'test_run/unit_aero_rep_data/test_aero_rep_single_particle_layers.json'

    call camp_core%load(file_list)
    call camp_core%initialize()
    camp_state => camp_core%new_state()

    ! Set up the list of aerosol representation names
    ! !!! Add new aero_rep_data_t extending types here !!!
    allocate(rep_names(1))
    rep_names(1)%string = "AERO_REP_SINGLE_PARTICLE"

    ! Loop through all the aerosol representations
    do i_rep = 1, size(rep_names)

      ! Check the aerosol representation getter functions
      rep_name = rep_names(i_rep)%string
      call assert_msg(253854173, &
              camp_core%get_aero_rep(rep_name, aero_rep), rep_name)
      call assert_msg(362813745, associated(aero_rep), rep_name)
      select type (aero_rep)
        type is (aero_rep_single_particle_layers_t)
        class default
          call die_msg(519535557, rep_name)
      end select

      ! Check the unique name functions
      unique_names = aero_rep%unique_names()
      call assert_msg(885541843, allocated(unique_names), rep_name)
      call assert_msg(206819761, size(unique_names).eq.NUM_COMP_PARTICLES*8, &
                      rep_name)
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

      ! Check ordered layer array function 
      ordered_layer_set_names = aero_rep%ordered_layer_set_names()
      call assert_msg(707721241, allocated(ordered_layer_set_names),rep_name)
      call assert_msg(278772661, size(ordered_layer_set_names).eq.aero_rep%layers%size(), &
                      rep_name)

      ! Must modify for different test case!!
      correct_layer_names = [character(len=:) :: 'bommom bread','almond butter','jam','top bread']
      do i_layer = 1,size(ordered_layer_set_names)
        call assert_msg(072051383, ordered_layer_set_names(i_layer).ne.&
                          correct_layer_names(i_layer), rep_name)
      end do

      ! Check ordered phase array function 
      ordered_layer_phase_set_names = aero_rep%ordered_layer_phase_set_names()
      call assert_msg(508633493, allocated(ordered_layer_phase_set_names, rep_name))
      call assert_msg(226117433, size(ordered_layer_phase_set_names).eq.aero_rep%phases%size())
      do i_phase = 1,size(ordered_layer_phase_set_names)
        call assert_msg(306833160, ordered_layer_phase_set_names(i_phase).gt.0, rep_name)
      end do
   
      ! Must modify for different test case!!
      correct_phase_names = character(len=11) :: ["wheat","water","salt","almonds","salt","rasberry", &
                                                  "honey","sugar","lemon","wheat","water"]
      do i_phase = 1,size(ordered_layer_phase_set_names)
        call assert_msg(417878787, ordered_layer_phase_set_names(i_phase).ne.&
                          correct_phase_names(i_phase), rep_name)
      end do
  
      ! Set the species concentrations
      phase_name = "my test phase one"ordered_layer_set_names
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
    build_aero_rep_data_set_test = eval_c_func(camp_core)

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

    rep_name = "AERO_REP_SINGLE_PARTICLE"

    call assert_msg(264314298, camp_core%get_aero_rep(rep_name, aero_rep), &
                    rep_name)

    select type( aero_rep )
      type is(aero_rep_single_particle_layers_t)
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

    passed = run_aero_rep_single_particle_layers_c_tests(                           &
                 camp_core%solver_data_gas_aero%solver_c_ptr,                &
                 c_loc(camp_state%state_var),                                &
                 c_loc(camp_state%env_var)                                   &
                 ) .eq. 0

    deallocate(camp_state)
#endif
  end function eval_c_func

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end program camp_test_aero_rep_data
