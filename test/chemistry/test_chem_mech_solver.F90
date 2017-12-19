! Copyright (C) 2017 Matt Dawson
! Licensed under the GNU General Public License version 2 or (at your
! option) any later version. See the file COPYING for details.

!> \file
!> The pmc_test_chem_mech_solver program

!> Integration test for the chemical mechanism module's solver
program pmc_test_chem_mech_solver

  use pmc_util,                         only: i_kind, dp, assert, &
                                              almost_equal, string_t, &
                                              warn_msg
  use pmc_model_data
  use pmc_model_state
#ifdef PMC_USE_JSON
  use json_module
#endif

  implicit none

  ! New-line character
  character(len=*), parameter :: new_line = char(10)
  ! Number of timesteps to output in mechanisms
  integer(kind=i_kind) :: NUM_TIME_STEP = 100

  if (run_pmc_chem_mech_solver_tests()) then
    write(*,*) "Mechanism solver tests - PASS"
  else
    write(*,*) "Mechanism solver tests - FAIL"
  end if

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Run all pmc_chem_mech_solver tests
  logical function run_pmc_chem_mech_solver_tests() result(passed)

    use pmc_integration_data,  only: pmc_integration_data_is_solver_available

    if (pmc_integration_data_is_solver_available()) then
      passed = run_consecutive_mech_test()
    else
      passed = .true.
      call warn_msg(465633118, "No solver available for mechanism integration")
    end if

  end function run_pmc_chem_mech_solver_tests

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Solve a mechanism of consecutive reactions
  !!
  !! The mechanism is of the form:
  !!
  !!   A -k1-> B -k2-> C
  !!
  !! where k1 and k2 are Arrhenius reaction rate constants:
  !!
  !!  k = A * exp( -Ea / (k_b * temp) )
  !!
  logical function run_consecutive_mech_test()

    use pmc_constants

    type(model_data_t), pointer :: model_data
    type(model_state_t), target :: model_state
    type(string_t), allocatable, dimension(:) :: input_file_path, output_file_path

    real(kind=dp), dimension(0:NUM_TIME_STEP, 3) :: model_conc, true_conc
    integer(kind=i_kind) :: idx_A, idx_B, idx_C
    character(len=:), allocatable :: key
    integer(kind=i_kind) :: i_time, i_spec
    real(kind=dp) :: time_step

    ! Parameters for calculating true concentrations
    real(kind=dp) :: k1, k2, temp, Ea1, Ea2, A1, A2, time

    run_consecutive_mech_test = .true.

    ! Set the rate constants (for calculating the true value)
    temp = 298.0
    A1 = 12.0
    Ea1 = 1.0e-20
    A2 = 13.0
    Ea2 = 2.0e-20
    k1 = A1 * exp( -Ea1 / (const%boltzmann * temp) )
    k2 = A2 * exp( -Ea2 / (const%boltzmann * temp) )

    ! Set output time step (s)
    time_step = 0.1

    ! Get the consecutive-rxn mechanism json file
    allocate(input_file_path(1))
    input_file_path(1)%string = 'consecutive.json'

    ! Construct a model_data variable
    model_data => model_data_t(input_file_path)

    ! Initialize the model
    call model_data%initialize()

    ! Get a model state variable
    model_state = model_data%new_state()

    ! Set the environmental conditions
    model_state%env_state%temp = temp

    ! Get species indices
    key = "A"
    idx_A = model_data%chem_spec_data%state_id(key);
    key = "B"
    idx_B = model_data%chem_spec_data%state_id(key);
    key = "C"
    idx_C = model_data%chem_spec_data%state_id(key);

    ! Make sure the expected species are in the model
    call assert(629811894, idx_A.gt.0)
    call assert(226395220, idx_B.gt.0)
    call assert(338713565, idx_C.gt.0)

    ! Save the initial concentrations
    true_conc(0,idx_A) = 1.0
    true_conc(0,idx_B) = 0.0
    true_conc(0,idx_C) = 0.0
    model_conc(0,:) = true_conc(0,:)

    ! Set the initial concentrations in the model
    model_state%chem_spec_state%conc(:) = model_conc(0,:)

    ! Integrate the mechanism
    do i_time = 1, NUM_TIME_STEP

      ! Get the modeled conc
      call model_data%solve(model_state, time_step)
      model_conc(i_time,:) = model_state%chem_spec_state%conc(:)

      ! Get the analytic conc
      time = i_time * time_step
      true_conc(i_time,idx_A) = true_conc(0,idx_A) * exp(-k1*time)
      true_conc(i_time,idx_B) = true_conc(0,idx_A) * (k1/(k2-k1)) * &
              (exp(-k1*time) - exp(-k2*time))
      true_conc(i_time,idx_C) = true_conc(0,idx_A) * &
             (1.0 + (k1*exp(-k2*time) - k2*exp(-k1*time))/(k2-k1))

    end do

    ! Save the results
    open(unit=7, file="out/consecutive_results.txt", status="replace", action="write")
    do i_time = 0, NUM_TIME_STEP
      write(7,*) i_time*time_step, &
            ' ', true_conc(i_time, idx_A),' ', model_conc(i_time, idx_A), &
            ' ', true_conc(i_time, idx_B),' ', model_conc(i_time, idx_B), &
            ' ', true_conc(i_time, idx_C),' ', model_conc(i_time, idx_C)
    end do
    close(7)

    ! Analyze the results
    do i_time = 1, NUM_TIME_STEP
      do i_spec = 1, size(model_conc, 2)
        call assert(848069355, &
          almost_equal(model_conc(i_time, i_spec), true_conc(i_time, i_spec), &
          real(1.0e-5, kind=dp)))
      end do
    end do

    run_consecutive_mech_test = .true.

  end function run_consecutive_mech_test

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end program pmc_test_chem_mech_solver