! Copyright (C) 2021 Barcelona Supercomputing Center and University of
! Illinois at Urbana-Champaign
! SPDX-License-Identifier: MIT

!> Mock version of the MONARCH model for testing integration with CAMP
program mock_monarch_t

  use camp_constants, only: const
  use camp_util, only : assert_msg, almost_equal, to_string
  use camp_monarch_interface
  use camp_mpi
  use json_module

  implicit none

  integer(kind=i_kind), parameter :: NUM_EBI_SPEC = 72
  integer(kind=i_kind), parameter :: NUM_EBI_PHOTO_RXN = 23
  integer, parameter :: NUM_MONARCH_SPEC = 250
  integer :: NUM_VERT_CELLS
  integer :: I_W
  integer :: I_E
  integer :: I_S
  integer :: I_N
  integer :: NUM_WE_CELLS
  integer :: NUM_SN_CELLS
  integer, parameter :: START_CAMP_ID = 1!100
  integer, parameter :: END_CAMP_ID = 260!350
  real(kind=dp):: TIME_STEP ! (min)
  integer :: NUM_TIME_STEP
  integer, parameter :: WATER_VAPOR_ID = 5
  real, parameter :: START_TIME = 0
  integer :: n_cells = 1
  real,allocatable :: temperature(:, :, :)
  real, allocatable  :: species_conc(:, :, :, :)
  real, allocatable  :: water_conc(:, :, :, :) !(kg_H2O/kg_air)
  real, allocatable  :: air_density(:, :, :) !(kg_air/m^3)
  real, allocatable  :: pressure(:, :, :) !(Pa)
  real, allocatable  :: conv(:, :, :) !(mol s-1 m-2 to ppmv)
  integer :: i_hour = 0
  real :: curr_time = START_TIME
  type(camp_monarch_interface_t), pointer :: camp_interface
  character(len=:), allocatable :: camp_input_file, chemFile
  character(len=:), allocatable :: output_path, output_file_title, str_to_int_aux
  character(len=:), allocatable :: str
  character, allocatable :: buffer(:)
  integer(kind=i_kind) :: pos, pack_size, ierr
  character(len=512) :: arg
  integer :: i_time, i_spec, i_case, i, j, k, z,r
  integer :: plot_case, new_v_cells, aux_int
  type(json_file) :: jfile
  type(json_core) :: json
  character(len=:), allocatable :: export_path
  character(len=128) :: i_str
  integer :: id, n_cells_monarch, load_gpu, load_balance

  call camp_mpi_init()
  I_W=1
  I_E=1
  I_S=1
  I_N=1
  call jfile%initialize()
    CHARACTER(len=255) :: cwd
  CALL getcwd(cwd)
  WRITE(*,*) TRIM(cwd)
  export_path = "../test/binned_gpu/settings/TestMonarch"//".json"
  call jfile%load_file(export_path); if (jfile%failed()) print*,&
          "JSON not found at ",export_path
  call jfile%get('chemFile',output_file_title)
  camp_input_file = "../test/binned_gpu/settings/"//output_file_title//"/config.json"
  output_path = "out/"//output_file_title
  call jfile%get('nCells',NUM_VERT_CELLS)
  n_cells_monarch = (I_E - I_W+1)*(I_N - I_S+1)*NUM_VERT_CELLS
  call jfile%get('load_gpu',load_gpu)
  if(load_gpu == 0) then
    n_cells = 1
  else
    n_cells = n_cells_monarch
  end if
  call jfile%get('load_balance',load_balance)
  call jfile%get('timeSteps',NUM_TIME_STEP)
  call jfile%get('timeStepsDt',TIME_STEP)
  NUM_WE_CELLS = I_E-I_W+1
  NUM_SN_CELLS = I_N-I_S+1
  if (camp_mpi_rank()==0) then
    write(*,*) "Time-steps:", NUM_TIME_STEP, "Cells:",&
        NUM_WE_CELLS*NUM_SN_CELLS*NUM_VERT_CELLS, &
           " MPI processes:", camp_mpi_size()
  end if
  allocate(temperature(NUM_WE_CELLS,NUM_SN_CELLS,NUM_VERT_CELLS))
  allocate(species_conc(NUM_WE_CELLS,NUM_SN_CELLS,NUM_VERT_CELLS,NUM_MONARCH_SPEC))
  allocate(water_conc(NUM_WE_CELLS,NUM_SN_CELLS,NUM_VERT_CELLS,WATER_VAPOR_ID))
  allocate(air_density(NUM_WE_CELLS,NUM_SN_CELLS,NUM_VERT_CELLS))
  allocate(pressure(NUM_WE_CELLS,NUM_SN_CELLS,NUM_VERT_CELLS))
  allocate(conv(NUM_WE_CELLS, NUM_SN_CELLS, NUM_VERT_CELLS))

  camp_interface => camp_monarch_interface_t(camp_input_file, output_file_title, &
          START_CAMP_ID, END_CAMP_ID, n_cells, load_gpu, load_balance)
  camp_interface%camp_state%state_var(:) = 0.0
  species_conc(:,:,:,:) = 0.0
  air_density(:,:,:) = 1.225
  water_conc(:,:,:,WATER_VAPOR_ID) = 0.

  call camp_interface%get_init_conc(species_conc, water_conc, WATER_VAPOR_ID, &
          i_W,I_E,I_S,I_N)

  call set_env(camp_interface,output_path)

  if(TIME_STEP*NUM_TIME_STEP>(60*24)) then !24h limit time-step
    print*,"ERROR TIME_STEP*NUM_TIME_STEP.gt.(60*24): Reduce number of time-step or time-step size"
    STOP
  end if

  do i_time=1, NUM_TIME_STEP
    call camp_interface%integrate(curr_time,         & ! Starting time (min)
       TIME_STEP,         & ! Time step (min)
       I_W,               & ! Starting W->E grid cell
       I_E,               & ! Ending W->E grid cell
       I_S,               & ! Starting S->N grid cell
       I_N,               & ! Ending S->N grid cell
       temperature,       & ! Temperature (K)
       species_conc,      & ! Tracer array
       water_conc,        & ! Water concentrations (kg_H2O/kg_air)
       WATER_VAPOR_ID,    & ! Index in water_conc() corresponding to water vapor
       air_density,       & ! Air density (kg_air/m^3)
       pressure,          & ! Air pressure (Pa)
       conv,              &
       i_hour,&
       NUM_TIME_STEP,&
       i_time)
  curr_time = curr_time + TIME_STEP
  end do
  call camp_mpi_barrier()
  call camp_interface%camp_core%join_solver_state()
  call camp_interface%camp_core%export_solver_stats()
  call camp_mpi_barrier()
  if (camp_mpi_rank()==0) then
    write(*,*) "Model run time: ", comp_time, " s"
  end if
  call camp_mpi_finalize()
contains
  subroutine set_env(camp_interface,file_prefix)
    type(camp_monarch_interface_t), intent(inout) :: camp_interface
    character(len=:), allocatable, intent(in) :: file_prefix
    character(len=:), allocatable :: file_name
    integer :: z,z1,o,i,j,k,r,t,i_cell,i_spec,mpi_size,ncells,tid,ncells_mpi
    integer :: n_cells_print
    real :: rate_emi,temp_init,press,press_init,&
      press_end,press_range,press_slide

    temp_init = 290.016
    press_init = 100000.
    allocate(camp_interface%rate_emi(24,n_cells_monarch))
    camp_interface%rate_emi(:,:)=0.0
    ncells=(I_E - I_W+1)*(I_N - I_S+1)*NUM_VERT_CELLS
    mpi_size=camp_mpi_size()
    tid=camp_mpi_rank()
    ncells_mpi=ncells*mpi_size
    press_end = 10000.
    press_range = press_end-press_init
    if((ncells_mpi)==1) then
      press_slide = 0.
    else
      press_slide = press_range/(ncells_mpi-1)
    end if
    do i=I_W,I_E
      do j=I_S,I_N
        do k=1,NUM_VERT_CELLS
          o = (j-1)*(I_E) + (i-1)
          z1 = (k-1)*(I_E*I_N) + o
          z = tid*ncells+z1
          pressure(i,j,k)=press_init+press_slide*z
          temperature(i,j,k)=temp_init*((pressure(i,j,k)/press_init)**(287./1004.)) !dry_adiabatic formula
          rate_emi=abs((press_end-pressure(i,j,k))/press_range)
          do t=1,12 !12 first hours
            camp_interface%rate_emi(t,z1+1)=rate_emi
          end do
        end do
      end do
    end do
    air_density(:,:,:) = pressure(:,:,:)/(287.04*temperature(:,:,:)* &
            (1.+0.60813824*water_conc(:,:,:,WATER_VAPOR_ID))) !kg m-3
    conv(:,:,:)=0.02897/air_density(:,:,:)*(TIME_STEP*60.)*1e6 !units of time_step to seconds
    call camp_mpi_barrier()
  end subroutine

end program mock_monarch_t
