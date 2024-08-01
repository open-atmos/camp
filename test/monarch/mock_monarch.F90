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
  character(len=:), allocatable :: DIFF_CELLS
  real,allocatable :: temperature(:, :, :)
  real, allocatable  :: species_conc(:, :, :, :)
  real, allocatable  :: water_conc(:, :, :, :) !(kg_H2O/kg_air)
  real, allocatable  :: air_density(:, :, :) !(kg_air/m^3)
  real, allocatable  :: pressure(:, :, :) !(Pa)
  real, allocatable  :: conv(:, :, :) !(mol s-1 m-2 to ppmv)
  integer :: i_hour = 0
  real :: curr_time = START_TIME
  type(camp_monarch_interface_t), pointer :: camp_interface
  character(len=:), allocatable :: camp_input_file, chemFile,&
    diffCells,caseGpuCpu
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
  integer :: id, n_cells_monarch, load_gpu

  call camp_mpi_init()
  I_W=1
  I_E=1
  I_S=1
  I_N=1
  DIFF_CELLS = "OFF"
  call jfile%initialize()
  export_path = "settings/TestMonarch"//".json"
  call jfile%load_file(export_path); if (jfile%failed()) print*,&
          "JSON not found at ",export_path
  call jfile%get('chemFile',output_file_title)
  camp_input_file = "settings/"//output_file_title//"/config.json"
  output_path = "out/"//output_file_title
  call jfile%get('nCells',NUM_VERT_CELLS)
  n_cells_monarch = (I_E - I_W+1)*(I_N - I_S+1)*NUM_VERT_CELLS
  load_gpu=0
  open(unit=32, file='settings/config_variables_c_solver.txt', status='old')
  read(32, *) load_gpu
  close(32)
  if(load_gpu == 0) then
    n_cells = 1
  else
    n_cells = n_cells_monarch
  end if
  call jfile%get('timeSteps',NUM_TIME_STEP)
  call jfile%get('timeStepsDt',TIME_STEP)
  call jfile%get('diffCells',diffCells)
  if(diffCells=="Realistic") then
    DIFF_CELLS = "ON"
  end if
  NUM_WE_CELLS = I_E-I_W+1
  NUM_SN_CELLS = I_N-I_S+1
  call jfile%get('caseGpuCpu',caseGpuCpu)
  if (camp_mpi_rank()==0) then
    write(*,*) "Time-steps:", NUM_TIME_STEP, "Cells:",&
        NUM_WE_CELLS*NUM_SN_CELLS*NUM_VERT_CELLS, &
            diffCells,caseGpuCpu, "MPI processes",camp_mpi_size()

  end if
  allocate(temperature(NUM_WE_CELLS,NUM_SN_CELLS,NUM_VERT_CELLS))
  allocate(species_conc(NUM_WE_CELLS,NUM_SN_CELLS,NUM_VERT_CELLS,NUM_MONARCH_SPEC))
  allocate(water_conc(NUM_WE_CELLS,NUM_SN_CELLS,NUM_VERT_CELLS,WATER_VAPOR_ID))
  allocate(air_density(NUM_WE_CELLS,NUM_SN_CELLS,NUM_VERT_CELLS))
  allocate(pressure(NUM_WE_CELLS,NUM_SN_CELLS,NUM_VERT_CELLS))
  allocate(conv(NUM_WE_CELLS, NUM_SN_CELLS, NUM_VERT_CELLS))

  camp_interface => camp_monarch_interface_t(camp_input_file, output_file_title, &
          START_CAMP_ID, END_CAMP_ID, n_cells, load_gpu)

  camp_interface%camp_state%state_var(:) = 0.0
  species_conc(:,:,:,:) = 0.0
  air_density(:,:,:) = 1.225
  water_conc(:,:,:,WATER_VAPOR_ID) = 0.

  call camp_interface%get_init_conc(species_conc, water_conc, WATER_VAPOR_ID, &
          i_W,I_E,I_S,I_N)

  if(output_file_title=="monarch_cb05") then
    call import_camp_input_json(camp_interface)
  end if

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
    if(DIFF_CELLS=="ON") then
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
    else
      if(output_file_title=="cb05_paperV2") then
        temperature(:,:,:) = temp_init
        pressure(:,:,:) = press_init
      end if
      do t=1,12 !12 first hours
        camp_interface%rate_emi(t,:)=1.0
      end do
    end if
    if(output_file_title=="cb05_paperV2") then
      air_density(:,:,:) = pressure(:,:,:)/(287.04*temperature(:,:,:)* &
              (1.+0.60813824*water_conc(:,:,:,WATER_VAPOR_ID))) !kg m-3
      conv(:,:,:)=0.02897/air_density(:,:,:)*(TIME_STEP*60.)*1e6 !units of time_step to seconds

    end if
    call camp_mpi_barrier()
  end subroutine

  subroutine import_camp_input_json(camp_interface)
    type(camp_monarch_interface_t), intent(inout) :: camp_interface
    integer :: z,i,j,k,r,o,i_cell,i_spec,i_photo_rxn
    integer :: state_size_per_cell
    type(json_file) :: jfile
    type(json_core) :: json
    character(len=:), allocatable :: export_path, spec_name_json
    real(kind=dp) :: dt, temp, press, real_val
    type(string_t), allocatable :: camp_spec_names(:), unique_names(:)
    real, dimension(NUM_EBI_PHOTO_RXN) :: ebi_photo_rates
    character(len=128) :: mpi_rank_str, i_str
    integer :: mpi_rank, id
    character, allocatable :: buffer(:)
    integer :: max_spec_name_size=512
    integer(kind=i_kind) :: pos, pack_size, size_state_per_cell
    character(len=:), allocatable :: spec_name
    real(kind=dp) :: base_rate

    state_size_per_cell = camp_interface%camp_core%size_state_per_cell
    call jfile%initialize()
    export_path = "settings/monarch_cb05/monarch_cell_init_concs.json"
    call jfile%load_file(export_path); if (jfile%failed()) print*,&
            "JSON not found at ",export_path
    size_state_per_cell = camp_interface%camp_core%size_state_per_cell
    mpi_rank = camp_mpi_rank()
    if (mpi_rank==0) then
    unique_names=camp_interface%camp_core%unique_names()
    pack_size = 0
    do z=1, size_state_per_cell
      pack_size = pack_size +  camp_mpi_pack_size_string(trim(unique_names(z)%string))
    end do
    allocate(buffer(pack_size))
    pos = 0
    do z=1, size(unique_names)
      call camp_mpi_pack_string(buffer, pos, trim(unique_names(z)%string))
    end do
    end if
    call camp_mpi_bcast_integer(pack_size, MPI_COMM_WORLD)
    if (mpi_rank/=0) then
      allocate(buffer(pack_size))
    end if
    call camp_mpi_bcast_packed(buffer, MPI_COMM_WORLD)
    if (mpi_rank/=0) then
      pos = 0
      allocate(unique_names(size_state_per_cell))
      spec_name=""
      do z=1,max_spec_name_size
        spec_name=spec_name//" "
      end do
      do z=1, size_state_per_cell
        call camp_mpi_unpack_string(buffer, pos, spec_name)
        unique_names(z)%string= trim(spec_name)
      end do
    end if
    deallocate(buffer)
    camp_spec_names=unique_names
    do i=1, size(camp_spec_names)
      call jfile%get('input.species.'//camp_spec_names(i)%string,&
              camp_interface%camp_state%state_var(i))
    end do
    do z=0,n_cells-1
      do i=1,state_size_per_cell
        camp_interface%camp_state%state_var(i+(z*state_size_per_cell))=&
                camp_interface%camp_state%state_var(i)
      end do
    end do
    do i=I_W,I_E
      do j=I_S,I_N
        do k=1,NUM_VERT_CELLS
          o = (j-1)*(I_E) + (i-1)
          z = (k-1)*(I_E*I_N) + o
          species_conc(i,j,k,camp_interface%map_monarch_id(:)) = &
                  camp_interface%camp_state%state_var(camp_interface%map_camp_id(:))
          call jfile%get('input.temperature',temp)
          temperature(i,j,k)=temp
          call jfile%get('input.pressure',press)
          pressure(i,j,k)=press
        end do
      end do
    end do
    do i = 1, state_size_per_cell
      if (trim(camp_spec_names(i)%string)=="H2O") then
        water_conc(:,:,:,WATER_VAPOR_ID) = camp_interface%camp_state%state_var(i)
      end if
    end do
    do i=1, camp_interface%n_photo_rxn
      write(i_str,*) i
      i_str=adjustl(i_str)
      call jfile%get('input.photo_rates.'//trim(i_str),&
              camp_interface%base_rates(i))
    end do
    do z =1, n_cells
      do i = 1, camp_interface%n_photo_rxn
        base_rate = camp_interface%base_rates(i)
        call camp_interface%photo_rxns(i)%set_rate(base_rate)
        call camp_interface%camp_core%update_data(camp_interface%photo_rxns(i),z)
      end do
    end do
    call jfile%destroy()
  end subroutine import_camp_input_json

end program mock_monarch_t
