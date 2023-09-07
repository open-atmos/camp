! Copyright (C) 2021 Barcelona Supercomputing Center and University of
! Illinois at Urbana-Champaign
! SPDX-License-Identifier: MIT


!> Mock version of the MONARCH model for testing integration with CAMP
program mock_monarch_t

  use camp_constants,                    only: const
  use camp_util,                          only : assert_msg, almost_equal, &
                                                to_string
  use camp_monarch_interface_2
  use camp_mpi
  use camp_solver_stats
  use json_module

#ifdef SOLVE_EBI_IMPORT_CAMP_INPUT
  ! EBI Solver
  use module_bsc_chem_data
  use EXT_HRDATA
  use EXT_RXCM,                               only : NRXNS, RXLABEL
  ! KPP Solver
  use cb05cl_ae5_Initialize,                  only : KPP_Initialize => Initialize
  use cb05cl_ae5_Model,                       only : KPP_NSPEC => NSPEC, &
          KPP_STEPMIN => STEPMIN, &
          KPP_STEPMAX => STEPMAX, &
          KPP_RTOL => RTOL, &
          KPP_ATOL => ATOL, &
          KPP_TIME => TIME, &
          KPP_C => C, &
          KPP_RCONST => RCONST, &
          KPP_Update_RCONST => Update_RCONST, &
          KPP_INTEGRATE => INTEGRATE, &
          KPP_SPC_NAMES => SPC_NAMES, &
          KPP_PHOTO_RATES => PHOTO_RATES, &
          KPP_TEMP => TEMP, &
          KPP_PRESS => PRESS, &
          KPP_SUN => SUN, &
          KPP_M => M, &
          KPP_N2 => N2, &
          KPP_O2 => O2, &
          KPP_H2 => H2, &
          KPP_H2O => H2O, &
          KPP_N2O => N2O, &
          KPP_CH4 => CH4, &
          KPP_NVAR => NVAR, &
          KPP_NREACT => NREACT, &
          KPP_DT => DT
  use cb05cl_ae5_Parameters,                  only : KPP_IND_O2 => IND_O2
  use cb05cl_ae5_Initialize, ONLY: Initialize

#endif

  implicit none

  integer(kind=i_kind), parameter :: NUM_EBI_SPEC = 72
  integer(kind=i_kind), parameter :: NUM_EBI_PHOTO_RXN = 23

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Parameters for mock MONARCH model !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  integer, parameter :: NUM_MONARCH_SPEC = 250 !800
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
  real, parameter :: TIME_STEP_MONARCH37= 1.6
  integer :: NUM_TIME_STEP
  integer, parameter :: WATER_VAPOR_ID = 5
  real, parameter :: START_TIME = 0
  integer :: n_cells = 1
  character(len=:), allocatable :: ADD_EMISIONS
  character(len=:), allocatable :: DIFF_CELLS

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! State variables for mock MONARCH model !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  real,allocatable :: temperature(:, :, :)
  real, allocatable  :: species_conc(:, :, :, :)
  real, allocatable  :: water_conc(:, :, :, :) !(kg_H2O/kg_air)
  real, allocatable  :: air_density(:, :, :) !(kg_air/m^3)
  real, allocatable  :: pressure(:, :, :) !(Pa)
  real, allocatable  :: conv(:, :, :) !(mol s-1 m-2 to ppmv)
  integer :: i_hour = 0
  real :: curr_time = START_TIME
  type(camp_monarch_interface_t), pointer :: camp_interface

  ! Mock model setup and evaluation variables !
  character(len=:), allocatable :: camp_input_file, chemFile,&
  caseMulticellsOnecell, diffCells,caseGpuCpu
  character(len=:), allocatable :: output_path, output_file_title, str_to_int_aux
  character(len=:), allocatable :: str

  ! MPI
  character, allocatable :: buffer(:)
  integer(kind=i_kind) :: pos, pack_size, mpi_threads, ierr

  character(len=512) :: arg
  integer :: i_time, i_spec, i_case, i, j, k, z,r
  integer :: plot_case, new_v_cells, aux_int
  type(solver_stats_t), target :: solver_stats

  integer :: n_cells_tstep
  integer :: plot_species = 0
  type(json_file) :: jfile
  type(json_core) :: json
  character(len=:), allocatable :: export_path
  character(len=128) :: i_str
  integer :: id

  !netcdf
  integer :: ncid, pres_varid, temp_varid
  integer(kind=i_kind), allocatable :: species_id_netcdf(:)
  integer :: counter_export_netcdf
  character(len=:), allocatable :: file_name

  ! initialize mpi (to take the place of a similar MONARCH call)
  !todo put MPI as an option to reduce time execution when developing, testing, and debugging
  call camp_mpi_init()

  I_W=1
  I_E=1
  I_S=1
  I_N=1
  ADD_EMISIONS = "OFF"
  DIFF_CELLS = "OFF"
  call jfile%initialize()
  export_path = "settings/TestMonarch"//".json"
  call jfile%load_file(export_path); if (jfile%failed()) print*,&
          "JSON not found at ",export_path
  call jfile%get('_chemFile',output_file_title)
  camp_input_file = "settings/"//output_file_title//"/config.json"
  output_path = "out/"//output_file_title
  if(output_file_title.eq."monarch_binned") then
    ADD_EMISIONS = "monarch_binned"
  end if
  call jfile%get('nCells',NUM_VERT_CELLS)
  call jfile%get('caseMulticellsOnecell',caseMulticellsOnecell)
  output_path = output_path//"_"//caseMulticellsOnecell
  if(caseMulticellsOnecell.eq."One-cell") then
    n_cells = 1
  else
    n_cells = (I_E - I_W+1)*(I_N - I_S+1)*NUM_VERT_CELLS
  end if
  call jfile%get('timeSteps',NUM_TIME_STEP)
  call jfile%get('timeStepsDt',TIME_STEP)
  call jfile%get('diffCells',diffCells)
  if(diffCells.eq."Realistic") then
    DIFF_CELLS = "ON"
  else
    DIFF_CELLS = "OFF"
  end if
  NUM_WE_CELLS = I_E-I_W+1
  NUM_SN_CELLS = I_N-I_S+1
  n_cells_tstep = NUM_WE_CELLS*NUM_SN_CELLS*NUM_VERT_CELLS
  call jfile%get('caseGpuCpu',caseGpuCpu)
  if (camp_mpi_rank().eq.0) then
    write(*,*) "Time-steps:", NUM_TIME_STEP, "Cells:",&
            n_cells_tstep, &
            diffCells,  caseMulticellsOnecell,caseGpuCpu, "MPI processes",camp_mpi_size()

  end if
  allocate(temperature(NUM_WE_CELLS,NUM_SN_CELLS,NUM_VERT_CELLS))
  allocate(species_conc(NUM_WE_CELLS,NUM_SN_CELLS,NUM_VERT_CELLS,NUM_MONARCH_SPEC))
  allocate(water_conc(NUM_WE_CELLS,NUM_SN_CELLS,NUM_VERT_CELLS,WATER_VAPOR_ID))
  allocate(air_density(NUM_WE_CELLS,NUM_SN_CELLS,NUM_VERT_CELLS))
  allocate(pressure(NUM_WE_CELLS,NUM_SN_CELLS,NUM_VERT_CELLS))
  allocate(conv(NUM_WE_CELLS, NUM_SN_CELLS, NUM_VERT_CELLS))

  camp_interface => camp_monarch_interface_t(camp_input_file, output_file_title, &
          START_CAMP_ID, END_CAMP_ID, n_cells, n_cells_tstep, ADD_EMISIONS)

  camp_interface%camp_state%state_var(:) = 0.0
  species_conc(:,:,:,:) = 0.0
  air_density(:,:,:) = 1.225
  water_conc(:,:,:,WATER_VAPOR_ID) = 0.

  !print*,"mock_monarch water_conc end"

  if(.not.output_file_title.eq."cb05_yarwood2005")  then
    call camp_interface%get_init_conc(species_conc, water_conc, WATER_VAPOR_ID, &
            i_W,I_E,I_S,I_N)
  end if

  !print*,"mock_monarch get_init_conc end"

  if(output_file_title.eq."monarch_cb05") then
    call import_camp_input_json(camp_interface)
  end if

  call set_env(camp_interface,output_path)

#ifdef SOLVE_EBI_IMPORT_CAMP_INPUT
  if(caseMulticellsOnecell.eq."EBI") then
    call solve_ebi(camp_interface)
  end if
#endif

  if(3*NUM_TIME_STEP.gt.(60*24)) then !24h limit time-step
    print*,"ERROR 3*NUM_TIME_STEP.gt.(60*24): Reduce number of time-step or time-step size"
    STOP
  end if

  if(.not.caseMulticellsOnecell.eq."EBI") then
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
         solver_stats,DIFF_CELLS)
    curr_time = curr_time + TIME_STEP
#ifdef CAMP_DEBUG_GPU
      call camp_interface%camp_core%export_solver_stats(solver_stats=solver_stats)
      call camp_interface%camp_core%reset_solver_stats(solver_stats=solver_stats)
#endif
    end do
  end if

  if (camp_mpi_rank().eq.0) then
    write(*,*) "Model run time: ", comp_time, " s"
  end if

  call camp_mpi_barrier()
  deallocate(camp_input_file)
  deallocate(output_path)
  deallocate(output_file_title)
  mpi_threads = camp_mpi_size()
  if ((mpi_threads.gt.1)) then
  else
    deallocate(camp_interface)
  end if
  call camp_mpi_finalize()

contains

  subroutine set_env(camp_interface,file_prefix)
    type(camp_monarch_interface_t), intent(inout) :: camp_interface
    character(len=:), allocatable, intent(in) :: file_prefix
    character(len=:), allocatable :: file_name
    integer :: z,o,i,j,k,r,i_cell,i_spec,mpi_size,ncells,tid,ncells_mpi
    integer :: n_cells_print
    real :: temp_init,press,press_init,press_end,press_range,press_slide

    temp_init = 290.016
    press_init = 100000. !Should be equal to camp_monarch_interface
    if(DIFF_CELLS.eq."ON") then
      ncells=(I_E - I_W+1)*(I_N - I_S+1)*NUM_VERT_CELLS
      mpi_size=camp_mpi_size()
      tid=camp_mpi_rank()
      ncells_mpi=ncells*mpi_size
      press_end = 10000.
      press_range = press_end-press_init
      if((ncells_mpi).eq.1) then
        press_slide = 0.
      else
        press_slide = press_range/(ncells_mpi-1)
      end if
      do i=I_W,I_E
        do j=I_S,I_N
          do k=1,NUM_VERT_CELLS
            o = (j-1)*(I_E) + (i-1)
            z = (k-1)*(I_E*I_N) + o
            z = tid*ncells+z
            pressure(i,j,k)=press_init+press_slide*z
            temperature(i,j,k)=temp_init*((pressure(i,j,k)/press_init)**(287./1004.)) !dry_adiabatic formula
          end do
        end do
      end do
    else
      if(ADD_EMISIONS.eq."monarch_binned") then
        temperature(:,:,:) = temp_init
        pressure(:,:,:) = press_init
      end if
    end if
    if(ADD_EMISIONS.eq."monarch_binned") then
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
    integer, dimension(NUM_EBI_PHOTO_RXN) :: photo_id_camp
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
    if (mpi_rank.eq.0) then
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
    if (mpi_rank.ne.0) then
      allocate(buffer(pack_size))
    end if
    call camp_mpi_bcast_packed(buffer, MPI_COMM_WORLD)
    if (mpi_rank.ne.0) then
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
      if (trim(camp_spec_names(i)%string).eq."H2O") then
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
        call camp_interface%photo_rxns(i)%set_rate(base_rate) !not used if exported cb05
        call camp_interface%camp_core%update_data(camp_interface%photo_rxns(i),z)
      end do
    end do
    call jfile%destroy()
  end subroutine import_camp_input_json

#ifdef SOLVE_EBI_IMPORT_CAMP_INPUT

  subroutine solve_ebi(camp_interface)
    type(camp_monarch_interface_t), intent(inout) :: camp_interface
    integer :: z,i,j,k,r,o,i_cell,i_spec,i_photo_rxn,i_time
    real(kind=dp) :: dt, temp, press_json, auxr
    real :: press
    type(json_core) :: json
    type(json_value),pointer :: p, species_in, species_out, input, output&
            , photo_rates
    character(len=:), allocatable :: export_path, spec_name
    integer :: mpi_rank
    character(len=128) :: mpi_rank_str, i_str
    type(string_t), dimension(NUM_EBI_SPEC) :: ebi_spec_names
    type(string_t), dimension(NUM_EBI_SPEC) :: monarch_spec_names
    type(string_t), allocatable :: camp_spec_names(:), monarch_species_names(:)
    integer, dimension(NUM_EBI_SPEC) :: map_ebi_monarch
    logical :: is_sunny
    real, dimension(NUM_EBI_PHOTO_RXN) :: ebi_photo_rates
    integer, dimension(NUM_EBI_PHOTO_RXN) :: photo_id_camp
    real(kind=dp) :: rel_error_in_out
    real(kind=dp), allocatable :: ebi_init(:)
    real(kind=dp) :: mwair = 28.9628 !mean molecular weight for dry air [ g/mol ]
    real(kind=dp) :: mwwat = 18.0153 ! mean molecular weight for water vapor [ g/mol ]
    real(kind=dp) :: comp_start, comp_end, ebi_time, ebi_start

    ! Set the BSC chem parameters
    call init_bsc_chem_data()
    ! Set the output unit
    LOGDEV = 6
    ! Set the aerosol flag
    L_AE_VRSN = .false.
    ! Set the aq. chem flag
    L_AQ_VRSN = .false.
    ! Initialize the solver
    call EXT_HRINIT
    RKI(:) = 0.0
    RXRAT(:) = 0.0
    YC(:) = 0.0
    YC0(:) = 0.0
    YCP(:) = 0.0
    PROD(:) = 0.0
    LOSS(:) = 0.0
    PNEG(:) = 0.0
    EBI_TMSTEP = TIME_STEP
    N_EBI_STEPS = 1
    N_INR_STEPS = 1
    is_sunny = .true.
    ebi_time = 0.0
    allocate(ebi_init(size(YC)))
    call set_ebi_species(ebi_spec_names)
    !call set_monarch_species(monarch_spec_names)
    monarch_species_names=camp_interface%monarch_species_names
    call set_ebi_photo_ids_with_camp(photo_id_camp)
    do i=1, NUM_EBI_PHOTO_RXN
      ebi_photo_rates(i)=&
              camp_interface%base_rates(photo_id_camp(i))*60
      !print*,i,ebi_photo_rates(i)
    end do
    do i = 1, NUM_EBI_SPEC
      do z = 1, size(monarch_species_names)
        if (trim(ebi_spec_names(i)%string).eq.trim(monarch_species_names(z)%string)) then
          !YC(i) = camp_interface%camp_state%state_var(z)
          !map_ebi_monarch(z) = i
          map_ebi_monarch(camp_interface%map_monarch_id(z)) = i
          !print*,YC(map_ebi_monarch(z)),YC(i)
          !debug
          ebi_init(i) = YC(i)
          !print*,ebi_spec_names(i)%string, camp_interface%camp_state%state_var(j)
        end if
      end do
    end do
    !print*, water_conc(1,1,1,WATER_VAPOR_ID)
    do i_time=1, NUM_TIME_STEP
      ebi_time = 0.
      do i=I_W,I_E
      do j=I_S,I_N
      do k=1,NUM_VERT_CELLS
      do z = 1, size(monarch_species_names)
        YC(map_ebi_monarch(z)) = species_conc(i,j,k,z)
        !print*,ebi_spec_names(i)%string, camp_interface%camp_state%state_var(j)
      end do
      press=pressure(i,j,k)/const%air_std_press
      !print*,"EBI press, pressure(1,1,1), const%air_std_press"&
      !,press,pressure(1,1,1),const%air_std_press
      !print*,temperature
      !print*,pressure
      call cpu_time(comp_start)
      ebi_start = MPI_Wtime()
      call EXT_HRCALCKS( NUM_EBI_PHOTO_RXN,       & ! Number of EBI solver photolysis reactions
              is_sunny,                & ! Flag for sunlight
              ebi_photo_rates,             & ! Photolysis rates
              temperature(i,j,k),             & ! Temperature (K)
              press,                & ! Air pressure (atm)
              water_conc(i,j,k,WATER_VAPOR_ID),&! * mwair / mwwat * 1.e6, &
              !water_conc(1,1,1,WATER_VAPOR_ID) ,              & ! Water vapor concentration (ppmV)
              RKI)                       ! Rate constants
      call EXT_HRSOLVER( 2018012, 070000, 1, 1, 1) ! These dummy variables are just for output
      call cpu_time(comp_end)
      comp_time = comp_time + (comp_end-comp_start)
      ebi_time = ebi_time + MPI_Wtime()-ebi_start
      !H2O  = MAX(WATER(C,R,kflip,P_QV) * MAOMV * 1.0e+06,0.0)
      !print*,YC(:)
      do z = 1, size(monarch_species_names)
       species_conc(i,j,k,z) = YC(map_ebi_monarch(z))
        !print*,ebi_spec_names(i)%string, camp_interface%camp_state%state_var(j)
      end do
      end do
      end do
      end do
      if(export_results_all_cells.eq.1) then
        call export_file_results_all_cells(camp_interface)
      end if
    end do
#ifdef PRINT_EBI_INPUT
    print*,"EBI species"
    print*, "TIME_STEP", TIME_STEP
    print*, "Temp", temperature(1,1,1)
    print*, "Press", pressure(1,1,1)
    print*,"water_conc",water_conc(1,1,1,WATER_VAPOR_ID)
    print*,"ebi_photo_rates:", ebi_photo_rates(:)
    print*,"species_conc:", species_conc
    print*,"ebi_init:", ebi_init
#endif
#ifdef PRINT_EBI_SPEC_BY_EBI_ORDER
    print*,"EBI order:"
    print*,"Name, id, init, out, rel. error [(init-out)/(init+out)]"
    do j = 1, NUM_EBI_SPEC
      i = j
      rel_error_in_out=abs((ebi_init(i)-YC(i))/&
              (ebi_init(i)+YC(i)+1.0d-30))
      print*,ebi_spec_names(i)%string, i, ebi_init(i)&
              ,YC(i)!, rel_error_in_out
    end do
#endif
  end subroutine

  subroutine solve_kpp(camp_interface)
    type(camp_monarch_interface_t), intent(inout) :: camp_interface
    integer :: z,i,j,k,r,o,i_cell,i_spec,i_photo_rxn,i_time
    ! Set the step limits
    KPP_STEPMIN = 0.0d0
    KPP_STEPMAX = 0.0d0
    KPP_SUN = 1.0
    ! Set the tolerances
    do i_spec = 1, KPP_NVAR
      KPP_RTOL(i_spec) = 1.0d-4
      KPP_ATOL(i_spec) = 1.0d-3
    end do
    CALL KPP_Initialize()
    KPP_PHOTO_RATES(:) = camp_interface%base_rates(:)
  end subroutine

#endif


  subroutine set_ebi_species(spec_names)
    !> EBI solver species names
    type(string_t), dimension(NUM_EBI_SPEC) :: spec_names

    spec_names(1)%string = "NO2"
    spec_names(2)%string = "NO"
    spec_names(3)%string = "O"
    spec_names(4)%string = "O3"
    spec_names(5)%string = "NO3"
    spec_names(6)%string = "O1D"
    spec_names(7)%string = "OH"
    spec_names(8)%string = "HO2"
    spec_names(9)%string = "N2O5"
    spec_names(10)%string = "HNO3"
    spec_names(11)%string = "HONO"
    spec_names(12)%string = "PNA"
    spec_names(13)%string = "H2O2"
    spec_names(14)%string = "XO2"
    spec_names(15)%string = "XO2N"
    spec_names(16)%string = "NTR"
    spec_names(17)%string = "ROOH"
    spec_names(18)%string = "FORM"
    spec_names(19)%string = "ALD2"
    spec_names(20)%string = "ALDX"
    spec_names(21)%string = "PAR"
    spec_names(22)%string = "CO"
    spec_names(23)%string = "MEO2"
    spec_names(24)%string = "MEPX"
    spec_names(25)%string = "MEOH"
    spec_names(26)%string = "HCO3"
    spec_names(27)%string = "FACD"
    spec_names(28)%string = "C2O3"
    spec_names(29)%string = "PAN"
    spec_names(30)%string = "PACD"
    spec_names(31)%string = "AACD"
    spec_names(32)%string = "CXO3"
    spec_names(33)%string = "PANX"
    spec_names(34)%string = "ROR"
    spec_names(35)%string = "OLE"
    spec_names(36)%string = "ETH"
    spec_names(37)%string = "IOLE"
    spec_names(38)%string = "TOL"
    spec_names(39)%string = "CRES"
    spec_names(40)%string = "TO2"
    spec_names(41)%string = "TOLRO2"
    spec_names(42)%string = "OPEN"
    spec_names(43)%string = "CRO"
    spec_names(44)%string = "MGLY"
    spec_names(45)%string = "XYL"
    spec_names(46)%string = "XYLRO2"
    spec_names(47)%string = "ISOP"
    spec_names(48)%string = "ISPD"
    spec_names(49)%string = "ISOPRXN"
    spec_names(50)%string = "TERP"
    spec_names(51)%string = "TRPRXN"
    spec_names(52)%string = "SO2"
    spec_names(53)%string = "SULF"
    spec_names(54)%string = "SULRXN"
    spec_names(55)%string = "ETOH"
    spec_names(56)%string = "ETHA"
    spec_names(57)%string = "CL2"
    spec_names(58)%string = "CL"
    spec_names(59)%string = "HOCL"
    spec_names(60)%string = "CLO"
    spec_names(61)%string = "FMCL"
    spec_names(62)%string = "HCL"
    spec_names(63)%string = "TOLNRXN"
    spec_names(64)%string = "TOLHRXN"
    spec_names(65)%string = "XYLNRXN"
    spec_names(66)%string = "XYLHRXN"
    spec_names(67)%string = "BENZENE"
    spec_names(68)%string = "BENZRO2"
    spec_names(69)%string = "BNZNRXN"
    spec_names(70)%string = "BNZHRXN"
    spec_names(71)%string = "SESQ"
    spec_names(72)%string = "SESQRXN"

  end subroutine set_ebi_species

  subroutine set_monarch_species(spec_names)
    !> EBI solver species names
    type(string_t), dimension(NUM_EBI_SPEC) :: spec_names
    !Monarch order
    spec_names(1)%string = "NO2"
    spec_names(2)%string = "NO"
    spec_names(3)%string = "O3"
    spec_names(4)%string = "NO3"
    spec_names(5)%string = "N2O5"
    spec_names(6)%string = "HNO3"
    spec_names(7)%string = "HONO"
    spec_names(8)%string = "PNA"
    spec_names(9)%string = "H2O2"
    spec_names(10)%string = "NTR"
    spec_names(11)%string = "ROOH"
    spec_names(12)%string = "FORM"
    spec_names(13)%string = "ALD2"
    spec_names(14)%string = "ALDX"
    spec_names(15)%string = "PAR"
    spec_names(16)%string = "CO"
    spec_names(17)%string = "MEPX"
    spec_names(18)%string = "MEOH"
    spec_names(19)%string = "FACD"
    spec_names(20)%string = "PAN"
    spec_names(21)%string = "PACD"
    spec_names(22)%string = "AACD"
    spec_names(23)%string = "PANX"
    spec_names(24)%string = "OLE"
    spec_names(25)%string = "ETH"
    spec_names(26)%string = "IOLE"
    spec_names(27)%string = "TOL"
    spec_names(28)%string = "CRES"
    spec_names(29)%string = "OPEN"
    spec_names(30)%string = "MGLY"
    spec_names(31)%string = "XYL"
    spec_names(32)%string = "ISOP"
    spec_names(33)%string = "ISPD"
    spec_names(34)%string = "TERP"
    spec_names(35)%string = "SO2"
    spec_names(36)%string = "SULF"
    spec_names(37)%string = "ETOH"
    spec_names(38)%string = "ETHA"
    spec_names(39)%string = "CL2"
    spec_names(40)%string = "HOCL"
    spec_names(41)%string = "FMCL"
    spec_names(42)%string = "HCL"
    spec_names(43)%string = "BENZENE"
    spec_names(44)%string = "SESQ"
    spec_names(45)%string = "O"
    spec_names(46)%string = "O1D"
    spec_names(47)%string = "OH"
    spec_names(48)%string = "HO2"
    spec_names(49)%string = "XO2"
    spec_names(50)%string = "XO2N"
    spec_names(51)%string = "MEO2"
    spec_names(52)%string = "HCO3"
    spec_names(53)%string = "C2O3"
    spec_names(54)%string = "CXO3"
    spec_names(55)%string = "ROR"
    spec_names(56)%string = "TO2"
    spec_names(57)%string = "TOLRO2"
    spec_names(58)%string = "CRO"
    spec_names(59)%string = "XYLRO2"
    spec_names(60)%string = "ISOPRXN"
    spec_names(61)%string = "TRPRXN"
    spec_names(62)%string = "SULRXN"
    spec_names(63)%string = "CL"
    spec_names(64)%string = "CLO"
    spec_names(65)%string = "TOLNRXN"
    spec_names(66)%string = "TOLHRXN"
    spec_names(67)%string = "XYLNRXN"
    spec_names(68)%string = "XYLHRXN"
    spec_names(69)%string = "BENZRO2"
    spec_names(70)%string = "BNZNRXN"
    spec_names(71)%string = "BNZHRXN"
    spec_names(72)%string = "SESQRXN"

  end subroutine set_monarch_species

  subroutine set_ebi_photo_ids_with_camp(photo_id_camp)
    !> EBI solver species names
    integer, dimension(NUM_EBI_PHOTO_RXN) :: photo_id_camp
    !Monarch order
    photo_id_camp(1) = 1
    photo_id_camp(2) = 2
    photo_id_camp(3) = 3
    photo_id_camp(4) = 4
    photo_id_camp(5) = 5
    photo_id_camp(6) = 6
    photo_id_camp(7) = 7
    photo_id_camp(8) = 8
    photo_id_camp(9) = 9
    photo_id_camp(10) = 10
    photo_id_camp(11) = 11
    photo_id_camp(12) = 12
    photo_id_camp(13) = 14
    photo_id_camp(14) = 15
    photo_id_camp(15) = 16
    photo_id_camp(16) = 17
    photo_id_camp(17) = 18
    photo_id_camp(18) = 19
    photo_id_camp(19) = 22
    photo_id_camp(20) = 23
    photo_id_camp(21) = 25 !0.0
    photo_id_camp(22) = 25 !0.0
    photo_id_camp(23) = 25 !0.0s

  end subroutine set_ebi_photo_ids_with_camp

end program mock_monarch_t
