! Copyright (C) 2021 Barcelona Supercomputing Center and University of
! Illinois at Urbana-Champaign
! SPDX-License-Identifier: MIT

!> \file
!> The camp_monarch_interface_t object and related functions

!> Interface for the MONACH model and CAMP-camp
module camp_monarch_interface_2

  use camp_constants, only : i_kind
  use camp_mpi
  use camp_util, only : assert_msg, string_t,warn_assert_msg
  use camp_camp_core
  use camp_camp_state
  use camp_aero_rep_data
  use camp_aero_rep_factory
  use camp_aero_rep_modal_binned_mass
  use camp_chem_spec_data
  use camp_property
  use camp_camp_solver_data
  use camp_mechanism_data, only : mechanism_data_t
  use camp_rxn_data, only : rxn_data_t
  use camp_rxn_photolysis
  use camp_solver_stats
  use mpi
  use json_module

  implicit none
  private
  public :: camp_monarch_interface_t
  type :: camp_monarch_interface_t
    type(camp_core_t), pointer :: camp_core
    type(camp_state_t), pointer :: camp_state
    type(string_t), allocatable :: monarch_species_names(:)
    character(len=:), allocatable :: output_file_title
    integer(kind=i_kind), allocatable :: map_monarch_id(:), map_camp_id(:)
    integer(kind=i_kind), allocatable :: init_conc_camp_id(:),specs_emi_id(:)
    real(kind=dp), allocatable :: init_conc(:)
    integer(kind=i_kind) :: n_cells = 1
    integer(kind=i_kind) :: tracer_starting_id
    integer(kind=i_kind) :: tracer_ending_id
    type(property_t), pointer :: species_map_data
    integer(kind=i_kind) :: gas_phase_water_id
    type(property_t), pointer :: init_conc_data
    type(property_t), pointer :: property_set
    type(rxn_update_data_photolysis_t), allocatable :: photo_rxns(:)
    real(kind=dp), allocatable :: base_rates(:),specs_emi(:),offset_photo_rates_cells(:)
    integer :: n_photo_rxn
    integer :: nrates_cells
    logical :: solve_multiple_cells = .false.
    type(string_t), allocatable :: kpp_rxn_labels(:)
    real(kind=dp) :: KPP_RSTATE(20)
    integer :: KPP_ICNTRL(20)
  contains
    procedure :: integrate
    procedure :: get_init_conc
    procedure, private :: load
    procedure, private :: create_map
    procedure, private :: load_init_conc
    final :: finalize
  end type camp_monarch_interface_t

  interface camp_monarch_interface_t
    procedure :: constructor
  end interface camp_monarch_interface_t

  integer(kind=i_kind) :: MONARCH_PROCESS
  real(kind=dp), public, save :: comp_time = 0.0d0
  real(kind=dp), parameter :: mwair = 28.9628 !mean molecular weight for dry air [ g/mol ]
  real(kind=dp), parameter :: mwwat = 18.0153 ! mean molecular weight for water vapor [ g/mol ]
contains


  function constructor(camp_config_file, output_file_title, &
   starting_id, ending_id, n_cells, n_cells_tstep, mpi_comm) result (this)
    type(camp_monarch_interface_t), pointer :: this
    character(len=:), allocatable, optional :: camp_config_file
    character(len=*), intent(in):: output_file_title
    integer, optional :: starting_id
    integer, optional :: ending_id
    integer, intent(in), optional :: mpi_comm
    integer, optional :: n_cells
    integer, optional :: n_cells_tstep
    type(camp_solver_data_t), pointer :: camp_solver_data
    character, allocatable :: buffer(:)
    integer(kind=i_kind) :: pos, pack_size
    integer(kind=i_kind) :: i_spec, i_photo_rxn, rank, n_ranks, ierr
    type(string_t), allocatable :: unique_names(:)
    character(len=:), allocatable :: spec_name, settings_interface_file
    real(kind=dp) :: base_rate
    class(aero_rep_data_t), pointer :: aero_rep
    integer(kind=i_kind) :: i_sect_om, i_sect_bc, i_sect_sulf, i_sect_opm, i, z
    type(aero_rep_factory_t) :: aero_rep_factory
    type(aero_rep_update_data_modal_binned_mass_GMD_t) :: update_data_GMD
    type(aero_rep_update_data_modal_binned_mass_GSD_t) :: update_data_GSD
    real(kind=dp) :: comp_start, comp_end
    integer :: local_comm

    if (present(mpi_comm)) then
      local_comm = mpi_comm
    else
      local_comm = MPI_COMM_WORLD
    endif
    MONARCH_PROCESS = camp_mpi_rank()
    allocate(this)
    if (.not.present(n_cells).or.n_cells==1) then
      this%solve_multiple_cells = .false.
    else
      this%solve_multiple_cells = .true.
      this%n_cells=n_cells
    end if
    this%output_file_title=output_file_title
    camp_solver_data => camp_solver_data_t()
    call assert_msg(332298164, camp_solver_data%is_solver_available(), &
            "No solver available")
    deallocate(camp_solver_data)
    allocate(this%specs_emi_id(15))
    allocate(this%specs_emi(size(this%specs_emi_id)))
    if (MONARCH_PROCESS==0) then
      call cpu_time(comp_start)
      settings_interface_file="settings/"//output_file_title//"/interface.json"
      call this%load(settings_interface_file)
      this%camp_core => camp_core_t(camp_config_file, this%n_cells)
      call this%camp_core%initialize()
      if (this%camp_core%get_aero_rep("MONARCH mass-based", aero_rep)) then
        select type (aero_rep)
          type is (aero_rep_modal_binned_mass_t)
            call this%camp_core%initialize_update_object( aero_rep, &
                                                             update_data_GMD)
            call this%camp_core%initialize_update_object( aero_rep, &
                                                             update_data_GSD)
            call assert(889473105, &
                      aero_rep%get_section_id("organic_matter", i_sect_om))
            call assert(648042550, &
                        aero_rep%get_section_id("black_carbon", i_sect_bc))
            i_sect_sulf=-1
            call assert(307728742, &
                        aero_rep%get_section_id("other_PM", i_sect_opm))
            class default
            call die_msg(351392791, &
                         "Wrong type for aerosol representation "// &
                         "'MONARCH mass-based'")
        end select
      else
        i_sect_om = -1
        i_sect_bc = -1
        i_sect_sulf = -1
        i_sect_opm = -1
      end if
      this%tracer_starting_id = starting_id
      this%tracer_ending_id = ending_id
      call this%create_map()
      call this%load_init_conc()
      pack_size = this%camp_core%pack_size() + &
              update_data_GMD%pack_size() + &
              update_data_GSD%pack_size() + &
              camp_mpi_pack_size_integer_array(this%map_monarch_id) + &
              camp_mpi_pack_size_integer_array(this%map_camp_id) + &
              camp_mpi_pack_size_integer_array(this%init_conc_camp_id) + &
              camp_mpi_pack_size_real_array(this%init_conc) + &
              camp_mpi_pack_size_integer(this%gas_phase_water_id) + &
              camp_mpi_pack_size_integer(i_sect_om) + &
              camp_mpi_pack_size_integer(i_sect_bc) + &
              camp_mpi_pack_size_integer(i_sect_sulf) + &
              camp_mpi_pack_size_integer(i_sect_opm)
      pack_size = pack_size + camp_mpi_pack_size_integer(this%n_photo_rxn)
      do i = 1, this%n_photo_rxn
        pack_size = pack_size + this%photo_rxns(i)%pack_size( local_comm )
      end do
      pack_size = pack_size + camp_mpi_pack_size_real_array(this%base_rates)
      pack_size = pack_size + camp_mpi_pack_size_integer_array(this%specs_emi_id)
      pack_size = pack_size + camp_mpi_pack_size_real_array(this%specs_emi)
      allocate(buffer(pack_size))
      pos = 0
      call this%camp_core%bin_pack(buffer, pos)
      call update_data_GMD%bin_pack(buffer, pos)
      call update_data_GSD%bin_pack(buffer, pos)
      call camp_mpi_pack_integer_array(buffer, pos, this%map_monarch_id)
      call camp_mpi_pack_integer_array(buffer, pos, this%map_camp_id)
      call camp_mpi_pack_integer_array(buffer, pos, this%init_conc_camp_id)
      call camp_mpi_pack_real_array(buffer, pos, this%init_conc)
      call camp_mpi_pack_integer(buffer, pos, this%gas_phase_water_id)
      call camp_mpi_pack_integer(buffer, pos, i_sect_om)
      call camp_mpi_pack_integer(buffer, pos, i_sect_bc)
      call camp_mpi_pack_integer(buffer, pos, i_sect_sulf)
      call camp_mpi_pack_integer(buffer, pos, i_sect_opm)
      call camp_mpi_pack_integer(buffer, pos, this%n_photo_rxn)
      do i = 1, this%n_photo_rxn
        call this%photo_rxns(i)%bin_pack( buffer, pos, local_comm )
      end do
      call camp_mpi_pack_real_array(buffer, pos, this%base_rates)
      call camp_mpi_pack_integer_array(buffer, pos, this%specs_emi_id)
      call camp_mpi_pack_real_array(buffer, pos, this%specs_emi)
    endif
    call camp_mpi_bcast_integer(pack_size, local_comm)
    if (MONARCH_PROCESS/=0) then
      allocate(buffer(pack_size))
    end if
    call camp_mpi_bcast_packed(buffer, local_comm)
    if (MONARCH_PROCESS/=0) then
      this%camp_core => camp_core_t(n_cells=this%n_cells)
      pos = 0
      call this%camp_core%bin_unpack(buffer, pos)
      call update_data_GMD%bin_unpack(buffer, pos)
      call update_data_GSD%bin_unpack(buffer, pos)
      call camp_mpi_unpack_integer_array(buffer, pos, this%map_monarch_id)
      call camp_mpi_unpack_integer_array(buffer, pos, this%map_camp_id)
      call camp_mpi_unpack_integer_array(buffer, pos, this%init_conc_camp_id)
      call camp_mpi_unpack_real_array(buffer, pos, this%init_conc)
      call camp_mpi_unpack_integer(buffer, pos, this%gas_phase_water_id)
      call camp_mpi_unpack_integer(buffer, pos, i_sect_om)
      call camp_mpi_unpack_integer(buffer, pos, i_sect_bc)
      call camp_mpi_unpack_integer(buffer, pos, i_sect_sulf)
      call camp_mpi_unpack_integer(buffer, pos, i_sect_opm)
      call camp_mpi_unpack_integer(buffer, pos, this%n_photo_rxn)
      if( allocated( this%photo_rxns  ) ) deallocate( this%photo_rxns  )
      allocate(this%photo_rxns(this%n_photo_rxn))
      allocate(this%base_rates(this%n_photo_rxn))
      do i = 1, this%n_photo_rxn
        call this%photo_rxns(i)%bin_unpack( buffer, pos, local_comm )
      end do
      call camp_mpi_unpack_real_array(buffer, pos, this%base_rates)
      call camp_mpi_unpack_integer_array(buffer, pos, this%specs_emi_id)
      call camp_mpi_unpack_real_array(buffer, pos, this%specs_emi)
    end if
    deallocate(buffer)
    call this%camp_core%solver_initialize(n_cells_tstep)
    this%camp_state => this%camp_core%new_state()
    if(this%output_file_title=="cb05_paperV2") then
      allocate(this%offset_photo_rates_cells(this%n_cells))
      this%offset_photo_rates_cells(:) = 0.
      do z =1, this%n_cells
        do i = 1, this%n_photo_rxn
          base_rate = this%base_rates(i)
          call this%photo_rxns(i)%set_rate(base_rate)
          call this%camp_core%update_data(this%photo_rxns(i),z)
        end do
      end do
      deallocate(this%offset_photo_rates_cells)
    end if
    call camp_mpi_barrier(MPI_COMM_WORLD)
    do z =1, this%n_cells
      if (i_sect_bc>0) then
        call update_data_GMD%set_GMD(i_sect_bc, 1.18d-8)
        call update_data_GSD%set_GSD(i_sect_bc, 2.00d0)
        call this%camp_core%update_data(update_data_GMD)
        call this%camp_core%update_data(update_data_GSD)
      end if
      if (i_sect_sulf>0) then
        call update_data_GMD%set_GMD(i_sect_sulf, 6.95d-8)
        call update_data_GSD%set_GSD(i_sect_sulf, 2.12d0)
        call this%camp_core%update_data(update_data_GMD)
        call this%camp_core%update_data(update_data_GSD)
      end if
      if (i_sect_opm>0) then
        call update_data_GMD%set_GMD(i_sect_opm, 2.12d-8)
        call update_data_GSD%set_GSD(i_sect_opm, 2.24d0)
        call this%camp_core%update_data(update_data_GMD)
        call this%camp_core%update_data(update_data_GSD)
      end if
    end do
    !unique_names=this%camp_core%unique_names()
    !do i=1, size(unique_names)
    !  print*,i,trim(unique_names(i)%string)
    !end do
    if (MONARCH_PROCESS==0) then
      call cpu_time(comp_end)
      write(*,*) "Initialization time: ", comp_end-comp_start, " s"
    end if
  end function constructor

  subroutine integrate(this, curr_time, time_step, I_W, I_E, I_S, &
                  I_N, temperature, MONARCH_conc, water_conc, &
                  water_vapor_index, air_density, pressure, conv, i_hour,&
          NUM_TIME_STEP,solver_stats, DIFF_CELLS, i_time)
    class(camp_monarch_interface_t) :: this
    real, intent(in) :: curr_time
    real(kind=dp), intent(in) :: time_step
    integer, intent(in) :: I_W
    integer, intent(in) :: I_E
    integer, intent(in) :: I_S
    integer, intent(in) :: I_N
    real, intent(in) :: temperature(:,:,:)
    real, intent(inout) :: MONARCH_conc(:,:,:,:)
    real, intent(in) :: water_conc(:,:,:,:)
    integer, intent(in) :: water_vapor_index
    real, intent(in) :: air_density(:,:,:)
    real, intent(in) :: pressure(:,:,:)
    real, intent(in) :: conv(:,:,:)
    integer, intent(inout) :: i_hour
    integer, intent(in) :: NUM_TIME_STEP
    character(len=*),intent(in) :: DIFF_CELLS
    integer, intent(in) :: i_time
    type(chem_spec_data_t), pointer :: chem_spec_data
    integer, parameter :: emi_len=1
    real, allocatable :: rate_emi(:,:)
    character, allocatable :: buffer(:)
    integer(kind=i_kind) :: pos, pack_size
    integer :: rank, ierr, n_ranks
    real(kind=dp), allocatable :: mpi_conc(:)
    character(len=:), allocatable :: file_name
    integer :: i, j, k, i_spec, z, o, t, r, i_cell, i_photo_rxn
    integer :: NUM_VERT_CELLS, i_hour_max
    real :: press_init, press_end, press_range,&
            emi_slide, press_norm
    integer :: n_cells
    real(kind=dp) :: comp_start, comp_end
    type(solver_stats_t), intent(inout) :: solver_stats
    integer :: state_size_per_cell, n_cell_check
    integer :: counterLS = 0
    real :: timeLS = 0.0
    real :: timeCvode = 0.0

    if(this%n_cells==1) then
      state_size_per_cell = 0
    else
      state_size_per_cell = this%camp_core%size_state_per_cell
    end if
    NUM_VERT_CELLS = size(MONARCH_conc,3)
    if(this%output_file_title=="cb05_paperV2") then
      call assert_msg(731700229, &
              this%camp_core%get_chem_spec_data(chem_spec_data), &
              "No chemical species data in camp_core.")
      n_cells=(I_E - I_W+1)*(I_N - I_S+1)*NUM_VERT_CELLS
      i_hour_max=24
      allocate(rate_emi(i_hour_max,n_cells))
      rate_emi(:,:)=0.0
      if(DIFF_CELLS=="ON") then
        press_init = 100000.!Should be equal to mock_monarch
        press_end = 10000.
        press_range = press_end-press_init
        do i=I_W, I_E
          do j=I_S, I_N
            do k=1, NUM_VERT_CELLS
              o = (j-1)*(I_E) + (i-1)
              z = (k-1)*(I_E*I_N) + o
              press_norm=(press_end-pressure(i,j,k))/(press_range)
              do t=1,12 !12 first hours
                rate_emi(t,z+1)=press_norm
              end do
            end do
          end do
        end do
      else
        do t=1,12
          rate_emi(t,:)=1.0
        end do
      end if
      call camp_mpi_barrier(MPI_COMM_WORLD)
    end if
    i_hour = int(curr_time/60)+1
    if(mod(int(curr_time),60)==0) then
      if (camp_mpi_rank()==0) then
        write(*,*) "i_hour", i_hour,"i_time", i_time
      end if
    end if
    if(.not.this%solve_multiple_cells) then
      do i=I_W, I_E
        do j=I_S, I_N
          do k=1, NUM_VERT_CELLS
            o = (j-1)*(I_E) + (i-1)
            z = (k-1)*(I_E*I_N) + o
            call this%camp_state%env_states(1)%set_temperature_K( &
              real( temperature(i,j,k), kind=dp ) )
            call this%camp_state%env_states(1)%set_pressure_Pa(   &
              real( pressure(i,j,k), kind=dp ) )
            do r=1, size(this%camp_state%state_var)
              this%camp_state%state_var(r) = 0.
            end do
            this%camp_state%state_var(this%map_camp_id(:)) = &
                            MONARCH_conc(i,j,k,this%map_monarch_id(:))
            !print*,"MONARCH_conc381",MONARCH_conc(i,j,k,this%map_monarch_id(:))
            !print*,"state_var421",this%camp_state%state_var(:)
            if(this%output_file_title=="monarch_cb05") then
              this%camp_state%state_var(this%gas_phase_water_id) = &
              water_conc(1,1,1,water_vapor_index)
            else
              this%camp_state%state_var(this%gas_phase_water_id) = &
                      water_conc(1,1,1,water_vapor_index) * &
                              mwair / mwwat * 1.e6
            end if
            !print*,"state_var430",this%camp_state%state_var(:)
            if(this%output_file_title=="cb05_paperV2") then
              do r=1,size(this%specs_emi_id)
                this%camp_state%state_var(this%specs_emi_id(r))=&
                        this%camp_state%state_var(this%specs_emi_id(r))&
                                +this%specs_emi(r)*rate_emi(i_hour,z+1)*conv(i,j,k)
              end do
            !print*,"state_var436",this%camp_state%state_var(1)
            end if
            call cpu_time(comp_start)
            call this%camp_core%solve(this%camp_state, real(time_step*60., kind=dp),solver_stats=solver_stats)
            call cpu_time(comp_end)
            comp_time = comp_time + (comp_end-comp_start)
            MONARCH_conc(i,j,k,this%map_monarch_id(:)) = &
                    this%camp_state%state_var(this%map_camp_id(:))
          end do
        end do
      end do
    else
      do r=1, size(this%camp_state%state_var)
        this%camp_state%state_var(r) = 0.
      end do
      do i=I_W, I_E
        do j=I_S, I_N
          do k=1, NUM_VERT_CELLS
            o = (j-1)*(I_E) + (i-1)
            z = (k-1)*(I_E*I_N) + o
            call this%camp_state%env_states(z+1)%set_temperature_K(real(temperature(i,j,k),kind=dp))
            call this%camp_state%env_states(z+1)%set_pressure_Pa(real(pressure(i,j,k),kind=dp))
            this%camp_state%state_var(this%map_camp_id(:) + (z*state_size_per_cell))&
             = MONARCH_conc(i,j,k,this%map_monarch_id(:))
            !print*,"MONARCH_conc381",MONARCH_conc(i,j,k,this%map_monarch_id(:))
            !print*,"state_var421",this%camp_state%state_var(:)
            if(this%output_file_title=="monarch_cb05") then
              this%camp_state%state_var(this%gas_phase_water_id+(z*state_size_per_cell)) = &
                      water_conc(1,1,1,water_vapor_index)
            else
              this%camp_state%state_var(this%gas_phase_water_id+(z*state_size_per_cell)) = &
                      water_conc(1,1,1,water_vapor_index) * mwair / mwwat * 1.e6
            end if
            !print*,"state_var430",this%camp_state%state_var(:)
            if(this%output_file_title=="cb05_paperV2") then
              do r=1,size(this%specs_emi_id)
                this%camp_state%state_var(this%specs_emi_id(r)+z*state_size_per_cell)=&
                        this%camp_state%state_var(this%specs_emi_id(r)+z*state_size_per_cell)&
                                +this%specs_emi(r)*rate_emi(i_hour,z+1)*conv(i,j,k)
              end do
            endif
            !print*,"state_var436",this%camp_state%state_var(1+z*state_size_per_cell)
          end do
        end do
      end do
      call cpu_time(comp_start)
      call this%camp_core%solve(this%camp_state, &
              real(time_step*60., kind=dp), solver_stats = solver_stats)
      call cpu_time(comp_end)
      comp_time = comp_time + (comp_end-comp_start)
      do i=I_W, I_E
        do j=I_S, I_N
          do k=1, NUM_VERT_CELLS
            o = (j-1)*(I_E) + (i-1)
            z = (k-1)*(I_E*I_N) + o
            MONARCH_conc(i,j,k,this%map_monarch_id(:)) = &
                    this%camp_state%state_var(this%map_camp_id(:)+(z*state_size_per_cell))
          end do
        end do
      end do
    end if

  if(this%output_file_title=="cb05_paperV2") then
    deallocate(rate_emi)
  end if
  end subroutine integrate

  subroutine load(this, config_file)
    class(camp_monarch_interface_t) :: this
    character(len=:), allocatable :: config_file
    type(json_core), pointer :: json
    type(json_file) :: j_file
    type(json_value), pointer :: j_obj, j_next, j_child
    character(kind=json_ck, len=:), allocatable :: key, unicode_str_val
    character(len=:), allocatable :: str_val
    integer(kind=i_kind) :: var_type
    logical :: found

    this%species_map_data => property_t()
    this%init_conc_data => property_t()
    this%property_set => property_t()
    allocate(json)
    j_obj => null()
    j_next => null()
    call j_file%initialize()
    call j_file%get_core(json)
    call assert_msg(207035903, allocated(config_file), &
              "Received non-allocated string for file path")
    call assert_msg(368569727, trim(config_file)/="", &
              "Received empty string for file path")
    inquire( file=config_file, exist=found )
    call assert_msg(134309013, found, "Cannot find file: "// &
              config_file)
    call j_file%load_file(filename = config_file)
    call j_file%get('monarch-data(1)', j_obj)
    do while (associated(j_obj))
      call json%get(j_obj, 'type', unicode_str_val, found)
      call assert_msg(236838162, found, "Missing type in json input file "// &
              config_file)
      str_val = unicode_str_val
      if (str_val=="SPECIES_MAP") then
        call json%get_child(j_obj, j_child)
        do while (associated(j_child))
          call json%info(j_child, name=key, var_type=var_type)
          if (key/="type".and.key/="name") then
            call this%species_map_data%load(json, j_child, .false., key)
          end if
          j_next => j_child
          call json%get_next(j_next, j_child)
        end do
      else if (str_val=="INIT_CONC") then
        call json%get_child(j_obj, j_child)
        do while (associated(j_child))
          call json%info(j_child, name=key, var_type=var_type)
          if (key/="type".and.key/="name") then
            call this%init_conc_data%load(json, j_child, .false., key)
          end if
          j_next => j_child
          call json%get_next(j_next, j_child)
        end do
      else
        call this%property_set%load(json, j_obj, .false., str_val)
      end if
      j_next => j_obj
      call json%get_next(j_next, j_obj)
    end do
    call j_file%destroy()
    call json%destroy()
    deallocate(json)
  end subroutine load

  subroutine create_map(this)
    class(camp_monarch_interface_t) :: this
    type(chem_spec_data_t), pointer :: chem_spec_data
    class(aero_rep_data_t), pointer :: aero_rep_ptr
    type(property_t), pointer :: gas_species_list, aero_species_list, species_data
    character(len=:), allocatable :: key_name, spec_name, rep_name
    integer(kind=i_kind) :: i_spec, num_spec
    integer :: i_rxn, i_photo_rxn, i_base_rate, i_mech, i
    type(mechanism_data_t), pointer :: mechanism
    class(rxn_data_t), pointer :: rxn
    character(len=:), allocatable :: key, str_val, rxn_key, rate_key, rxn_val
    real(kind=dp) :: rate_val

    key = "MONARCH mod37"
    call assert(418262750, this%camp_core%get_mechanism(key, mechanism))
    rxn_key = "type"
    rxn_val = "PHOTOLYSIS"
    rate_key = "base rate"
    this%n_photo_rxn = 0
    do i_mech = 1, size(this%camp_core%mechanism)
      do i_rxn = 1, this%camp_core%mechanism(i_mech)%val%size()
        rxn => this%camp_core%mechanism(i_mech)%val%get_rxn(i_rxn)
        call assert(106297725, rxn%property_set%get_string(rxn_key, str_val))
        if (trim(str_val)==rxn_val) this%n_photo_rxn = this%n_photo_rxn + 1
      end do
    end do
    allocate(this%photo_rxns(this%n_photo_rxn))
    allocate(this%base_rates(this%n_photo_rxn))
    i_photo_rxn = 0
    do i_mech = 1, size(this%camp_core%mechanism)
      do i_rxn = 1, this%camp_core%mechanism(i_mech)%val%size()
        rxn => this%camp_core%mechanism(i_mech)%val%get_rxn(i_rxn)
        call assert(799145523, rxn%property_set%get_string(rxn_key, str_val))
        if (trim(str_val)/=rxn_val) cycle
        i_photo_rxn = i_photo_rxn + 1
        call assert_msg(501329648, &
                rxn%property_set%get_real(rate_key, rate_val), &
                "Missing 'base rate' for photolysis reaction "// &
                        trim(to_string(i_photo_rxn)))
        this%base_rates(i_photo_rxn) = rate_val
        select type (rxn_photo => rxn)
        class is (rxn_photolysis_t)
          call this%camp_core%initialize_update_object(rxn_photo, &
                  this%photo_rxns(i_photo_rxn))
        class default
          call die(722633162)
        end select
      end do
    end do
    key_name = "gas-phase species"
    call assert_msg(939097252, &
            this%species_map_data%get_property_t(key_name, gas_species_list), &
            "Missing set of gas-phase species MONARCH ids")
    num_spec = gas_species_list%size()
    key_name = "aerosol-phase species"
    if (this%species_map_data%get_property_t(key_name, &
            aero_species_list)) then
      num_spec = num_spec + aero_species_list%size()
    end if
    allocate(this%monarch_species_names(num_spec))
    allocate(this%map_monarch_id(num_spec))
    allocate(this%map_camp_id(num_spec))
    call assert_msg(731700229, &
            this%camp_core%get_chem_spec_data(chem_spec_data), &
            "No chemical species data in camp_core.")
    key_name = "gas-phase water"
    call assert_msg(413656652, &
            this%species_map_data%get_string(key_name, spec_name), &
            "Missing gas-phase water species for MONARCH interface.")
    this%gas_phase_water_id = chem_spec_data%gas_state_id(spec_name)
    call assert_msg(910692272, this%gas_phase_water_id>0, &
            "Could not find gas-phase water species '"//spec_name//"'.")
    call gas_species_list%iter_reset()
    i_spec = 1
    do while (gas_species_list%get_key(spec_name))
      this%monarch_species_names(i_spec)%string = spec_name
      call assert_msg(599522862, &
              gas_species_list%get_property_t(val=species_data), &
              "Missing species data for '"//spec_name//"' in CAMP-camp "// &
              "<-> MONARCH species map.")
      key_name = "monarch id"
      call assert_msg(643926329, &
              species_data%get_int(key_name, this%map_monarch_id(i_spec)), &
              "Missing monarch id for species '"//spec_name//" in "// &
              "CAMP-camp <-> MONARCH species map.")
      this%map_monarch_id(i_spec) = this%map_monarch_id(i_spec) + &
              this%tracer_starting_id - 1
      call assert_msg(450258014, &
              this%map_monarch_id(i_spec)<=this%tracer_ending_id, &
              "Monarch id for species '"//spec_name//"' out of specified "// &
              "tracer array bounds.")
      this%map_camp_id(i_spec) = chem_spec_data%gas_state_id(spec_name)
      call assert_msg(916977002, this%map_camp_id(i_spec)>0, &
                "Could not find species '"//spec_name//"' in CAMP-camp.")
      call gas_species_list%iter_next()
      i_spec = i_spec + 1
    end do

    if (associated(aero_species_list)) then
      call aero_species_list%iter_reset()
      do while(aero_species_list%get_key(spec_name))
        this%monarch_species_names(i_spec)%string = spec_name
        call assert_msg(567689501, &
                aero_species_list%get_property_t(val=species_data), &
                "Missing species data for '"//spec_name//"' in " //&
                "CAMP-camp <-> MONARCH species map.")
        key_name = "monarch id"
        call assert_msg(615451741, &
                species_data%get_int(key_name, this%map_monarch_id(i_spec)), &
                "Missing monarch id for species '"//spec_name//"' in "// &
                "CAMP-camp <-> MONARCH species map.")
        this%map_monarch_id(i_spec) = this%map_monarch_id(i_spec) + &
                this%tracer_starting_id - 1
        call assert_msg(382644266, &
                this%map_monarch_id(i_spec)<=this%tracer_ending_id, &
                "Monarch id for species '"//spec_name//"' out of "// &
                "specified tracer array bounds.")
        key_name = "aerosol representation name"
        call assert_msg(963222513, &
                species_data%get_string(key_name, rep_name), &
                "Missing aerosol representation name for species '"// &
                spec_name//"' in CAMP-camp <-> MONARCH species map.")
        this%map_camp_id(i_spec) = 0
        call assert_msg(377850668, &
                this%camp_core%get_aero_rep(rep_name, aero_rep_ptr), &
                "Could not find aerosol representation '"//rep_name//"'")
        this%map_camp_id(i_spec) = aero_rep_ptr%spec_state_id(spec_name)
        call assert_msg(887136850, this%map_camp_id(i_spec) > 0, &
                "Could not find aerosol species '"//spec_name//"' in "// &
                "aerosol representation '"//rep_name//"'.")
        call aero_species_list%iter_next()
        i_spec = i_spec + 1
      end do
    end if

  end subroutine create_map

  subroutine load_init_conc(this)
    class(camp_monarch_interface_t) :: this
    type(chem_spec_data_t), pointer :: chem_spec_data
    class(aero_rep_data_t), pointer :: aero_rep_ptr
    type(property_t), pointer :: gas_species_list, aero_species_list, species_data
    character(len=:), allocatable :: key_name, spec_name, rep_name
    integer(kind=i_kind) :: i_spec, num_spec, i
    real :: factor_ppb_to_ppm

    if(this%output_file_title=="cb05_paperV2") then
      factor_ppb_to_ppm=1.0E-3
    else
      factor_ppb_to_ppm=1.0
    end if
    num_spec = 0
    key_name = "gas-phase species"
    if (this%init_conc_data%get_property_t(key_name, gas_species_list)) then
      num_spec = num_spec + gas_species_list%size()
    end if
    key_name = "aerosol-phase species"
    if (this%init_conc_data%get_property_t(key_name, aero_species_list)) then
      num_spec = num_spec + aero_species_list%size()
    end if
    call assert_msg(885063268, &
            this%camp_core%get_chem_spec_data(chem_spec_data), &
            "No chemical species data in camp_core.")
    allocate(this%init_conc_camp_id(num_spec))
    allocate(this%init_conc(num_spec))
    if (associated(gas_species_list)) then
      call gas_species_list%iter_reset()
      i_spec = 1
      do while (gas_species_list%get_key(spec_name))
        call assert_msg(325582312, &
                gas_species_list%get_property_t(val=species_data), &
                "Missing species data for '"//spec_name//"' for "// &
                "CAMP-camp initial concentrations.")
        key_name = "init conc"
        call assert_msg(445070498, &
                species_data%get_real(key_name, this%init_conc(i_spec)), &
                "Missing 'init conc' for species '"//spec_name//" for "// &
                "CAMP-camp initial concentrations.")
        this%init_conc(i_spec) = this%init_conc(i_spec) * factor_ppb_to_ppm
        this%init_conc_camp_id(i_spec) = &
                chem_spec_data%gas_state_id(spec_name)
        call assert_msg(940200584, this%init_conc_camp_id(i_spec)>0, &
                "Could not find species '"//spec_name//"' in CAMP-camp.")
        call gas_species_list%iter_next()
        i_spec = i_spec + 1
      end do
    end if

    if (associated(aero_species_list)) then
      call aero_species_list%iter_reset()
      do while(aero_species_list%get_key(spec_name))
        call assert_msg(331096555, &
                aero_species_list%get_property_t(val=species_data), &
                "Missing species data for '"//spec_name//"' for " //&
                "CAMP-camp initial concentrations.")
        key_name = "init conc"
        call assert_msg(782275469, &
                species_data%get_real(key_name, this%init_conc(i_spec)), &
                "Missing 'init conc' for species '"//spec_name//"' for "// &
                "CAMP-camp initial concentrations.")

        key_name = "aerosol representation name"
        call assert_msg(150863332, &
                species_data%get_string(key_name, rep_name), &
                "Missing aerosol representation name for species '"// &
                spec_name//"' for CAMP-camp initial concentrations.")
        this%init_conc_camp_id(i_spec) = 0
        call assert_msg(258814777, &
                this%camp_core%get_aero_rep(rep_name, aero_rep_ptr), &
                "Could not find aerosol representation '"//rep_name//"'")
        this%init_conc_camp_id(i_spec) = &
                aero_rep_ptr%spec_state_id(spec_name)
        call assert_msg(437149649, this%init_conc_camp_id(i_spec) > 0, &
                "Could not find aerosol species '"//spec_name//"' in "// &
                "aerosol representation '"//rep_name//"'.")
        call aero_species_list%iter_next()
        i_spec = i_spec + 1
      end do
    end if

    this%specs_emi_id(1)=chem_spec_data%gas_state_id("SO2")
    this%specs_emi_id(2)=chem_spec_data%gas_state_id("NO2")
    this%specs_emi_id(3)=chem_spec_data%gas_state_id("NO")
    this%specs_emi_id(4)=chem_spec_data%gas_state_id("NH3")
    this%specs_emi_id(5)=chem_spec_data%gas_state_id("CO")
    this%specs_emi_id(6)=chem_spec_data%gas_state_id("ALD2")
    this%specs_emi_id(7)=chem_spec_data%gas_state_id("FORM")
    this%specs_emi_id(8)=chem_spec_data%gas_state_id("ETH")
    this%specs_emi_id(9)=chem_spec_data%gas_state_id("IOLE")
    this%specs_emi_id(10)=chem_spec_data%gas_state_id("OLE")
    this%specs_emi_id(11)=chem_spec_data%gas_state_id("TOL")
    this%specs_emi_id(12)=chem_spec_data%gas_state_id("XYL")
    this%specs_emi_id(13)=chem_spec_data%gas_state_id("PAR")
    this%specs_emi_id(14)=chem_spec_data%gas_state_id("ISOP")
    this%specs_emi_id(15)=chem_spec_data%gas_state_id("MEOH")
    this%specs_emi(1)=1.06E-09
    this%specs_emi(2)=7.56E-12
    this%specs_emi(3)=1.44E-10
    this%specs_emi(4)=8.93E-09
    this%specs_emi(5)=1.96E-09
    this%specs_emi(6)=4.25E-12
    this%specs_emi(7)=1.02E-11
    this%specs_emi(8)=4.62E-11
    this%specs_emi(9)=1.49E-11
    this%specs_emi(10)=1.49E-11
    this%specs_emi(11)=1.53E-11
    this%specs_emi(12)=1.40E-11
    this%specs_emi(13)=4.27E-10
    this%specs_emi(14)=6.03E-12
    this%specs_emi(15)=5.92E-13

  end subroutine load_init_conc

  subroutine get_init_conc(this, MONARCH_conc, water_conc, &
      WATER_VAPOR_ID,i_W,I_E,I_S,I_N)
    class(camp_monarch_interface_t) :: this
    real, intent(inout) :: MONARCH_conc(:,:,:,:)
    real, intent(inout) :: water_conc(:,:,:,:)
    integer, intent(in) :: WATER_VAPOR_ID
    integer, intent(in) :: i_W,I_E,I_S,I_N
    integer(kind=i_kind) :: i_spec, water_id,i,j,k,r,NUM_VERT_CELLS,state_size_per_cell
    NUM_VERT_CELLS=size(MONARCH_conc,3)
    this%camp_state%state_var(this%init_conc_camp_id(:)) = this%init_conc(:)
    if(this%n_cells==1) then
      forall (i_spec = 1:size(this%map_monarch_id))
        MONARCH_conc(:,:,:,this%map_monarch_id(i_spec)) = &
            this%camp_state%state_var(this%map_camp_id(i_spec))
      end forall
      this%camp_state%state_var(this%gas_phase_water_id +(r*state_size_per_cell)) = &
          water_conc(i,j,k,WATER_VAPOR_ID) * &
              mwair / mwwat * 1.e6
    else
      do i=i_W, I_E
        do j=I_S, I_N
          do k=1, NUM_VERT_CELLS
            r=(k-1)*(I_E*I_N) + (j-1)*(I_E) + i-1
            forall (i_spec = 1:size(this%map_monarch_id))
              this%camp_state%state_var(this%init_conc_camp_id(i_spec)&
              +r*state_size_per_cell) = this%init_conc(i_spec)
            end forall
            do i_spec=1, size(this%map_monarch_id)
              MONARCH_conc(i,j,k,this%map_monarch_id(i_spec)) = &
                this%camp_state%state_var(this%map_camp_id(i_spec))
            end do
            this%camp_state%state_var(this%gas_phase_water_id +(r*state_size_per_cell)) = &
                    water_conc(i,j,k,WATER_VAPOR_ID) * &
                            mwair / mwwat * 1.e6
          end do
        end do
      end do
    end if
  end subroutine get_init_conc

  elemental subroutine finalize(this)
    type(camp_monarch_interface_t), intent(inout) :: this
    if (associated(this%camp_core)) &
            deallocate(this%camp_core)
    if (allocated(this%monarch_species_names)) &
            deallocate(this%monarch_species_names)
    if (allocated(this%map_monarch_id)) &
            deallocate(this%map_monarch_id)
    if (allocated(this%map_camp_id)) &
            deallocate(this%map_camp_id)
    if (allocated(this%init_conc_camp_id)) &
            deallocate(this%init_conc_camp_id)
    if (allocated(this%init_conc)) &
            deallocate(this%init_conc)
  end subroutine finalize

end module
