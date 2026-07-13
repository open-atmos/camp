module boxmodel_io
  use netcdf
  use mpi, only: MPI_INFO_NULL

  use camp_chem_spec_data, only: chem_spec_data_t, CHEM_SPEC_GAS_PHASE
  use camp_aero_rep_data, only: aero_rep_data_ptr
  use camp_aero_rep_modal_binned_mass, only: aero_rep_modal_binned_mass_t
  use camp_camp_core, only: camp_core_t
  use camp_constants, only: i_kind, dp, sp
  use camp_util, only: assert_msg, string_t, warn_msg, unique_character, to_string
  use camp_property, only: property_t
  use camp_mpi

  use boxmodel_time_control, only: time_control_t
  use camp_camp_state, only: camp_state_t

  use boxmodel_log
  implicit none

  integer(kind=dp), parameter  :: FILL_VALUE_DP = -9999.9999

  type ncdf_writer
    ! can we write aerosol information
    logical, allocatable                :: write_aerosol_rep_fg(:)
    integer(kind=i_kind), allocatable   :: aero_rep_to_write(:)
    ! information about state size in case of multiple cells
    integer(kind=i_kind)                :: state_size_per_cell
    ! output file id
    integer(kind=i_kind)   :: ncid
    ! dimensions ids
    integer(kind=i_kind)   :: time_dimid, cells_dimid, species_dimid
    integer(kind=i_kind)   :: aer_rep_dimid, aer_section_dimid, bins_dimid
    integer(kind=i_kind)   :: aer_phase_dimid, gas_species_dimid
    integer(kind=i_kind)   :: aer_species_dimid
    ! variable ids
    integer(kind=i_kind)   :: cells_varid
    integer(kind=i_kind)   :: pressure_varid, temperature_varid, humidity_varid
    integer(kind=i_kind)   :: time_varid, sza_id, height_id, species_name_varid
    integer(kind=i_kind)   :: latitude_varid, longitude_varid, altitude_varid
    integer(kind=i_kind)   :: aer_rep_name_varid, aer_section_name_varid
    integer(kind=i_kind)   :: n_aer_section_varid, num_bins_varid
    integer(kind=i_kind)   :: aer_phase_name_varid, num_aer_species_varid
    integer(kind=i_kind)   :: num_phase_varid
    integer(kind=i_kind)   :: gas_species_names_varid, gas_species_molw_varid
    integer(kind=i_kind)   :: gas_species_hlc_varid, gas_species_psat_varid
    integer(kind=i_kind)   :: aer_species_names_varid, aer_species_molw_varid
    integer(kind=i_kind)   :: aer_species_hlc_varid, aer_species_psat_varid
    integer(kind=i_kind)   :: bin_diameter_varid, number_concentration_varid
    integer(kind=i_kind)   :: effective_radius_varid, phase_mass_varid
    integer(kind=i_kind)   :: phase_molw_varid
    integer(kind=i_kind)   :: emissions_varid
    ! concentrations varids
    integer(kind=i_kind)   :: gas_concentrations, aer_concentrations
    ! map between output array and state_var indices
    integer(kind=i_kind), allocatable   :: num_sections(:), num_bins(:, :)
    type(string_t), dimension(:), allocatable :: phase_names
    integer(kind=i_kind), allocatable   :: phase_ids(:, :, :), num_unique_phases(:, :)
    integer(kind=i_kind), allocatable   :: output_map(:, :, :, :, :)
  contains
    procedure :: init
    procedure :: create_aerosol_output_map
    procedure :: sync
    procedure :: close

  end type ncdf_writer

contains
  !>  handling netcdf errors
  subroutine handle_err(camp_error_code, status)
    integer, intent(in) :: camp_error_code, status

    call assert_msg(camp_error_code, &
      status == nf90_noerr, &
      trim(nf90_strerror(status)))

  end subroutine handle_err

  !> initialize output netcdf file with dimensions
  subroutine init(this, filename, camp_core, ncells, time_control, state_size_per_cell, mpi_comm)
    class(ncdf_writer), intent(out) :: this
    class(time_control_t), intent(in) :: time_control
    character(len=*), intent(in) :: filename
    type(camp_core_t), pointer, intent(in) :: camp_core
    integer(kind=i_kind), intent(in)  :: ncells
    integer(kind=i_kind), intent(in)  :: state_size_per_cell
    integer(kind=i_kind), intent(in)  :: mpi_comm

    character(len=:), allocatable :: local_filename

    type(aero_rep_data_ptr), pointer :: aerosol_representation
    type(chem_spec_data_t), pointer :: chem_spec_data
    type(time_control_t) :: time_control_copy
    integer(kind=i_kind) :: ntimes

    integer(kind=i_kind) :: aero_rep_name_len_dimid, aero_section_name_len_dimid
    integer(kind=i_kind) :: aero_phase_name_len_id, aero_species_names_len_dimid
    integer(kind=i_kind) :: gas_species_names_len_dimid
    integer :: status, ispec, iaerrep, iaersect, i, iphase, jphase, k, itime,i_cell
    character(len=40) :: unit_name
    character(len=:), allocatable :: aer_rep_name, aer_sect_name, spec_name
    character(len=40), allocatable :: all_phase_names(:), unique_phase_names(:)
    type(string_t), allocatable, dimension(:) :: aer_species_names
    integer(kind=i_kind), allocatable, dimension(:, :) :: unique_phase_indices
    integer(kind=i_kind), allocatable, dimension(:) :: num_phases
    logical :: found
    integer(kind=i_kind) :: hour, min, sec, tz_hour, tz_min

    integer(kind=i_kind) :: max_gas_name_len, max_aero_rep_names_len, max_phases_names_len
    integer(kind=i_kind) :: max_aero_section_names_len, max_num_aero_sections, max_num_bins
    integer(kind=i_kind) :: max_num_phases, max_num_aero_species, max_aero_species_names_len
    integer(kind=i_kind) :: num_aerosol_sections, num_aero_rep, num_gas_specs
    integer(kind=i_kind), allocatable, dimension(:) :: num_bins

    ! species properties
    type(property_t), pointer :: property_set
    real(kind=dp)        :: molw, hlc, psat
    character(len=:), allocatable :: key_name

    character(len=:), allocatable :: phase_name

    class(aero_rep_modal_binned_mass_t), pointer :: modal_binned_aero_rep

    ! parallel mpi things
    character, allocatable :: buffer(:)
    integer(kind=i_kind) :: pos, BOXMOD_PROCESS
    integer :: pack_size
    integer :: old_fill_option
    character(len=100) :: string_val

    this%state_size_per_cell = state_size_per_cell

    ! \todo add parallel io capability with arguments comm and info
    BOXMOD_PROCESS = camp_mpi_rank()
    local_filename = filename//"_"//trim(to_string(BOXMOD_PROCESS))//".nc"

    if (BOXMOD_PROCESS == 0) then
      call assert_msg(604489561, &
        camp_core%get_chem_spec_data(chem_spec_data), &
        "chem spec data not initialized in camp_core")
    end if

    call thread_log%debug("creating file: "//trim(local_filename))

    status = nf90_create( &
      path=local_filename, &
      cmode=IOR(NF90_CLOBBER, NF90_NETCDF4), &
      ncid=this%ncid)
    call handle_err(446036539, status)

    ! define time dimension
    time_control_copy = time_control
    ntimes = 0
    do

      if (time_control_copy%is_output_time()) then
        ntimes = ntimes + 1
      endif
      if (.not. time_control_copy%increment()) exit
    end do
    status = nf90_def_dim(this%ncid, name="time", len=ntimes, dimid=this%time_dimid)
    call handle_err(319017077, status)

    ! define cells dimension (1 axis only)
    status = nf90_def_dim(this%ncid, name="cell", len=ncells, dimid=this%cells_dimid)
    call handle_err(511222049, status)

    ! define cell variable
    status = nf90_def_var(this%ncid, "cell", nf90_int, &
      (/this%cells_dimid/), &
      this%cells_varid)

    ! identify aerosol representation we can print
    if (BOXMOD_PROCESS == 0) then
      allocate (this%write_aerosol_rep_fg(size(camp_core%aero_rep)))
      this%write_aerosol_rep_fg = .false.
      do iaerrep = 1, size(camp_core%aero_rep)
        select type (rep => camp_core%aero_rep(iaerrep)%val)
         type is (aero_rep_modal_binned_mass_t)
          this%write_aerosol_rep_fg(iaerrep) = .true.
         class default
          call warn_msg(221238979, &
            "cannot output info about aerosol representation: "// &
            camp_core%aero_rep(iaerrep)%val%name())
        end select
      end do
      allocate (this%aero_rep_to_write(count(this%write_aerosol_rep_fg)))
      i = 0
      do iaerrep = 1, size(camp_core%aero_rep)
        if (this%write_aerosol_rep_fg(iaerrep)) then
          i = i + 1
          this%aero_rep_to_write(i) = iaerrep
        end if
      end do

      ! count aerosol representations
      num_aero_rep = size(this%aero_rep_to_write)
    end if

    call camp_mpi_bcast_integer(num_aero_rep, mpi_comm)
    allocate (this%num_sections(num_aero_rep))
    status = nf90_def_dim(this%ncid, name="aero_rep_dim", &
      len=num_aero_rep, dimid=this%aer_rep_dimid)
    call handle_err(254417574, status)

    ! define aerosol representation names character length
    if (BOXMOD_PROCESS == 0) then
      max_aero_rep_names_len = 0
      do i = 1, size(this%aero_rep_to_write)
        iaerrep = this%aero_rep_to_write(i)
        if (len(camp_core%aero_rep(iaerrep)%val%name()) > max_aero_rep_names_len) then
          max_aero_rep_names_len = len(camp_core%aero_rep(iaerrep)%val%name())
        end if
      end do
    end if

    call camp_mpi_bcast_integer(max_aero_rep_names_len, mpi_comm)
    status = nf90_def_dim(this%ncid, name="aero_rep_name_len", &
      len=max_aero_rep_names_len, dimid=aero_rep_name_len_dimid)
    call handle_err(248810290, status)

    ! define aerosol representations names
    status = nf90_def_var(this%ncid, "aerosol_representation_names", nf90_char, &
      (/aero_rep_name_len_dimid, this%aer_rep_dimid/), &
      this%aer_rep_name_varid &
      )
    call handle_err(34823873, status)

    ! find aerosol sections
    if (BOXMOD_PROCESS == 0) then
      max_num_aero_sections = 0
      max_aero_section_names_len = 0
      do i = 1, size(this%aero_rep_to_write)
        iaerrep = this%aero_rep_to_write(i)
        select type (modal_binned_aero_rep => camp_core%aero_rep(iaerrep)%val)
         type is (aero_rep_modal_binned_mass_t)
          if (size(modal_binned_aero_rep%section_name) > max_num_aero_sections) then
            max_num_aero_sections = size(modal_binned_aero_rep%section_name)
          end if
          do iaersect = 1, size(modal_binned_aero_rep%section_name)
            if (len(modal_binned_aero_rep%section_name(iaersect)%string) > max_aero_section_names_len) then
              max_aero_section_names_len = len(modal_binned_aero_rep%section_name(iaersect)%string)
            end if
          end do
        end select
      end do
    end if

    call camp_mpi_bcast_integer(max_num_aero_sections, mpi_comm)
    allocate (this%num_bins(max_num_aero_sections, num_aero_rep))

    status = nf90_def_dim(this%ncid, name="aero_section_dim", &
      len=max_num_aero_sections, dimid=this%aer_section_dimid)
    call handle_err(347930582, status)

    call camp_mpi_bcast_integer(max_num_aero_sections, mpi_comm)
    status = nf90_def_dim(this%ncid, name="aero_section_name_dim", &
      len=max_aero_section_names_len, dimid=aero_section_name_len_dimid)
    call handle_err(309009599, status)

    ! define aerosol section variables
    status = nf90_def_var(this%ncid, "num_aerosol_sections", nf90_int, &
      (/this%aer_rep_dimid/), &
      this%n_aer_section_varid)
    call handle_err(227410017, status)

    status = nf90_def_var(this%ncid, "aerosol_sections_names", nf90_char, &
      (/aero_section_name_len_dimid, this%aer_section_dimid, this%aer_rep_dimid/), &
      this%aer_section_name_varid)
    call handle_err(630997007, status)

    ! define bins numbers
    if (BOXMOD_PROCESS == 0) then
      max_num_bins = 0
      do i = 1, size(this%aero_rep_to_write)
        iaerrep = this%aero_rep_to_write(i)
        select type (modal_binned_aero_rep => camp_core%aero_rep(iaerrep)%val)
         type is (aero_rep_modal_binned_mass_t)

          num_bins = modal_binned_aero_rep%num_bins()
          if (maxval(num_bins) > max_num_bins) then
            max_num_bins = maxval(num_bins)
          end if

        end select
      end do
    end if

    call camp_mpi_bcast_integer(max_num_bins, mpi_comm)
    status = nf90_def_dim(this%ncid, name="aero_bins_dim", &
      len=max_num_bins, dimid=this%bins_dimid)
    call handle_err(230761708, status)

    status = nf90_def_var(this%ncid, "aerosol_bins_numbers", nf90_int, &
      (/this%aer_section_dimid, this%aer_rep_dimid/), &
      this%num_bins_varid)
    call handle_err(158244967, status)

    ! define phases
    if (BOXMOD_PROCESS == 0) then
      max_num_phases = size(camp_core%aero_phase)
      allocate (this%phase_names(max_num_phases))
      max_phases_names_len = 0
      do iphase = 1, max_num_phases
        phase_name = camp_core%aero_phase(iphase)%val%name()
        this%phase_names(iphase) = string_t(phase_name)
        if (len(phase_name) > max_phases_names_len) then
          max_phases_names_len = len(phase_name)
        end if
      end do
      max_aero_species_names_len = 0
      max_num_aero_species = 0
      do iphase = 1, size(this%phase_names)
        aer_species_names = camp_core%aero_phase(iphase)%val%get_species_names()
        if (size(aer_species_names) > max_num_aero_species) then
          max_num_aero_species = size(aer_species_names)
        end if
        do ispec = 1, size(aer_species_names)
          if (len(aer_species_names(ispec)%string) > max_aero_species_names_len) then
            max_aero_species_names_len = len(aer_species_names(ispec)%string)
          end if
        end do
      end do
    end if

    call camp_mpi_bcast_integer(max_num_phases, mpi_comm)
    status = nf90_def_dim(this%ncid, name="aero_phases_dim", &
      len=max_num_phases, dimid=this%aer_phase_dimid)
    call handle_err(390054617, status)

    call camp_mpi_bcast_integer(max_phases_names_len, mpi_comm)
    status = nf90_def_dim(this%ncid, name="aero_phase_name_dim", &
      len=max_phases_names_len, dimid=aero_phase_name_len_id)
    call handle_err(720170893, status)

    call camp_mpi_bcast_integer(max_num_aero_species, mpi_comm)
    status = nf90_def_dim(this%ncid, name="aero_species_dim", &
      len=max_num_aero_species, dimid=this%aer_species_dimid)
    call handle_err(720464381, status)

    call camp_mpi_bcast_integer(max_aero_species_names_len, mpi_comm)
    status = nf90_def_dim(this%ncid, name="aero_species_name_len", &
      len=max_aero_species_names_len, dimid=aero_species_names_len_dimid)
    call handle_err(367623596, status)

    status = nf90_def_var(this%ncid, "aerosol_phase_name", nf90_char, &
      (/aero_phase_name_len_id, this%aer_phase_dimid/), &
      this%aer_phase_name_varid &
      )
    call handle_err(181151986, status)

    status = nf90_def_var(this%ncid, "num_aerosol_species", nf90_int, &
      (/this%aer_phase_dimid/), &
      this%num_aer_species_varid &
      )
    call handle_err(278070470, status)

    status = nf90_def_var(this%ncid, "aerosol_species_name", nf90_char, &
      (/aero_species_names_len_dimid, this%aer_species_dimid, this%aer_phase_dimid/), &
      this%aer_species_names_varid &
      )
    call handle_err(266701006, status)

    status = nf90_def_var(this%ncid, "aerosol_species_molw", nf90_double, &
      (/this%aer_species_dimid, this%aer_phase_dimid/), &
      this%aer_species_molw_varid)
    call handle_err(117702477, status)
    status = nf90_put_att(this%ncid, this%aer_species_molw_varid, "units", "kg/mol")
    call handle_err(192787595, status)

    status = nf90_def_var(this%ncid, "aerosol_species_hlc", nf90_double, &
      (/this%aer_species_dimid, this%aer_phase_dimid/), &
      this%aer_species_hlc_varid)
    call handle_err(317130366, status)
    status = nf90_put_att(this%ncid, this%aer_species_hlc_varid, "units", "M/Pa")
    call handle_err(688524813, status)

    status = nf90_def_var(this%ncid, "aerosol_species_psat", nf90_double, &
      (/this%aer_species_dimid, this%aer_phase_dimid/), &
      this%aer_species_psat_varid &
      )
    call handle_err(223619136, status)
    status = nf90_put_att(this%ncid, this%aer_species_psat_varid, "units", "atm")
    call handle_err(332361472, status)

    ! aerosol properties
    status = nf90_def_var(this%ncid, "bin_diameter", nf90_double, &
      (/this%bins_dimid, this%aer_section_dimid, this%aer_rep_dimid, this%cells_dimid, this%time_dimid/), &
      this%bin_diameter_varid)
    call handle_err(289541578, status)
    status = nf90_put_att(this%ncid, this%bin_diameter_varid, "units", "m")
    call handle_err(644367529, status)

    status = nf90_def_var(this%ncid, "number_concentration", nf90_double, &
      (/this%bins_dimid, this%aer_section_dimid, this%aer_rep_dimid, this%cells_dimid, this%time_dimid/), &
      this%number_concentration_varid)
    call handle_err(167181975, status)
    status = nf90_put_att(this%ncid, this%number_concentration_varid, "units", "/m3")
    call handle_err(644367529, status)

    status = nf90_def_var(this%ncid, "effective_radius", nf90_double, &
      (/this%bins_dimid, this%aer_section_dimid, this%aer_rep_dimid, this%cells_dimid, this%time_dimid/), &
      this%effective_radius_varid)
    call handle_err(227113833, status)
    status = nf90_put_att(this%ncid, this%effective_radius_varid, "units", "m")
    call handle_err(644367529, status)

    status = nf90_def_var(this%ncid, "phase_mass", nf90_double, &
      (/this%bins_dimid, this%aer_phase_dimid, this%aer_section_dimid, &
      this%aer_rep_dimid, this%cells_dimid, this%time_dimid/), &
      this%phase_mass_varid)
    call handle_err(213059283, status)
    status = nf90_put_att(this%ncid, this%phase_mass_varid, "units", "kg/m3")
    call handle_err(305399630, status)

    status = nf90_def_var(this%ncid, "phase_average_molecular_weight", nf90_double, &
      (/this%bins_dimid, this%aer_phase_dimid, this%aer_section_dimid, &
      this%aer_rep_dimid, this%cells_dimid, this%time_dimid/), &
      this%phase_molw_varid)
    call handle_err(428412698, status)
    status = nf90_put_att(this%ncid, this%phase_molw_varid, "units", "kg/mol")
    call handle_err(265400687, status)

    status = nf90_def_var(this%ncid, "aerosol_species_concentration", nf90_double, &
      (/ &
      this%aer_species_dimid, &
      this%aer_phase_dimid, &
      this%bins_dimid, &
      this%aer_section_dimid, &
      this%aer_rep_dimid, &
      this%cells_dimid, &
      this%time_dimid &
      /), &
      this%aer_concentrations &
      )
    call handle_err(354888741, status)
    ! status = nf90_def_var_fill(this%ncid, this%aer_concentrations, &
    !                            0, 0.0)
    ! call handle_err(182053806, status)
    status = nf90_put_att(this%ncid, this%aer_concentrations, "units", "kg/m3")
    call handle_err(893984668, status)

    ! define gas phase species dimensions and variables
    if (BOXMOD_PROCESS == 0) then
      num_gas_specs = chem_spec_data%size(spec_phase=CHEM_SPEC_GAS_PHASE)
    end if
    call camp_mpi_bcast_integer(num_gas_specs, mpi_comm)
    status = nf90_def_dim(this%ncid, name="gas_species_dim", &
      len=num_gas_specs, &
      dimid=this%gas_species_dimid &
      )
    call handle_err(197309463, status)

    if (BOXMOD_PROCESS == 0) then
      max_gas_name_len = 0
      do ispec = 1, chem_spec_data%size(spec_phase=CHEM_SPEC_GAS_PHASE)
        if (len(chem_spec_data%gas_state_name(ispec)) > max_gas_name_len) then
          max_gas_name_len = len(chem_spec_data%gas_state_name(ispec))
        end if
      end do
    end if
    call camp_mpi_bcast_integer(max_gas_name_len, mpi_comm)
    status = nf90_def_dim(this%ncid, name="gas_species_name_len", &
      len=max_gas_name_len, dimid=gas_species_names_len_dimid)
    call handle_err(367623596, status)

    status = nf90_def_var(this%ncid, "gas_species_name", nf90_char, &
      (/gas_species_names_len_dimid, this%gas_species_dimid/), &
      this%gas_species_names_varid &
      )
    call handle_err(276518376, status)

    status = nf90_def_var(this%ncid, "gas_species_concentrations", nf90_double, &
      (/this%gas_species_dimid, this%cells_dimid, this%time_dimid/), &
      this%gas_concentrations)
    call handle_err(419637398, status)
    ! status = nf90_def_var_fill(this%ncid, this%gas_concentrations, &
    !                            0, 0.0)
    ! call handle_err(113005664, status)
    status = nf90_put_att(this%ncid, this%gas_concentrations, "units", "ppm")
    call handle_err(309928578, status)

    status = nf90_def_var(this%ncid, "gas_species_molw", nf90_double, &
      (/this%gas_species_dimid/), &
      this%gas_species_molw_varid &
      )
    call handle_err(125437350, status)
    status = nf90_put_att(this%ncid, this%gas_species_molw_varid, "units", "kg/mol")
    call handle_err(201514497, status)

    status = nf90_def_var(this%ncid, "gas_species_hlc", nf90_double, &
      (/this%gas_species_dimid/), &
      this%gas_species_hlc_varid &
      )
    call handle_err(171442386, status)
    status = nf90_put_att(this%ncid, this%gas_species_hlc_varid, "units", "M/Pa")
    call handle_err(209013890, status)

    status = nf90_def_var(this%ncid, "gas_species_psat", nf90_double, &
      (/this%gas_species_dimid/), &
      this%gas_species_psat_varid &
      )
    call handle_err(223619136, status)
    status = nf90_put_att(this%ncid, this%gas_species_psat_varid, "units", "atm")
    call handle_err(332361472, status)

    ! define latitude variable
    status = nf90_def_var(this%ncid, "latitude", nf90_double, &
      (/this%cells_dimid, this%time_dimid/), &
      this%latitude_varid)
    call handle_err(671436195, status)
    status = nf90_def_var_fill(this%ncid, this%latitude_varid, &
      0, FILL_VALUE_DP)
    call handle_err(286071899, status)

    ! define longitude variable
    status = nf90_def_var(this%ncid, "longitude", nf90_double, &
      (/this%cells_dimid, this%time_dimid/), &
      this%longitude_varid)
    call handle_err(760719185, status)
    status = nf90_def_var_fill(this%ncid, this%longitude_varid, &
      0, FILL_VALUE_DP)
    call handle_err(289890128, status)

    ! define altitude variable
    status = nf90_def_var(this%ncid, "altitude", nf90_double, &
      (/this%cells_dimid, this%time_dimid/), &
      this%altitude_varid)
    call handle_err(753921601, status)
    status = nf90_def_var_fill(this%ncid, this%altitude_varid, &
      0, FILL_VALUE_DP)
    call handle_err(182053806, status)

    ! define pressure variable
    status = nf90_def_var(this%ncid, "pressure", nf90_double, &
      (/this%cells_dimid, this%time_dimid/), &
      this%pressure_varid)
    call handle_err(438689656, status)
    status = nf90_def_var_fill(this%ncid, this%pressure_varid, &
      0, FILL_VALUE_DP)
    call handle_err(372550199, status)

    ! define temperature variable
    status = nf90_def_var(this%ncid, "temperature", nf90_double, &
      (/this%cells_dimid, this%time_dimid/), &
      this%temperature_varid)
    call handle_err(491856708, status)
    status = nf90_def_var_fill(this%ncid, this%temperature_varid, &
      0, FILL_VALUE_DP)
    call handle_err(361404091, status)

    ! define relative humidity variable
    status = nf90_def_var(this%ncid, "humidity", nf90_double, &
      (/this%cells_dimid, this%time_dimid/), &
      this%humidity_varid)
    call handle_err(594236370, status)
    status = nf90_def_var_fill(this%ncid, this%humidity_varid, &
      0, FILL_VALUE_DP)
    call handle_err(835042054, status)

    ! define solar zenith angle
    status = nf90_def_var(this%ncid, "solar_zenith_angle", nf90_double, &
      (/this%cells_dimid, this%time_dimid/), &
      this%sza_id)
    call handle_err(205538888, status)
    status = nf90_def_var_fill(this%ncid, this%sza_id, &
      0, FILL_VALUE_DP)
    call handle_err(878077584, status)

    ! define height
    status = nf90_def_var(this%ncid, "height", nf90_double, &
      (/this%cells_dimid, this%time_dimid/), &
      this%height_id)
    call handle_err(445077591, status)
    status = nf90_def_var_fill(this%ncid, this%height_id, &
      0, FILL_VALUE_DP)
    call handle_err(898479405, status)

    ! define time variable
    status = nf90_def_var(this%ncid, "time", nf90_double, &
      (/this%time_dimid/), &
      this%time_varid)

    call handle_err(576558615, status)
    status = nf90_def_var_fill(this%ncid, this%time_varid, &
      0, FILL_VALUE_DP)
    call handle_err(584079474, status)
    hour = int(floor(time_control%hour, kind=i_kind))
    min = int(floor((time_control%hour - hour)*60.0), kind=i_kind)
    sec = int(floor(time_control%hour*3600.-hour*3600.-min*60.), kind=i_kind)
    tz_hour = int(floor(time_control%tz), kind=i_kind)
    tz_min = int(floor(time_control%tz - tz_hour)*60.0, kind=i_kind)
    write (unit_name, "(A14, I4.4, A1, I2.2, A1, I2.2, 1X, I2.2, A1, I2.2, A1, I2.2, 1X, I3.2, A1, I2.2)") &
      "seconds since ", time_control%year, "-", time_control%month, "-", time_control%day, &
      hour, ":", min, ":", sec, &
      tz_hour, ":", tz_min

    status = nf90_put_att(this%ncid, this%time_varid, "units", trim(unit_name))
    call handle_err(435973364, status)

    status = nf90_put_att(this%ncid, this%time_varid, "long_name", "elapsed time")
    call handle_err(690176947, status)

    ! status = nf90_set_fill(this%ncid, NF90_NOFILL, old_fill_option)
    ! call handle_err(670815532, status)

    ! ---------------------------------
    ! EMISSIONS VARIABLE
    ! ---------------------------------

    status = nf90_def_var(this%ncid, "emissions_rate", nf90_double, &
      (/this%gas_species_dimid, this%cells_dimid, this%time_dimid/), &
      this%emissions_varid)
    call handle_err(303266163, status)
    status = nf90_put_att(this%ncid, this%emissions_varid, "units", "molec/cm2/s")
    call handle_err(170642801, status)

    call thread_log%debug("netcdf initialization done, file:"//local_filename//" ncid: "//to_string(this%ncid))

    status = nf90_enddef(this%ncid)
    call handle_err(304509321, status)

    call thread_log%debug("netcdf definition done")



    do i_cell = 1, ncells
      status = nf90_put_var( &
        ncid=this%ncid, &
        varid=this%cells_varid, &
        values=(/i_cell/), &
        start=(/i_cell/), &
        count=(/1/) &
        )
      call handle_err(196305551, status)
    enddo


    if (BOXMOD_PROCESS == 0) then
      do iphase = 1, max_num_phases
        status = nf90_put_var( &
          ncid=this%ncid, &
          varid=this%aer_phase_name_varid, &
          values=(/this%phase_names(iphase)%string/), &
          start=(/1, iphase/), &
          count=(/len(this%phase_names(iphase)%string), 1/) &
          )
        call handle_err(196305553, status)

        aer_species_names = camp_core%aero_phase(iphase)%val%get_species_names()
        status = nf90_put_var( &
          ncid=this%ncid, &
          varid=this%num_aer_species_varid, &
          values=(/size(aer_species_names)/), &
          start=(/iphase/), &
          count=(/1/) &
          )
        call handle_err(136409362, status)

        do ispec = 1, size(aer_species_names)
          status = nf90_put_var( &
            ncid=this%ncid, &
            varid=this%aer_species_names_varid, &
            values=(/aer_species_names(ispec)%string/), &
            start=(/1, ispec, iphase/), &
            count=(/len(aer_species_names(ispec)%string), 1, 1/) &
            )
          call handle_err(505759794, status)

          ! put properties of aerosol species
          call assert_msg(372988677, &
            chem_spec_data%get_property_set(trim(aer_species_names(ispec)%string), property_set), &
            "no property set found for species "//trim(aer_species_names(ispec)%string))

          key_name = "molecular weight [kg mol-1]"
          if (.not. property_set%get_real(key_name, molw)) then
            molw = FILL_VALUE_DP
          end if
          status = nf90_put_var( &
            ncid=this%ncid, &
            varid=this%aer_species_molw_varid, &
            values=(/molw/), &
            start=(/ispec, iphase/), &
            count=(/1, 1/) &
            )
          call handle_err(119158435, status)

          key_name = "HLC(298K) [M Pa-1]"
          if (.not. property_set%get_real(key_name, hlc)) then
            hlc = FILL_VALUE_DP
          end if
          status = nf90_put_var( &
            ncid=this%ncid, &
            varid=this%aer_species_hlc_varid, &
            values=(/hlc/), &
            start=(/ispec, iphase/), &
            count=(/1, 1/) &
            )
          call handle_err(286789373, status)

          key_name = "psat_298_atm"
          if (.not. property_set%get_real(key_name, psat)) then
            psat = FILL_VALUE_DP
          end if
          status = nf90_put_var( &
            ncid=this%ncid, &
            varid=this%aer_species_psat_varid, &
            values=(/psat/), &
            start=(/ispec, iphase/), &
            count=(/1, 1/) &
            )
          call handle_err(324897506, status)

        end do
      end do

      ! put aerosol representations names
      do i = 1, size(this%aero_rep_to_write)
        iaerrep = this%aero_rep_to_write(i)
        aer_rep_name = camp_core%aero_rep(iaerrep)%val%name()
        status = nf90_put_var( &
          ncid=this%ncid, &
          varid=this%aer_rep_name_varid, &
          values=camp_core%aero_rep(iaerrep)%val%name(), &
          start=(/1, iaerrep/), &
          count=(/len(aer_rep_name), 1/) &
          )
        call handle_err(170038191, status)
      end do

      ! put aerosol sections numbers and names
      do i = 1, size(this%aero_rep_to_write)
        iaerrep = this%aero_rep_to_write(i)
        select type (modal_binned_aero_rep => camp_core%aero_rep(iaerrep)%val)
         type is (aero_rep_modal_binned_mass_t)
          num_aerosol_sections = size(modal_binned_aero_rep%section_name)
          this%num_sections(i) = num_aerosol_sections
          status = nf90_put_var( &
            ncid=this%ncid, &
            varid=this%n_aer_section_varid, &
            values=(/num_aerosol_sections/), &
            start=(/i/), &
            count=(/1/) &
            )
          call handle_err(330352410, status)
          do iaersect = 1, num_aerosol_sections
            aer_sect_name = modal_binned_aero_rep%section_name(iaersect)%string
            status = nf90_put_var( &
              ncid=this%ncid, &
              varid=this%aer_section_name_varid, &
              values=aer_sect_name, &
              start=(/1, iaersect, i/), &
              count=(/len(aer_sect_name), 1, 1/) &
              )
            call handle_err(182971292, status)
          end do
        end select
      end do

      ! put bin numbers
      do i = 1, size(this%aero_rep_to_write)
        iaerrep = this%aero_rep_to_write(i)
        select type (modal_binned_aero_rep => camp_core%aero_rep(iaerrep)%val)
         type is (aero_rep_modal_binned_mass_t)

          num_aerosol_sections = size(modal_binned_aero_rep%section_name)
          num_bins = modal_binned_aero_rep%num_bins()
          this%num_bins(:, i) = num_bins
          status = nf90_put_var( &
            ncid=this%ncid, &
            varid=this%num_bins_varid, &
            values=num_bins, &
            start=(/1, i/), &
            count=(/size(num_bins), 1/) &
            )
          call handle_err(420407650, status)

        end select
      end do

      ! put gas species names and properties
      do ispec = 1, chem_spec_data%size(spec_phase=CHEM_SPEC_GAS_PHASE)
        spec_name = trim(chem_spec_data%gas_state_name(ispec))
        status = nf90_put_var( &
          ncid=this%ncid, &
          varid=this%gas_species_names_varid, &
          values=spec_name, &
          start=(/1, ispec/), &
          count=(/len(spec_name), 1/) &
          )
        call handle_err(533372823, status)

        call assert_msg(315103732, &
          chem_spec_data%get_property_set(spec_name, property_set), &
          "no property set found for species "//spec_name)

        key_name = "molecular weight [kg mol-1]"
        if (.not. property_set%get_real(key_name, molw)) then
          molw = FILL_VALUE_DP
        end if
        status = nf90_put_var( &
          ncid=this%ncid, &
          varid=this%gas_species_molw_varid, &
          values=(/molw/), &
          start=(/ispec/), &
          count=(/1/) &
          )

        key_name = "HLC(298K) [M Pa-1]"
        if (.not. property_set%get_real(key_name, hlc)) then
          hlc = FILL_VALUE_DP
        end if
        status = nf90_put_var( &
          ncid=this%ncid, &
          varid=this%gas_species_hlc_varid, &
          values=(/hlc/), &
          start=(/ispec/), &
          count=(/1/) &
          )

        key_name = "psat_298_atm"
        if (.not. property_set%get_real(key_name, psat)) then
          psat = FILL_VALUE_DP
        end if
        status = nf90_put_var( &
          ncid=this%ncid, &
          varid=this%gas_species_psat_varid, &
          values=(/psat/), &
          start=(/ispec/), &
          count=(/1/) &
          )

      end do

      allocate (this%num_unique_phases( &
        size(this%aero_rep_to_write), &
        max_num_aero_sections) &
        )

      allocate (this%phase_ids( &
        size(this%aero_rep_to_write), &
        max_num_aero_sections, &
        max_num_phases) &
        )
      this%phase_ids = -1

      allocate (this%output_map( &
        size(this%aero_rep_to_write), &
        max_num_aero_sections, &
        max_num_bins, &
        max_num_phases, &
        max_num_aero_species) &
        )
      this%output_map = -1

      call this%create_aerosol_output_map(camp_core)

#ifdef CAMP_USE_MPI
! send output_map to all processes
      pack_size = output_map_pack_size(this%output_map, comm=mpi_comm)
      pack_size = pack_size + &
        camp_mpi_pack_size_integer_array(this%num_sections, comm=mpi_comm) + &
        camp_mpi_pack_size_integer_array_2d(this%num_bins, comm=mpi_comm) + &
        camp_mpi_pack_size_integer_array_2d(this%num_unique_phases, comm=mpi_comm) + &
        camp_mpi_pack_size_integer_array_3d(this%phase_ids, comm=mpi_comm)
      allocate (buffer(pack_size))
      pos = 0
      call output_map_bin_pack(this%output_map, buffer, pos, comm=mpi_comm)
      call camp_mpi_pack_integer_array(buffer, pos, this%num_sections, comm=mpi_comm)
      call camp_mpi_pack_integer_array_2d(buffer, pos, this%num_bins, comm=mpi_comm)
      call camp_mpi_pack_integer_array_2d(buffer, pos, this%num_unique_phases, comm=mpi_comm)
      call camp_mpi_pack_integer_array_3d(buffer, pos, this%phase_ids, comm=mpi_comm)

    end if !! boxmod_process == 0

    ! broadcast the buffer size
    call camp_mpi_bcast_integer(pack_size, comm=mpi_comm)

    if (BOXMOD_PROCESS .ne. 0) then
      ! allocate the buffer to receive data
      allocate (buffer(pack_size))
    end if

    ! broadcast the buffer
    call camp_mpi_bcast_packed(buffer, comm=mpi_comm)

    if (BOXMOD_PROCESS .ne. 0) then
      ! unpack the data
      pos = 0
      call output_map_bin_unpack(this%output_map, buffer, pos, comm=mpi_comm)
      call camp_mpi_unpack_integer_array(buffer, pos, this%num_sections, comm=mpi_comm)
      call camp_mpi_unpack_integer_array_2d(buffer, pos, this%num_bins, comm=mpi_comm)
      call camp_mpi_unpack_integer_array_2d(buffer, pos, this%num_unique_phases, comm=mpi_comm)
      call camp_mpi_unpack_integer_array_3d(buffer, pos, this%phase_ids, comm=mpi_comm)
#endif
    end if

#ifdef CAMP_USE_MPI
    deallocate (buffer)
#endif
  end subroutine

  subroutine create_aerosol_output_map(this, camp_core)
    class(ncdf_writer), intent(inout) :: this

    type(camp_core_t), pointer, intent(in) :: camp_core
    integer(kind=i_kind)   :: irep, iaerrep, isect, ibin, iphase, jphase, ispec, jspec, k
    integer(kind=i_kind)   :: curr_phase_indx, max_unique_phase

    integer(kind=i_kind)              :: num_aerosol_sections, num_spec, spec_id
    integer(kind=i_kind), allocatable :: num_bins(:), num_phases(:)
    character(len=:), allocatable     :: unique_name, curr_section_name
    character(len=:), allocatable     :: curr_bin_str, curr_phase_name
    TYPE(string_t), ALLOCATABLE, DIMENSION(:) :: spec_names, unique_names

    logical :: found

    max_unique_phase = size(this%phase_names)

    do irep = 1, size(this%aero_rep_to_write)
      iaerrep = this%aero_rep_to_write(irep)
      select type (modal_binned_aero_rep => camp_core%aero_rep(iaerrep)%val)
       type is (aero_rep_modal_binned_mass_t)

        unique_names = modal_binned_aero_rep%unique_names()

        num_aerosol_sections = size(modal_binned_aero_rep%section_name)
        num_bins = modal_binned_aero_rep%num_bins()
        num_phases = modal_binned_aero_rep%num_phases()

        iphase = 1
        ispec = 1
        do isect = 1, num_aerosol_sections

          curr_section_name = modal_binned_aero_rep%section_name(isect)%string

          do jphase = 1, num_phases(isect)
            ! Set the current phase name
            curr_phase_name = modal_binned_aero_rep%aero_phase(iphase)%val%name()

            curr_phase_indx = -1
            do k = 1, max_unique_phase
              if (this%phase_names(k)%string == curr_phase_name) then
                curr_phase_indx = k
                exit
              end if
            end do

            call assert_msg(178996356, &
              curr_phase_indx > 0, &
              "could not find curr_phase_name in the list of available phase names")

            ! if this phase is not already identified as one of the phases used by this section
            ! add it to the list
            found = .false.
            do k = 1, max_unique_phase
              if (this%phase_ids(irep, isect, k) == curr_phase_indx) then
                found = .true.
                exit
              end if
            end do
            if (.not. found) then
              do k = 1, size(this%phase_ids, 3)
                if (this%phase_ids(irep, isect, k) == -1) then
                  this%phase_ids(irep, isect, k) = curr_phase_indx
                  exit
                end if
              end do
            end if

            ! Get the list of species names in this phase
            spec_names = modal_binned_aero_rep%aero_phase(iphase)%val%get_species_names()

            do ibin = 1, num_bins(isect)
              ! Set the current bin label (except for single bins or mode)
              if (num_bins(isect) .gt. 1) then
                curr_bin_str = trim(to_string(ibin))//"."
              else
                curr_bin_str = ""
              end if

              ! Add species from this phase/bin
              num_spec = modal_binned_aero_rep%aero_phase(iphase)%val%size()
              do jspec = 1, num_spec
                unique_name = curr_section_name//"."// &
                  curr_bin_str//curr_phase_name//'.'// &
                  spec_names(jspec)%string
                spec_id = modal_binned_aero_rep%spec_state_id(unique_name)
                this%output_map(irep, isect, ibin, curr_phase_indx, jspec) = spec_id
                ispec = ispec + 1
              end do
              iphase = iphase + 1
            end do
          end do
        end do
      end select
    end do

    !! store the number of unique phases present in each section
    this%num_unique_phases(:, :) = 1
    do irep = 1, size(this%phase_ids, 1)
      do isect = 1, size(this%phase_ids, 2)
        do iphase = 1, size(this%phase_ids, 3)
          if (this%phase_ids(irep, isect, iphase) == -1) then
            this%num_unique_phases(irep, isect) = iphase - 1
            exit
          end if
        end do
      end do
    end do

  end subroutine create_aerosol_output_map

  subroutine sync(this)
    class(ncdf_writer), intent(inout) :: this

    integer(kind=i_kind) :: status

    status = nf90_sync(this%ncid)
    call handle_err(124460661, status)

  end subroutine sync

  subroutine close (this)
    class(ncdf_writer), intent(inout) :: this

    integer(kind=i_kind) :: status

    status = nf90_close(this%ncid)
    call handle_err(525971853, status)

  end subroutine close

  !> Determine the number of bytes required to pack the output_map
  integer(kind=i_kind) function output_map_pack_size(output_map, comm)
    integer(kind=i_kind), allocatable, dimension(:, :, :, :, :), intent(in) :: output_map
    !> MPI communicator
    integer, intent(in), optional :: comm

#ifdef CAMP_USE_MPI
    integer(kind=i_kind) :: l_comm
    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if
    output_map_pack_size = 0

    output_map_pack_size = output_map_pack_size + camp_mpi_pack_size_integer_array_5d(output_map, l_comm)

#else
    output_map_pack_size = 0
#endif
  end function output_map_pack_size

  !> Pack the given variable into a buffer, advancing position
  subroutine output_map_bin_pack(output_map, buffer, pos, comm)
    integer(kind=i_kind), allocatable, dimension(:, :, :, :, :), intent(in) :: output_map
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos
    !> MPI communicator
    integer, intent(in), optional :: comm

#ifdef CAMP_USE_MPI
    integer(kind=i_kind) :: prev_position, l_comm

    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if

    prev_position = pos
    call camp_mpi_pack_integer_array_5d(buffer, pos, output_map, l_comm)

    call assert(109979676, &
      pos - prev_position == output_map_pack_size(output_map, l_comm))
#endif

  end subroutine output_map_bin_pack

  !> Unpack the given variable from a buffer, advancing position
  subroutine output_map_bin_unpack(output_map, buffer, pos, comm)
    integer(kind=i_kind), allocatable, dimension(:, :, :, :, :), intent(inout) :: output_map
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos
    !> MPI communicator
    integer, intent(in), optional :: comm

#ifdef CAMP_USE_MPI
    integer(kind=i_kind) :: prev_position, l_comm

    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if

    prev_position = pos

    call camp_mpi_unpack_integer_array_5d(buffer, pos, output_map, l_comm)

    call assert(529867398, &
      pos - prev_position == output_map_pack_size(output_map, l_comm))
#endif
  end subroutine output_map_bin_unpack

end module boxmodel_io
