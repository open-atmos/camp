module boxmodel_photolysis
  use camp_constants, only: i_kind, sp, dp, const
  use camp_util, only: assert_msg, assert, string_t
  use camp_mpi

  use boxmodel_time_control, only: time_control_t
  use boxmodel_cloudj_interface, only: init, run_cloudj, cloudj_interface_t, JVN_, &
    JVMAP, JFACTA, NRATJ, JIND, TITLEJX

  use boxmodel_tuv_module
  use camp_mechanism_data, only: mechanism_data_t
  use camp_rxn_data, only: rxn_data_t
  use camp_property, only: property_t
  use camp_rxn_photolysis, only: rxn_photolysis_t, rxn_update_data_photolysis_t

  use boxmodel_log
  implicit none

  integer(kind=i_kind), parameter :: CLOUDJ_PHOTOLYSIS = 1, TUV_PHOTOLYSIS = 2

  type(cloudj_interface_t)  :: cloudj_interface
  !> TUV interface
  type(tuv_interface_t)     :: tuv_interface

  real(kind=dp) :: last_longitude, last_latitude, last_altitude

  !> map camp photolysis reaction index -> radiative transfer model reaction
  type :: photolysis_map_t
    integer(kind=i_kind) :: selected_photolysis

    !> tabled zenith angles
    integer(kind=i_kind) :: n_sza
    real(kind=dp), allocatable, dimension(:) :: sza  ! transfered

    !> reactions in the mechanism
    integer(kind=i_kind) :: nphot_reac
    integer(kind=i_kind), allocatable, dimension(:) :: n_assigned_j
    integer(kind=i_kind) :: max_assigned_j
    type(string_t), allocatable, dimension(:, :) :: photolysis_label
    !> the array that takes care of updating photolysis reactions
    !> indexed by cell id and reaction id
    type(rxn_update_data_photolysis_t), allocatable, dimension(:, :) :: photo_rate_updates ! transfered

    !> reactions in the photolysis interface
    integer(kind=i_kind) :: n_jvalues
    type(string_t), allocatable, dimension(:) :: jvalues_label
    real(kind=dp), allocatable, dimension(:, :):: jvalues !transfered

    !> for each photolysis reaction, index of the corresponding j-value in jvalues and jvalues_label
    real(kind=dp), allocatable, dimension(:, :) :: weighting_factors
    integer(kind=i_kind), allocatable, dimension(:, :) :: index_map ! transfered

  contains
    !> initialize photolysis map
    procedure, public :: init => init_photolysis_map

    !> get photolysis map info from tuv module
    procedure         :: set_from_tuv
    procedure         :: get_j

    !> routines for cloudj interface
    procedure         :: set_from_cloudj

    procedure         :: map_reactions

    procedure :: update_photolysis_reactions_rates

    procedure :: output_photolysis_table

    !> print photolysis map for debugging
    procedure, public :: print
    !> Determine the number of bytes required to pack the variable
    procedure, public :: pack_size
    !> Pack the given variable into a buffer, advancing position
    procedure, public :: bin_pack
    !> Unpack the given variable from a buffer, advancing position
    procedure, public :: bin_unpack

  end type photolysis_map_t

contains
  !> ----------------------------------------------------------------------
  !> Input/output of the original subroutine was adapted to the boxmodel
  !> and the code switched to fortran 90 by B. Aumont, summer 2016
  !> ----------------------------------------------------------------------
  !> Written by Sasha Madronich at NCAR 23 February 1989.
  !> Based on equations given by Paltridge and Platt [1976] "Radiative
  !> Processes in Meteorology and Climatology", Elsevier, pp. 62,63.
  !> Originally from Spencer, J.W., 1972, Fourier series representation
  !> of the  position of the sun, Search, 2:172.
  !> This subroutine calculates solar zenith and azimuth angles for a
  !> particular time and location.  Must specify:
  !>
  !> INPUT:
  !>  LAT - latitude in decimal degrees
  !>  LONG - longitude in decimal degrees
  !>  IDATE (not longer used) - Date at Greenwich
  !>                          - specify year (19yy), month (mm), day (dd)
  !>                            format is six-digit integer:  yymmdd)
  !>  GMT  - Greenwich mean time - decimal military eg.
  !>              22.75 = 45 min after ten pm gmt
  !> OUTPUT
  !>  Zenith
  !>  Azimuth
  !> NOTE:  this approximate program has no changes from year to year.
  !> ----------------------------------------------------------------------
  real(kind=dp) function zenith(lat, long, iiyear, imth, iday, gmt)
    !> latitude and longitude in decimal degrees
    real(kind=dp), intent(in)  :: LAT, LONG
    !> year, month, day
    integer(kind=i_kind), intent(in) :: IIYEAR, IMTH, IDAY
    !> greenwich mean time - decimal e.g.
    !> 22.75 = 45 min after 10 pm gmt
    real(kind=dp), intent(in)  :: GMT

    integer(kind=i_kind), dimension(12) :: IMN
    real(kind=dp) :: lbgmt, lzgmt

    real(kind=dp)    :: RLT, D, TZ, RDECL, EQR, EQH, ZPT
    real(kind=dp)    :: CSZ, ZR
    integer(kind=i_kind) :: IIY, IJD
    integer(kind=i_kind) :: I

    REAL(kind=dp) :: DR

    DR = const%pi/180.

    IMN = (/31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31/) ! day / month
    RLT = LAT*DR

! identify and correct leap years
    IIY = (IIYEAR/4)*4
    IF (IIY == IIYEAR) IMN(2) = 29

    ! compute current day of year IJD = 1 to 365
    IJD = 0
    DO I = 1, IMTH - 1
      IJD = IJD + IMN(I)
    END DO
    IJD = IJD + IDAY

    ! calculate fractional day from start of year:
    D = REAL(IJD - 1, kind=dp) + GMT/24.

    ! Equation 3.8 for "day-angle"
    TZ = 2.*const%pi*D/365.

    ! Equation 3.7 for declination in radians
    RDECL = 0.006918 - 0.399912*COS(TZ) + 0.070257*SIN(TZ) &
      - 0.006758*COS(2.*TZ) + 0.000907*SIN(2.*TZ) &
      - 0.002697*COS(3.*TZ) + 0.001480*SIN(3.*TZ)

    ! Equation 3.11 for Equation of time  in radians
    EQR = 0.000075 + 0.001868*COS(TZ) - 0.032077*SIN(TZ) &
      - 0.014615*COS(2.*TZ) - 0.040849*SIN(2.*TZ)

    ! convert equation of time to hours:
    EQH = EQR*24./(2.*const%PI)

    ! calculate local hour angle (hours):
    LBGMT = 12.-EQH - LONG*24./360

    ! convert to angle from GMT
    LZGMT = 15.*(GMT - LBGMT)
    ZPT = LZGMT*DR

    ! Equation 2.4 for cosine of zenith angle
    CSZ = SIN(RLT)*SIN(RDECL) + COS(RLT)*COS(RDECL)*COS(ZPT)
    ZR = ACOS(CSZ)
    ZENITH = ZR/DR

  end function zenith

  !> wrapper calling the photolysis initialization
  subroutine init_photolysis_map(this, photolysis_method, &
    atmosphere_profile_filename, &
    time_control, wind, chlr, &
    temperature, rh, psurf, cldoption, &
    altitude, longitude, latitude, &
    surf_albedo, o3col, so2col, no2col, &
    cloud_optical_depth, cloud_base_alt, cloud_top_alt, &
    aer_optical_depth, aer_ssa, aer_alpha, &
    air_density)

    class(photolysis_map_t), intent(inout) :: this
    character(len=*), intent(in)           :: photolysis_method
    character(len=*), intent(in)           :: atmosphere_profile_filename
    !> time information
    type(time_control_t), intent(in)       :: time_control
    !> surface wind (m/s)
    real(kind=dp), intent(in)              :: wind
    !> surface chlorophyl (mg/m3)
    real(kind=dp), intent(in)              :: chlr
    !> temperature (K)
    real(kind=dp), intent(in)              :: temperature
    !> relative humidity (dimensionless 0-1)
    real(kind=dp), intent(in)              :: rh
    !> surface pressure (hPa)
    real(kind=dp), intent(in)              :: psurf
    !> altitude (m)
    real(kind=dp), intent(in)              :: altitude
    !> longitude
    real(kind=dp), intent(in)              :: longitude
    !> latitude
    real(kind=dp), intent(in)              :: latitude
    !> surface albedo
    real(kind=dp), intent(in)              :: surf_albedo
    !> columns of o3, so2, no2 in DU
    real(kind=dp), intent(in)              :: o3col, so2col, no2col
    !> cloud info (km)
    real(kind=dp), intent(in)              :: cloud_optical_depth, cloud_base_alt, cloud_top_alt
    !> absorbing aerosol properties (km, dimensionless)
    real(kind=dp), intent(in)              :: aer_optical_depth, aer_ssa, aer_alpha
    !> air density (kg/m3)
    real(kind=dp), intent(in)              :: air_density
    !>       cldoption = 1  :  Clear sky J's
    !>       cldoption = 2  :  Averaged cloud cover
    !>       cldoption = 3  :  cloud-fract**3/2, then average cloud cover
    !>       cldoption = 4  :  ****not used
    !>       cldoption = 5  :  Random select NRANDO ICA's from all(Independent Column Atmos.)
    !>       cldoption = 6  :  Use all (up to 4) quadrature cloud cover QCAs (mid-pts of bin)
    !>       cldoption = 7  :  Use all (up to 4) QCAs (average clouds within each Q-bin)
    !>       cldoption = 8  :  Calculate J's for ALL ICAs (up to 20,000 per cell!)
    integer(kind=i_kind), intent(in)      :: cldoption

    integer(kind=i_kind)  :: i

    !> set photolysis method
    select case (photolysis_method)
     case ("CLOUDJ")
      this%selected_photolysis = CLOUDJ_PHOTOLYSIS
     case ("TUV")
      this%selected_photolysis = TUV_PHOTOLYSIS
     case default
      call die_msg(320136674, "invalid photolysis method:'"//photolysis_method//"'")
    end select

    call warn_assert_msg(207785065, this%nphot_reac > 0, &
      "calculating photolysis for a mechanism without matching photolysis reactions")

    select case (this%selected_photolysis)
      case (CLOUDJ_PHOTOLYSIS)
        call cloudj_interface%init(atmosphere_profile_filename, time_control%month, latitude, wind, chlr, &
          temperature, rh, psurf, cldoption)
        ! replace default cloudj info with our mechanism photolysis reaction data
        call this%set_from_cloudj(altitude)
      case (TUV_PHOTOLYSIS)
        call tuv_interface%tuv_init( &
          real(latitude, kind=sp), &
          real(longitude, kind=sp), &
          real(altitude, kind=sp), &
          real(psurf, kind=sp), &
          real(surf_albedo, kind=sp), &
          real(o3col, kind=sp), &
          real(so2col, kind=sp), &
          real(no2col, kind=sp), &
          real(cloud_optical_depth, kind=sp), &
          real(cloud_base_alt, kind=sp), &
          real(cloud_top_alt, kind=sp), &
          real(aer_optical_depth, kind=sp), &
          real(aer_ssa, kind=sp), &
          real(aer_alpha, kind=sp), &
          real(temperature, kind=sp), &
          real(air_density, kind=sp), &
          time_control)
        call this%set_from_tuv()
    end select

    call this%map_reactions()

    call this%output_photolysis_table(&
      photolysis_method//"_table_"//trim(to_string(camp_mpi_rank()))//".csv")

  end subroutine init_photolysis_map

  subroutine output_photolysis_table(this, filename)
    class(photolysis_map_t), intent(in) :: this
    character(len=*), intent(in) :: filename

    integer(kind=i_kind) :: tabled_output_unit, ij, it
    ! open tabled output file
    tabled_output_unit = get_unit()
    open (unit=tabled_output_unit, file=filename, status="replace")

    do ij = 1, this%n_jvalues
      do it = 1, this%n_sza
        write(tabled_output_unit, *) this%jvalues_label(ij)%string, ",", this%sza(it), ",", this%jvalues(it, ij)
      end do
    end do

    close (tabled_output_unit)
    call free_unit(tabled_output_unit)

  end subroutine output_photolysis_table

  subroutine set_from_tuv(this)
    class(photolysis_map_t), intent(inout) :: this

    character(len=:), allocatable :: temp_label
    integer(kind=i_kind) :: ij

    this%selected_photolysis = TUV_PHOTOLYSIS

    this%n_jvalues = nj
    this%n_sza = nt

    allocate (this%sza(this%n_sza))

    if (.not. allocated(this%jvalues_label)) then
      allocate (this%jvalues_label(this%n_jvalues))
      allocate (this%jvalues(this%n_sza, this%n_jvalues))
    end if

    this%sza = sza

    do ij = 1, this%n_jvalues
      temp_label = trim(jlabel(ij))
      this%jvalues_label(ij) = string_t(temp_label)
      deallocate (temp_label)
    end do

    this%jvalues = tuv_j

  end subroutine set_from_tuv

  subroutine set_from_cloudj(this, altitude)
    class(photolysis_map_t), intent(inout) :: this
    integer(kind=i_kind) :: i_photo_rxn, i_jvalues, i_sza
    character(len=:), allocatable :: temp_label
    real(kind=dp) :: altitude ! altitude in m

    this%n_sza = 11
    this%n_jvalues = cloudj_interface%NJX
    allocate (this%sza(this%n_sza))

    allocate (this%jvalues_label(this%n_jvalues))
    allocate (this%jvalues(this%n_sza, this%n_jvalues))

    call assert_msg(440991927, &
      this%nphot_reac <= JVN_, &
      "too many photolysis reactions for cloudJ, increase JVN_ in boxmodel/src/cloud-j/cldj_cmn_mod.F90")

    do i_jvalues = 1, this%n_jvalues
      temp_label = trim(cloudj_interface%cloudj_labels(i_jvalues))
      this%jvalues_label(i_jvalues) = string_t(temp_label)
      deallocate (temp_label)
    end do

    do i_sza = 1, this%n_sza
      this%sza(i_sza) = (i_sza - 1)*10 ! do sza from 0 to 100°

      call cloudj_interface%run_cloudj(this%sza(i_sza))

      !! interpolate to our altitude
      do i_jvalues = 1, this%n_jvalues

        this%jvalues(i_sza, i_jvalues) = &
          interp_1d( &
          cloudj_interface%zzz(1:size(cloudj_interface%valjxx, 1)), &
          cloudj_interface%valjxx(:, i_jvalues), &
          altitude*100.0) ! convert altitude to cm because zzz is in cm !
      end do
    end do

  end subroutine set_from_cloudj

  !> set map between mechanism reactions and calculated photolysis j-values
  subroutine map_reactions(this)
    class(photolysis_map_t), intent(inout) :: this

    integer(kind=i_kind)  :: ilabel, ij, k
    logical :: found

    ! now that jlabel and photolysis_label is populated, we check that we have everything needed
    ! and populate the index_map array
    allocate (this%index_map(this%nphot_reac, this%max_assigned_j))
    this%index_map = -1
    do ilabel = 1, this%nphot_reac
      do k = 1, this%n_assigned_j(ilabel)
        found = .false.
        do ij = 1, this%n_jvalues
          if (trim(this%photolysis_label(ilabel, k)%string) == trim(this%jvalues_label(ij)%string)) then
            found = .true.
            this%index_map(ilabel, k) = ij
            exit
          end if
        end do
        call assert_msg(204233309, found, &
          "reference photolysis reaction with label '"//trim(this%photolysis_label(ilabel, k)%string)//"' not found.")
      end do
    end do

  end subroutine map_reactions

  ! interpolate jvalues for a specific reaction and zenith angle
  real(kind=dp) function get_j(this, sza, ij)
    class(photolysis_map_t), intent(in) :: this
    real(kind=dp), intent(in) :: sza
    integer(kind=i_kind), intent(in) :: ij
    integer(kind=i_kind) :: i

    get_j = interp_1d(this%sza(1:this%n_sza), this%jvalues(1:this%n_sza, ij), sza)

  end function get_j

  subroutine update_photolysis_reactions_rates(this, sza)
    class(photolysis_map_t), intent(inout) :: this
    real(kind=dp), dimension(:), intent(in) :: sza

    integer(kind=i_kind) :: ij, ireac, k, i_cell, n_cells

    real(kind=dp), dimension(:, :), allocatable :: jvalue

    real(kind=dp) :: rate

    n_cells = size(sza, 1)

    allocate (jvalue(n_cells, this%n_jvalues))

    jvalue = 0.0
    do i_cell = 1, n_cells
      do ij = 1, this%n_jvalues
        jvalue(i_cell, ij) = this%get_j(sza(i_cell), ij)
        call assert_msg(36239543, jvalue(i_cell, ij) >= 0.0, "invalid j_value for reaction "//trim(this%jvalues_label(ij)%string))
      end do

      do ireac = 1, this%nphot_reac
        ! do the weighted sum of assigned j_values
        rate = 0.0
        do k = 1, this%n_assigned_j(ireac)
          rate = rate + this%weighting_factors(ireac, k)*&
            jvalue(i_cell, this%index_map(ireac, k))
        enddo

        call this%photo_rate_updates(i_cell, ireac)%set_rate(rate)

      end do
    enddo

  end subroutine update_photolysis_reactions_rates

  subroutine photolysis_update(this, latitude, longitude, altitude, pressure, surface_albedo, &
    o3_col, so2_col, no2_col, &
    cloud_optical_depth, cloud_base_alt, cloud_top_alt, &
    aer_optical_depth, aer_ssa, aer_alpha, &
    temperature, air_density, &
    time_control)
    class(photolysis_map_t), intent(in) :: this
    real(kind=sp), intent(in) :: latitude, longitude, altitude, pressure
    real(kind=sp), intent(in) :: surface_albedo, o3_col, so2_col, no2_col
    real(kind=sp), intent(in) :: cloud_optical_depth, cloud_base_alt, cloud_top_alt
    real(kind=sp), intent(in) :: aer_optical_depth, aer_ssa, aer_alpha
    real(kind=sp), intent(in) :: temperature, air_density
    type(time_control_t), intent(in) :: time_control

    !TODO: give possibility to adjust tolerances and other parameters
    if (abs(longitude - last_longitude) > 1.0 .or. & !deg
      abs(latitude - last_latitude) > 1.0 .or. & !deg
      abs(altitude - last_altitude) > 0.5) then !km
      select case (this%selected_photolysis)

       case (TUV_PHOTOLYSIS)
        call tuv_interface%tuv_init(latitude, longitude, altitude, pressure, surface_albedo, &
          o3_col, so2_col, no2_col, &
          cloud_optical_depth, cloud_base_alt, cloud_top_alt, &
          aer_optical_depth, aer_ssa, aer_alpha, &
          temperature, air_density, &
          time_control)
       case (CLOUDJ_PHOTOLYSIS)
        !TODO:
      end select
      last_longitude = longitude
      last_latitude = latitude
      last_altitude = altitude
    end if
  end subroutine photolysis_update

  subroutine print(this)
    class(photolysis_map_t), intent(inout) :: this

    integer(kind=i_kind) :: i, k

    print *, "********************"
    print *, "** PHOTOLYSIS MAP **"
    print *, "********************"

    print *, "  nphot_reac: ", this%nphot_reac
    print *, "  n_jvalues: ", this%n_jvalues
    print *, "  n_sza: ", this%n_sza

    print *, "  sza values: "
    print *, this%sza

    print *, "  jvalues_labels and jvalues:"
    do i = 1, this%n_jvalues
      print *, i, ": ", this%jvalues_label(i)%string
      print *, this%jvalues(:, i)
    end do

    print *, "  photolysis reactions and associated jvalue index"
    do i = 1, this%nphot_reac
      do k = 1, this%n_assigned_j(i)
        print *, i, ": ", this%photolysis_label(i, k)%string
        print *, this%weighting_factors(i, k), "*", this%index_map(i, k)
      end do
    end do

  end subroutine print

  !> Determine the number of bytes required to pack the variable
  integer(kind=i_kind) function pack_size(this, comm)
    class(photolysis_map_t), intent(in) :: this
    !> MPI communicator
    integer, intent(in), optional :: comm

#ifdef CAMP_USE_MPI
    integer(kind=i_kind) :: l_comm, i_reac, k
    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if
    pack_size = 0

    pack_size = pack_size + camp_mpi_pack_size_integer(this%selected_photolysis, l_comm) + &
      camp_mpi_pack_size_integer(this%nphot_reac, l_comm) + &
      camp_mpi_pack_size_integer(this%n_sza, l_comm) + &
      camp_mpi_pack_size_integer(this%n_jvalues, l_comm) + &
      camp_mpi_pack_size_real_array(this%sza, l_comm) + &
      camp_mpi_pack_size_real_array_2d(this%jvalues, l_comm) + &
      camp_mpi_pack_size_integer_array(this%n_assigned_j, l_comm) + &
      camp_mpi_pack_size_integer_array_2d(this%index_map, l_comm) + &
      camp_mpi_pack_size_real_array_2d(this%weighting_factors, l_comm) + &
      camp_mpi_pack_size_integer(this%max_assigned_j, l_comm)
    ! TODO: account for n_cells to pack this
    ! do i_reac = 1, this%nphot_reac
    !   pack_size = pack_size + this%photo_rate_updates(i_reac)%pack_size(l_comm)
    !   do k = 1, this%n_assigned_j(i_reac)
    !     pack_size = pack_size + camp_mpi_pack_size_string(this%photolysis_label(i_reac, k)%string)
    !   end do
    ! end do

#else
    pack_size = 0
#endif
  end function pack_size

  !> Pack the given variable into a buffer, advancing position
  subroutine bin_pack(this, buffer, pos, comm)
    class(photolysis_map_t), intent(in) :: this
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos
    !> MPI communicator
    integer, intent(in), optional :: comm

#ifdef CAMP_USE_MPI
    integer(kind=i_kind) :: prev_position, l_comm, i_reac, k

    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if

    prev_position = pos
    call camp_mpi_pack_integer(buffer, pos, this%selected_photolysis, l_comm)
    call camp_mpi_pack_integer(buffer, pos, this%nphot_reac, l_comm)
    call camp_mpi_pack_integer(buffer, pos, this%n_sza, l_comm)
    call camp_mpi_pack_integer(buffer, pos, this%n_jvalues, l_comm)
    call camp_mpi_pack_real_array(buffer, pos, this%sza, l_comm)
    call camp_mpi_pack_real_array_2d(buffer, pos, this%jvalues, l_comm)
    call camp_mpi_pack_integer_array(buffer, pos, this%n_assigned_j, l_comm)
    call camp_mpi_pack_integer_array_2d(buffer, pos, this%index_map, l_comm)
    call camp_mpi_pack_real_array_2d(buffer, pos, this%weighting_factors, l_comm)
    call camp_mpi_pack_integer(buffer, pos, this%max_assigned_j, l_comm)

    ! TODO: account for n_cells to pack this
    ! do i_reac = 1, this%nphot_reac
    !   call this%photo_rate_updates(i_reac)%bin_pack(buffer, pos, l_comm)
    !   do k = 1, this%n_assigned_j(i_reac)
    !     call camp_mpi_pack_string(buffer, pos, this%photolysis_label(i_reac, k)%string, l_comm)
    !   end do
    ! end do

    call assert(109979676, &
      pos - prev_position == this%pack_size(l_comm))
#endif

  end subroutine bin_pack

  !> Unpack the given variable from a buffer, advancing position
  subroutine bin_unpack(this, buffer, pos, comm)
    class(photolysis_map_t), intent(inout) :: this
    !> Memory buffer
    character, intent(inout) :: buffer(:)
    !> Current buffer position
    integer, intent(inout) :: pos
    !> MPI communicator
    integer, intent(in), optional :: comm

#ifdef CAMP_USE_MPI
    integer(kind=i_kind) :: prev_position, l_comm, i_reac, k
    character(len=75) :: temp_string

    if (present(comm)) then
      l_comm = comm
    else
      l_comm = MPI_COMM_WORLD
    end if

    prev_position = pos

    call camp_mpi_unpack_integer(buffer, pos, this%selected_photolysis, l_comm)
    call camp_mpi_unpack_integer(buffer, pos, this%nphot_reac, l_comm)
    call camp_mpi_unpack_integer(buffer, pos, this%n_sza, l_comm)
    call camp_mpi_unpack_integer(buffer, pos, this%n_jvalues, l_comm)
    call camp_mpi_unpack_real_array(buffer, pos, this%sza, l_comm)
    call camp_mpi_unpack_real_array_2d(buffer, pos, this%jvalues, l_comm)
    call camp_mpi_unpack_integer_array(buffer, pos, this%n_assigned_j, l_comm)
    call camp_mpi_unpack_integer_array_2d(buffer, pos, this%index_map, l_comm)
    call camp_mpi_unpack_real_array_2d(buffer, pos, this%weighting_factors, l_comm)
    call camp_mpi_unpack_integer(buffer, pos, this%max_assigned_j, l_comm)
    ! TODO: account for n_cells to unpack this
    ! allocate (this%photo_rate_updates(this%nphot_reac))
    ! allocate(this%photolysis_label(this%nphot_reac, this%max_assigned_j))
    ! do i_reac = 1, this%nphot_reac
    !   call this%photo_rate_updates(i_reac)%bin_unpack(buffer, pos, l_comm)
    !   do k = 1, this%n_assigned_j(i_reac)
    !     call camp_mpi_unpack_string(buffer, pos, temp_string, l_comm)
    !     this%photolysis_label(i_reac, k)%string = trim(temp_string)
    !   end do
    ! end do

    call assert(529867398, &
      pos - prev_position == this%pack_size(l_comm))
#endif

  end subroutine bin_unpack

end module boxmodel_photolysis
