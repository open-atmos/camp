!! \todo: in tuv subroutines, replace all INCLUDE 'params' by a use module boxmodel_tuv_module
module boxmodel_tuv_module
  use tuv_params
  use camp_constants, only: i_kind, sp, dp, const
  use boxmodel_time_control, only: time_control_t
  use camp_util, only: string_t, get_unit, free_unit, assert_msg, to_string, interp_1d, warn_msg
  use boxmodel_log
  implicit none

  public :: tuv_interface_t, nj, jlabel, nt, sza, tuv_j
  private

  type tuv_interface_t

  contains
    procedure :: tuv_init

  end type tuv_interface_t


  integer(kind=i_kind) :: nstr
  real(kind=sp)        :: lat, lon, tmzone
  integer(kind=i_kind) :: iyear, imonth, iday
  real(kind=sp)        :: zstart, zstop, wstart, wstop, tstart, tstop
  integer(kind=i_kind) :: nz, nzint, nt
  logical              :: lzenit
  real(kind=sp)        :: alsurf, psurf, o3col, so2col, no2col
  real(kind=sp)        :: taucld, zbase, ztop, tauaer, ssaaer, alpha
  real(kind=sp)        :: dirsun, difdn, difup, zout, zaird, ztemp
  logical              :: lirrad, laflux, lrates, ljvals, lmmech

  integer(kind=i_kind) :: isfix, ijfix, itfix, izfix, iwfix
  integer(kind=i_kind) :: nms, nmj
  integer(kind=i_kind), dimension(:), allocatable :: ims, imj

  ! wavelength grid
  integer(kind=i_kind) :: nw, iw, nwint
  real(kind=sp), dimension(:), allocatable  :: wl, wc, wu

  ! Altitude grid
  integer(kind=i_kind) :: nzm1, iz, izout
  real(kind=sp), dimension(:), allocatable ::  z

  ! Time and/or solar zenith angle
  integer(kind=i_kind) :: it
  real(kind=sp), dimension(:), allocatable :: t

  ! Solar zenith angle and azimuth
  ! slant pathlengths in spherical geometry

  real(kind=sp), dimension(:), allocatable :: sza
  real(kind=sp)        :: zen, sznoon
  integer(kind=i_kind), dimension(:), allocatable ::  nid
  real(kind=sp), dimension(:, :), allocatable :: dsdh(:, :)

  ! Extra terrestrial solar flux and earth-Sun distance ^-2
  real(kind=sp), dimension(:), allocatable :: f, etf, esfact

  ! ozone absorption cross section
  integer              :: mabs
  real(kind=sp), dimension(:, :), allocatable :: o3xs

  ! o2 absorption cross section
  real(kind=sp), dimension(:, :), allocatable :: o2xs
  real(kind=sp), dimension(:), allocatable    :: o2xs1

  ! so2 absoption cross section
  real(kind=sp), dimension(:), allocatable    :: so2xs

  ! no2 absorption cross section
  real(kind=sp), dimension(:, :), allocatable :: no2xs

  ! atmospheric optical parameters
  real(kind=sp), dimension(:), allocatable       :: tlev, tlay
  real(kind=sp), dimension(:), allocatable       :: aircon, aircol, vcol, scol
  real(kind=sp), dimension(:, :), allocatable    :: dtrl
  real(kind=sp), dimension(:), allocatable       :: co3
  real(kind=sp), dimension(:, :), allocatable    :: dto3, dto2, dtso2, dtno2
  real(kind=sp), dimension(:, :), allocatable    :: dtcld, omcld, gcld
  real(kind=sp), dimension(:, :), allocatable    :: dtaer, omaer, gaer
  real(kind=sp), dimension(:, :), allocatable    :: dtsnw, omsnw, gsnw
  real(kind=sp), dimension(:), allocatable       :: albedo
  real(kind=sp), dimension(:, :), allocatable    :: dt_any, om_any, g_any

  ! Spectral irradiance and actinic flux (scalar irradiance)
  real(kind=sp), dimension(:), allocatable       :: edir, edn, eup
  real(kind=sp), dimension(:, :), allocatable    :: sirrad
  real(kind=sp), dimension(:), allocatable       :: fdir, fdn, fup
  real(kind=sp), dimension(:, :), allocatable    :: saflux

  ! photolysis coefficients (j-values)
  integer(kind=i_kind) :: nj, ij
  real(kind=sp), dimension(:, :, :), allocatable :: sj
  real(kind=sp), dimension(:, :), allocatable    :: valj
  !> j-value at chosen altitude as a function of sza
  real(kind=sp), dimension(:, :), allocatable    :: tuv_j
  real(kind=sp)                                  :: djdw
  integer(kind=i_kind), dimension(:), allocatable:: tpflag

  ! re-scaling factors
  ! Total columns of O3, SO2, NO2 (Dobson Units)
  real(kind=sp)                                  :: o3_tc, so2_tc, no2_tc
  ! planetary boundary layer height and pollutant concentrations
  integer(kind=i_kind) :: ipbl
  real(kind=sp) :: zpbl, o3pbl, so2pbl, no2pbl, aod330

  character(len=50), dimension(:), allocatable :: jlabel

  logical :: allocated = .false.

contains

  !> allocate arrays for tuv
  subroutine tuv_allocate()

    allocate (z(kz))
    z = 0.0

    allocate (t(kt), sza(kt), esfact(kt))
    t = 0.0
    sza = 0.0
    esfact = 0.0

    allocate (wl(kw), wc(kw), wu(kw))
    wl = 0.0
    wc = 0.0
    wu = 0.0

    allocate (tlev(kz), tlay(kz))
    tlev = 0.0
    tlay = 0.0

    allocate (aircon(kz), aircol(kz))
    aircon = 0.0
    aircol = 0.0

    allocate (co3(kz))
    co3 = 0.0

    allocate (f(kw))
    f = 0.0

    allocate (o2xs1(kw))
    allocate (o3xs(kz, kw))
    allocate (so2xs(kw))
    allocate (no2xs(kz, kw))
    o2xs1 = 0.0
    o3xs = 0.0
    so2xs = 0.0
    no2xs = 0.0

    allocate (sj(kj, kz, kw))
    sj = 0.0
    allocate (jlabel(kj), tpflag(kj))
    jlabel = ""
    tpflag = 0

    allocate (dtrl(kz, kw))
    dtrl = 0.0

    allocate (dto2(kz, kw))
    dto2 = 0.0

    allocate (dto3(kz, kw))
    dto3 = 0.0

    allocate (dtso2(kz, kw))
    dtso2 = 0.0

    allocate (dtno2(kz, kw))
    dtno2 = 0.0

    allocate (dtcld(kz, kw), omcld(kz, kw), gcld(kz, kw))
    dtcld = 0.0
    omcld = 0.0
    gcld = 0.0

    allocate (dtaer(kz, kw), omaer(kz, kw), gaer(kz, kw))
    dtaer = 0.0
    omaer = 0.0
    gaer = 0.0

    allocate (dtsnw(kz, kw), omsnw(kz, kw), gsnw(kz, kw))
    dtsnw = 0.0
    omsnw = 0.0
    gsnw = 0.0

    allocate (albedo(kw))
    albedo = 0.0

    allocate (dt_any(kz, kw), om_any(kz, kw), g_any(kz, kw))
    dt_any = 0.0
    om_any = 0.0
    g_any = 0.0

    allocate (etf(kw))
    etf = 0.0

    allocate (dsdh(kz + 1, kz), nid(kz + 1))
    dsdh = 0.0
    nid = 0.0

    allocate (vcol(kz), scol(kz))
    vcol = 0.0
    scol = 0.0

    allocate (o2xs(kz, kw))
    o2xs = 0.0

    allocate (valj(kj, kz), tuv_j(kt, kj))
    valj = 0.0
    tuv_j = 0.0

    allocate (edir(kz), edn(kz), eup(kz), &
      fdir(kz), fdn(kz), fup(kz))
    edir = 0.0
    edn = 0.0
    eup = 0.0
    fdir = 0.0
    fdn = 0.0
    fup = 0.0

    allocate (sirrad(kz, kw), saflux(kz, kw))
    sirrad = 0.0
    saflux = 0.0

    allocated = .true.

  end subroutine tuv_allocate

  !> fill tuv_j array by running tuv computations of J-values orç
  !! reading a precomputed table
  subroutine tuv_init(this, latitude, longitude, altitude, pressure, surface_albedo, &
    o3_col, so2_col, no2_col, &
    cloud_optical_depth, cloud_base_alt, cloud_top_alt, &
    aer_optical_depth, aer_ssa, aer_alpha, &
    temperature, air_density, &
    time_control, &
    table_path)
    class(tuv_interface_t), intent(inout) :: this
    real(kind=sp), intent(in) :: latitude, longitude, altitude, pressure
    real(kind=sp), intent(in) :: surface_albedo, o3_col, so2_col, no2_col
    real(kind=sp), intent(in) :: cloud_optical_depth, cloud_base_alt, cloud_top_alt
    real(kind=sp), intent(in) :: aer_optical_depth, aer_ssa, aer_alpha
    real(kind=sp), intent(in) :: temperature, air_density
    type(time_control_t), intent(in) :: time_control

    character(len=*), intent(in), optional :: table_path

    integer(kind=i_kind) :: ij, ilabel
    logical              :: found

    if (present(table_path)) then
      call read_tuv_table(latitude, longitude, altitude, table_path)

    else
      call tuv_run(latitude, longitude, altitude, pressure, surface_albedo, &
        o3_col, so2_col, no2_col, &
        cloud_optical_depth, cloud_base_alt, cloud_top_alt, &
        aer_optical_depth, aer_ssa, aer_alpha, &
        temperature, air_density, &
        time_control)
    end if

  end subroutine tuv_init

  subroutine tuv_run(latitude, longitude, altitude, pressure, surface_albedo, &
    o3_col, so2_col, no2_col, &
    cloud_optical_depth, cloud_base_alt, cloud_top_alt, &
    aer_optical_depth, aer_ssa, aer_alpha, &
    temperature, air_density, &
    time_control)
    real(kind=sp), intent(in) :: latitude, longitude, altitude, pressure
    real(kind=sp), intent(in) :: surface_albedo, o3_col, so2_col, no2_col
    real(kind=sp), intent(in) :: cloud_optical_depth, cloud_base_alt, cloud_top_alt
    real(kind=sp), intent(in) :: aer_optical_depth, aer_ssa, aer_alpha
    real(kind=sp), intent(in) :: temperature, air_density
    type(time_control_t), intent(in) :: time_control

    integer(kind=i_kind) :: tabled_output_unit
    character(len=:), allocatable :: tabled_output_filename

    if (.not. allocated) then
      call tuv_allocate()
    end if

    !TODO: decide with parameters to expose to the user
    !TODO: find good default values for harcoded values
    ! hardcoded inputs
    ! Radiative transfer scheme:
    !   nstr = number of streams
    !          If nstr < 2, will use 2-stream Delta Eddington
    !          If nstr > 1, will use nstr-stream discrete ordinates
    nstr = -2 ! select 2-stream delta eddington
    !   lzenit = switch for solar zenith angle (sza) grid rather than time
    lzenit = .true.
    !  Time of day grid:
    !    tstart = starting time, local hours
    !    tstop = stopping time, local hours
    !    nt = number of time steps
    !    lzenit = switch for solar zenith angle (sza) grid rather than time
    !              grid. If lzenit = .TRUE. then
    !                 tstart = first sza in deg.,
    !                 tstop = last sza in deg.,
    !                 nt = number of sza steps.
    tstart = 0.0
    tstop = 100.0
    nt = 11
    ! Vertical grid:
    ! zstart = surface elevation above sea level, km
    ! zstop = top of the atmosphere (exospheric), km
    ! nz = number of vertical levels, equally spaced
    !      (nz will increase by +1 if zout does not match altitude grid)
    zstart = 0.0
    zstop = 80.0
    nz = 81
    ! Wavelength grid:
    !   wstart = starting wavelength, nm
    !   wstop  = final wavelength, nm
    !   nwint = number of wavelength intervals, equally spaced
    !           if nwint < 0, the standard atmospheric wavelength grid, not
    !           equally spaced, from 120 to 735 nm, will be used. In this
    !           case, wstart and wstop values are ignored.
    ! nwint = -7 : fast-TUV, troposheric wavelengths only, nw = 8
    wstart = 280.0
    wstop = 420.0
    nwint = -7

    ! Directional components of radiation, weighting factors:
    !   dirsun = direct sun
    !   difdn = down-welling diffuse
    !   difup = up-welling diffuse
    !        e.g. use:
    !        dirsun = difdn = 1.0, difup = 0 for total down-welling irradiance
    !        dirsun = difdn = difup = 1.0 for actinic flux from all directions
    !        dirsun = difdn = 1.0, difup = -1 for net irradiance
    dirsun = 1.0
    difdn = 1.0
    difup = 1.0

    !-----------------------------
    ! user selected inputs
    ! Location (geographic):
    !   lat = LATITUDE (degrees, North = positive)
    !   lon = LONGITUDE (degrees, East = positive)
    !   tmzone = Local time zone difference (hrs) from Universal Time (ut):
    !            ut = timloc - tmzone
    lat = latitude
    lon = longitude
    tmzone = time_control%tz
    ! date:
    !    iyear = year (1950 to 2050)
    !    imonth = month (1 to 12)
    !    iday = day of month
    iyear = time_control%year
    imonth = time_control%month
    iday = time_control%day

    ! Surface condition:
    !   alsurf = surface albedo, wavelength independent
    !   psurf = surface pressure, mbar.  Set to negative value to use
    !           US Standard Atmosphere, 1976 (USSA76)
    alsurf = surface_albedo
    psurf = pressure*1e-2 ! pressure is given in Pa

    ! column amounts of absorbers in Dobson units, from surface to space
    !          Vertical profile for O3 from USSA76.  For SO2 and NO2, vertical
    !          concentration profile is 2.69e10 molec cm-3 between 0 and
    !          1 km above sea level, very small residual (10/largest) above 1 km.
    o3_tc = o3_col ! ozone (O3)
    so2_tc = so2_col ! sulfur dioxide (SO2)
    no2_tc = no2_col ! nitrogen dioxide (NO2)

    ! Cloud, assumed horizontally uniform, total coverage, single scattering
    !         albedo = 0.9999, asymmetry factor = 0.85, indep. of wavelength,
    !         and also uniform vertically between zbase and ztop:
    taucld = cloud_optical_depth ! vertical optical depth, independent of wavelength
    zbase = cloud_base_alt ! altitude of base, km above sea level
    ztop = cloud_top_alt ! altitude of top, km above sea level

    ! Aerosols, assumed vertical provile typical of continental regions from
    !         Elterman (1968):
    !   tauaer = aerosol vertical optical depth at 550 nm, from surface to space.
    !           If negative, will default to Elterman's values (ca. 0.235
    !           at 550 nm).
    !   ssaaer = single scattering albedo of aerosols, wavelength-independent.
    !   alpha = Angstrom coefficient = exponent for wavelength dependence of
    !           tauaer, so that  tauaer1/tauaer2  = (w2/w1)**alpha.
    tauaer = aer_optical_depth
    ssaaer = aer_ssa
    alpha = aer_alpha

    ! Output altitude:
    !   zout = altitude, km, for desired output.
    !        If not within 1 m of altitude grid, an additional
    !        level will be inserted and nz will be increased by +1.
    !   zaird = air density (molec. cm-3) at zout.  Set to negative value for
    !        default USSA76 value interpolated to zout.
    !   ztemp = air temperature (K) at zout.  Set to negative value for
    !        default USSA76 value interpolated to zout.
    zout = altitude*1e-3
    zaird = air_density*const%avagadro*1e-6/const%air_molec_weight! convert kg/m3 to molec/cm3
    ztemp = temperature

    !___ SECTION 2: SET GRIDS _________________________________________________

! altitude (creates altitude grid, locates index for selected output, izout)

    call gridz(zstart, zstop, nz, z, zout, izout)

! time/zenith (creates time/zenith angle grid, starting at tstart)

    CALL gridt(lat, lon, tmzone, &
      iyear, imonth, iday, &
      lzenit, tstart, tstop, &
      nt, t, sza, sznoon, esfact)

! wavelength grid, user-set range and spacing.
! NOTE:  Wavelengths are in vacuum, and therefore independent of altitude.
! To use wavelengths in air, see options in subroutine gridw

    CALL gridw(wstart, wstop, nwint, &
      nw, wl, wc, wu)

!  ___ SECTION 3: SET UP VERTICAL PROFILES OF TEMPERATURE, AIR DENSITY, and OZONE

!**** Temperature vertical profile, Kelvin
!   can overwrite temperature at altitude z(izout)

    call vptmp(nz, z, tlev, tlay)
    IF (ztemp .GT. nzero) tlev(izout) = ztemp

!****Air density(molec cm - 3) vertical profile
!can overwrite air density at altitude z(izout)

    CALL vpair(psurf, nz, z, &
      aircon, aircol)
    IF (zaird .GT. nzero) aircon(izout) = zaird

!****
!! PBL pollutants will be added if zpbl > 0.
! CAUTIONS:
! 1. The top of the PBL, zpbl in km, should be on one of the z-grid altitudes.
! 2. Concentrations, column increments, and optical depths
!       will be overwritten between surface and zpbl.
! 3. Inserting PBL constituents may change their total column amount.
! 4. Above pbl, the following are used:
!       for O3:  USSA or other profile
!       for NO2 and SO2: set to zero.
!       for aerosols: Elterman
! Turning on pbl will affect subroutines:
! vpo3, setno2, setso2, and setaer. See there for details

    zpbl = -999.
!      zpbl = 3.

! locate z-index for top of pbl

    ipbl = 0
    IF (zpbl .GT. 0.) THEN
      DO iz = 1, nz - 1
        IF (z(iz + 1) .GT. z(1) + zpbl*1.00001) GO TO 19
      END DO
19    CONTINUE
      ipbl = iz - 1
!      write (*, *) 'top of PBL index, height (km) ', ipbl, z(ipbl)

! specify pbl concetrations, in parts per billion

      o3pbl = 100.
      so2pbl = 10.
      no2pbl = 50.

! PBL aerosol optical depth at 330 nm
! (to change ssa and g of pbl aerosols, go to subroutine setair.f)

      aod330 = 0.8

    END IF

! ozone vertical profile

    CALL vpo3(ipbl, zpbl, o3pbl, &
      o3_tc, nz, z, aircol, co3)

    ! ___ SECTION 4: READ SPECTRAL DATA ____________________________

    ! read (and grid) extra terrestrial flux data:

    call rdetfl(nw, wl, f)

    ! read cross section data for
    !    O2 (will overwrite at Lyman-alpha and SRB wavelengths
    !            see subroutine la_srb.f)
    !    O3 (temperature-dependent)
    !    SO2
    !    NO2

    nzm1 = nz - 1
    CALL rdo2xs(nw, wl, o2xs1)
    mabs = 1
    CALL rdo3xs(mabs, nzm1, tlay, nw, wl, o3xs)
    CALL rdso2xs(nw, wl, so2xs)
    CALL rdno2xs(nz, tlay, nw, wl, no2xs)

    !***** Spectral weighting functions
    ! (Some of these depend on temperature T and pressure P, and therefore
    !  on altitude z.  Therefore they are computed only after the T and P profiles
    !  are set above with subroutines settmp and setair.)
    ! Photo-physical   set in swphys.f (transmission functions)
    ! Photo-biological set in swbiol.f (action spectra)
    ! Photo-chemical   set in swchem.f (cross sections x quantum yields)* Physical
    !   and biological weigthing functions are assumed to depend
    !   only on wavelength.
    ! Chemical weighting functions (product of cross-section x quantum yield)
    !   for many photolysis reactions are known to depend on temperature
    !   and/or pressure, and therefore are functions of wavelength and altitude.
    ! Output:
    ! from swphys & swbiol:  sw(ks,kw) - for each weighting function slabel(ks)
    ! from swchem:  sj(kj,kz,kw) - for each reaction jlabel(kj)
    ! For swchem, need to know temperature and pressure profiles.

    ! CALL swphys(nw, wl, wc, ns, sw, slabel)
    ! CALL swbiol(nw, wl, wc, ns, sw, slabel)

    CALL swchem(nw, wl, nz, tlev, aircon, nj, sj, jlabel, tpflag)
    CALL swgecko(nw, wl, nz, tlev, aircon, nj, sj, jlabel)

    ! ___ SECTION 5: SET ATMOSPHERIC OPTICAL DEPTH INCREMENTS _____________________

    ! Rayleigh optical depth increments:

    CALL odrl(nz, z, nw, wl, aircol, dtrl)


    ! O2 vertical profile and O2 absorption optical depths
    !   For now, O2 densitiy assumed as 20.95% of air density, can be changed
    !   in subroutine.
    !   Optical depths in Lyman-alpha and SRB will be over-written
    !   in subroutine la_srb.f

    CALL seto2(nz, z, nw, wl, aircol, o2xs1, dto2)

    ! ozone optical depths

    call odo3(nz, z, nw, wl, o3xs, co3, dto3)

    ! SO2, vertical profile and optical depth

    call setso2(ipbl, zpbl, so2pbl, &
      so2_tc, nz, z, nw, wl, so2xs, &
      tlay, aircol, &
      dtso2)

    ! NO2 vertical profile and optical depths

    CALL setno2(ipbl, zpbl, no2pbl, &
      no2_tc, nz, z, nw, wl, no2xs, &
      tlay, aircol, &
      dtno2)

    ! Cloud vertical profile, optical depths, single scattering albedo, asymmetry factor

    CALL setcld(taucld, zbase, ztop, &
      nz, z, nw, wl, &
      dtcld, omcld, gcld)

    ! Aerosol vertical profile, optical depths, single scattering albedo, asymmetry factor

    CALL setaer(ipbl, zpbl, aod330, &
      tauaer, ssaaer, alpha, &
      nz, z, nw, wl, &
      dtaer, omaer, gaer)

    ! Snowpack physical and optical depths, single scattering albedo, asymmetry factor

    CALL setsnw( &
      nz, z, nw, wl, &
      dtsnw, omsnw, gsnw)

    ! Surface albedo

    CALL setalb(alsurf, nw, wl, &
      albedo)

    ! Set any additional absorber or scatterer:

    ! Must populate dt_any(kz,kw), om_any(kz,kw), g_any(kz,kw) manually
    ! This allows user to put in arbitrary absorber or scatterer
    ! could write a subroutine, e.g.:
    !      CALL setany(nz,z,nw,wl,aircol, dt_any,om_any, g_any)
    ! or write manually here.

    !      DO iz = 1, nz-1
    !         DO iw = 1, nw-1
    !            dt_any(iz,iw) = 0.79*aircol(iz) * 2.e-17 ! N2 VUV absorption
    !            dt_any(iz,iw) = 0.
    !            om_any(iz,iw) = 0.
    !            g_any(iz,iw) = 0.
    !         ENDDO
    !      ENDDO

    ! ___ SECTION 6: TIME/SZA LOOP  _____________________________________

    ! Loop over time or solar zenith angle (zen):
    do 20, it = 1, nt

      zen = sza(it)

      ! correction for earth-sun distance
      do iw = 1, nw - 1
        etf(iw) = f(iw)*esfact(it)
      end do

      ! ____ SECTION 7: CALCULATE ZENITH ANGLE-DEPENDENT QUANTITIES __________

      ! slant path lengths for spherical geometry
      call sphers(nz, z, zen, dsdh, nid)
      call airmas(nz, dsdh, nid, aircol, vcol, scol)

      ! Recalculate effective O2 optical depth and cross sections for Lyman-alpha
      ! and Schumann-Runge bands, must know zenith angle
      ! Then assign O2 cross section to sj(1,*,*)

      call la_srb(nz, z, tlev, nw, wl, vcol, scol, o2xs1, &
        dto2, o2xs)
      call sjo2(nz, nw, o2xs, 1, sj)

      ! ____ SECTION 8: WAVELENGTH LOOP ______________________________________
      ! initialize for wavelength integration

      call zero2(valj, kj, kz)

      !**** Main wavelength loop:

      do 10, iw = 1, nw - 1

        !* monochromatic radiative transfer. Outputs are:
        !  normalized irradiances     edir(iz), edn(iz), eup(iz)
        !  normalized actinic fluxes  fdir(iz), fdn(zi), fup(iz)
        !  where
        !  dir = direct beam, dn = down-welling diffuse, up = up-welling diffuse

        CALL rtlink(nstr, nz, &
          iw, albedo(iw), zen, &
          dsdh, nid, &
          dtrl, &
          dto3, &
          dto2, &
          dtso2, &
          dtno2, &
          dtcld, omcld, gcld, &
          dtaer, omaer, gaer, &
          dtsnw, omsnw, gsnw, &
          dt_any, om_any, g_any, &
          edir, edn, eup, fdir, fdn, fup)

        ! Spectral irradiance, W m-2 nm-1
        ! for downwelling only, use difup = 0.

        do iz = 1, nz
          sirrad(iz, iw) = etf(iw)* &
            (dirsun*edir(iz) + difdn*edn(iz) + difup*eup(iz))
        end do
        ! Spectral actinic flux, quanta s-1 nm-1 cm-2, all directions:
        !    units conversion:  1.e-4 * (wc*1e-9) / hc

        DO iz = 1, nz
          saflux(iz, iw) = etf(iw)*(1.e-13*wc(iw)/const%hc)* &
            (dirsun*fdir(iz) + difdn*fdn(iz) + difup*fup(iz))
        END DO

        !** Accumulate weighted integrals over wavelength, at all altitudes:

        do iz = 1, nz

! Photolysis rate coefficients (J-values) s-1
          do ij = 1, nj
            djdw = saflux(iz, iw)*sj(ij, iz, iw)
            valj(ij, iz) = valj(ij, iz) + djdw*(wu(iw) - wl(iw))
          end do
        end do

10    end do

      ! save valj for current sza
      do ij = 1, nj
        call assert_msg(48312890, valj(ij, izout) >= 0., "TUV computed an invalid photolysis rate for reaction "//trim(to_string(ij))//": "//trim(jlabel(ij)))
        tuv_j(it, ij) = valj(ij, izout)
      enddo

20  end do

  end subroutine tuv_run

  subroutine read_tuv_table(latitude, longitude, altitude, table_path)
    real(kind=sp), intent(in) :: latitude, longitude, altitude
    character(len=*), intent(in) :: table_path

    integer(kind=i_kind) :: tabled_output_unit

    tabled_output_unit = get_unit()
    open (unit=tabled_output_unit, file=table_path, status="old")

    ! get dimensions and allocate
    read (tabled_output_unit, *)
    read (tabled_output_unit, *) nt

    read (tabled_output_unit, *)
    read (tabled_output_unit, *) nj

    allocate (sza(nt))
    allocate (jlabel(nj))
    allocate (tuv_j(nt, nj))

    ! check that geographical position is not too far
    read (tabled_output_unit, *)
    read (tabled_output_unit, *) lon

    read (tabled_output_unit, *)
    read (tabled_output_unit, *) lat

    read (tabled_output_unit, *)
    read (tabled_output_unit, *) zout

    if (abs(lon - longitude) > 0.1 &
      .or. abs(lat - latitude) > 0.1 &
      .or. abs(zout - altitude*1e-3) > 0.5) then
      call warn_msg(302080103, "the position of the model is quite different from the precomputed tuv table")
    end if

    read (tabled_output_unit, *)
    read (tabled_output_unit, *) sza(1:nt)

    read (tabled_output_unit, *)
    read (tabled_output_unit, *) jlabel(1:nj)

    read (tabled_output_unit, *)
    read (tabled_output_unit, *) tuv_j(1:nt, 1:nj)

    close (tabled_output_unit)
    call free_unit(tabled_output_unit)

  end subroutine read_tuv_table

  real(kind=dp) function tuv_getj(this, index, zen_angle)
    class(tuv_interface_t), intent(in) :: this
    integer(kind=i_kind), intent(in) :: index
    real(kind=dp), intent(in) :: zen_angle

    call assert_msg(967274880, index > 0 .and. index < size(tuv_j, 2), &
      "ij is out of bound: "//to_string(index))

    tuv_getj = interp_1d(real(sza, kind=dp), real(tuv_j(:, index), kind=dp), zen_angle)
  end function tuv_getj

end module boxmodel_tuv_module
