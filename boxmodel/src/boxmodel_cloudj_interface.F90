module boxmodel_cloudj_interface
  use CLDJ_CMN_MOD
  use CLDJ_INIT_MOD
  use FJX_SUB_MOD
  use CLD_SUB_MOD, only: CLOUD_JX
  use OSA_SUB_MOD

  use camp_util, only: i_kind, sp, dp, open_file_read, assert_msg

  implicit none

  type :: cloudj_interface_t
    logical                    :: LPRTJ, LDARK
    integer(kind=i_kind)       :: IRAN
    integer(kind=i_kind)       :: JVNU, ANU, L1U
    integer(kind=i_kind)       :: NICA, JCOUNT
    real(kind=dp)                     :: U0, SZA, SOLF
    real(kind=dp), dimension(L2_)  :: PPP, ZZZ
    real(kind=dp), dimension(L1_)  :: TTT, HHH, DDD, RRR, OOO, CCC
    real(kind=dp), dimension(L1_)  :: O3, CH4, H2O, OD18
    real(kind=dp), dimension(S_ + 2, L1_):: SKPERD
    real(kind=dp), dimension(6)       :: SWMSQ
    real(kind=dp), dimension(L2_)    :: CLF, LWP, IWP, REFFL, REFFI
    integer(kind=i_kind), dimension(L2_)    :: CLDIW
    real(kind=dp), dimension(L2_, AN_):: AERSP
    integer(kind=i_kind), dimension(L2_, AN_):: NDXAER
    real(kind=dp), dimension(L_, JVN_) :: VALJXX
    real(kind=dp), dimension(5, S_)    :: RFL
    real(kind=dp), dimension(NQD_)    :: WTQCA
    real(kind=dp), dimension(LWEPAR) :: CLDFRW
    integer(kind=i_kind)             :: LTOP

    integer(kind=i_kind)              :: NJX ! number of photolysis reactions

    character(len=6), allocatable, dimension(:) :: cloudj_labels

    integer(kind=i_kind) :: MONTH
    real(kind=dp)        :: YLAT
    ! ALBEDO(1:4) : Default albedos 4 quad angles = 86,71,48 & 22 deg ZA
    ! ALBEDO(5)   : Default albedo for SZA incident ray
    real(kind=dp), dimension(5) :: ALBEDO
    real(kind=dp)   :: WIND, CHLR
  contains
    procedure :: init

    procedure, public :: run_cloudj

  end type cloudj_interface_t

contains
  !> initialize cloud-j interface
  subroutine init(this, atmosphere_profile_filename, month, latitude, wind, chlr, &
                  temperature, rh, psurf, cldoption)
    !> the cloud-j interface object
    class(cloudj_interface_t), intent(out) :: this

    character(len=*), intent(in)           :: atmosphere_profile_filename
    !> current month for climatology
    integer(kind=i_kind), intent(in)       :: month
    !> input latitude
    real(kind=dp), intent(in)              :: latitude
    !> surface wind (m/s)
    real(kind=dp), intent(in)              :: wind
    !> surface chlorophyle (mg/m3)
    real(kind=dp), intent(in)              :: chlr
    !> temperature (K)
    real(kind=dp), intent(in)              :: temperature
    !> relative humidity (dimensionless 0-1)
    real(kind=dp), intent(in)              :: rh
    !> surface pressure (hPa)
    real(kind=dp), intent(in)              :: psurf
    !>       cldoption = 1  :  Clear sky J's
      !!       cldoption = 2  :  Averaged cloud cover
      !!       cldoption = 3  :  cloud-fract**3/2, then average cloud cover
      !!       cldoption = 4  :  ****not used
      !!       cldoption = 5  :  Random select NRANDO ICA's from all(Independent Column Atmos.)
      !!       cldoption = 6  :  Use all (up to 4) quadrature cloud cover QCAs (mid-pts of bin)
      !!       cldoption = 7  :  Use all (up to 4) QCAs (average clouds within each Q-bin)
      !!       cldoption = 8  :  Calculate J's for ALL ICAs (up to 20,000 per cell!)
    integer(kind=i_kind), intent(in)      :: cldoption

    ! local
    real(kind=dp), dimension(JVN_)  :: ETAA, ETAB, TI, RI, ZOFL, AER1, AER2
    integer(kind=i_kind), dimension(L2_) :: NAA1, NAA2
    real*8, dimension(L_)  :: WLC, WIC
    real(kind=dp), dimension(LWEPAR) :: CLDIWCW, CLDLWCW
    real(kind=dp)  :: SCALEH, CF, PMID, PDEL, ZDEL, ICWC, F1

    character*6, dimension(JVN_)      :: TITLEJXX

    integer(kind=i_kind) :: header, i_line, L, J

    integer(kind=i_kind) :: INPUT_FILE_UNIT, err
    character(len=1000) :: buffer
    character(len=:), allocatable :: current_line

    this%ANU = AN_
    this%JVNU = JVN_
    this%L1U = L1_

    ! read in & store all fast-JX data
    call INIT_CLDJ(TITLEJXX, this%JVNU, this%NJX)

    ! store reaction labels that will be used to match with chemical mechanism
    allocate (this%cloudj_labels(this%NJX))
    this%cloudj_labels(:) = TITLEJXX(1:this%NJX)

    this%MONTH = month
    this%YLAT = latitude
    ! default albedo values
    ! \todo make that adjustable by user
    this%ALBEDO = (/0.05, 0.05, 0.05, 0.05, 0.05/)

    this%WIND = wind
    this%CHLR = chlr

    call assert_msg(379355232, &
                    cldoption > 0 .and. cldoption < 9 .and. cldoption /= 4, &
                    "Invalid cldoption used for initializing cloudj")
    CLDFLAG = cldoption

    ! get atmosphere profile
    call open_file_read(atmosphere_profile_filename, INPUT_FILE_UNIT)

    header = 0
    do
      read (INPUT_FILE_UNIT, '(A)', iostat=err) buffer
      call assert_msg(618826182, &
                      err == 0, &
                      "Error reading '"//atmosphere_profile_filename//"'")
      if (buffer(1:1) == "!") then
        header = header + 1
      else
        exit
      end if
    end do

    !print *, "header:", header

    rewind (INPUT_FILE_UNIT)

    do i_line = 1, header
      read (INPUT_FILE_UNIT, *)
    end do

    ! read atmosphere info
    do L = 1, LPAR + 1
      read (INPUT_FILE_UNIT, '(i3,1x,2f11.7,2x,f5.1,f5.2,f11.2,2(f7.3,i4))') &
        J, ETAA(L), ETAB(L), TI(L), RI(L), ZOFL(L) &
        , AER1(L), NAA1(L), AER2(L), NAA2(L)
    end do
    read (INPUT_FILE_UNIT, *)
    ! read cloud info
    do L = LWEPAR, 1, -1
      read (INPUT_FILE_UNIT, '(i3,1p,e14.5,28x,2e14.5)') &
        J, this%CLDFRW(L), CLDLWCW(L), CLDIWCW(L)
    end do

    close (INPUT_FILE_UNIT)

    ! override first level temperature and rh
    TI(1) = temperature
    RI(1) = rh

    ETAA(L2_) = 0.0
    ETAB(L2_) = 0.0

    do L = 1, L2_
      this%PPP(L) = ETAA(L) + ETAB(L)*PSURF
    end do

!---sets climatologies for O3, T, D & Z
    call ACLIM_FJX(this%YLAT, this%MONTH, this%PPP, this%TTT, this%O3, this%CH4, L1_)

    ! do L = 1, L1_
    !    print *, this%TTT(L), this%O3(L), this%CH4(L)
    ! enddo
    do L = 1, L_
      this%TTT(L) = TI(L) ! override climatology T's and RH's
      this%RRR(L) = RI(L)
    end do

    this%ZZZ(1) = 16.d5*log10(1013.25d0/this%PPP(1))        ! zzz in cm
    do L = 1, L_
      this%DDD(L) = (this%PPP(L) - this%PPP(L + 1))*MASFAC
         !!! geopotential since assumes g = constant
      SCALEH = 1.3806d-19*MASFAC*this%TTT(L)
      this%ZZZ(L + 1) = this%ZZZ(L) - (log(this%PPP(L + 1)/this%PPP(L))*SCALEH)
      this%OOO(L) = this%DDD(L)*this%O3(L)*1.d-6
      this%CCC(L) = this%DDD(L)*this%CH4(L)*1.d-9
    end do
    L = L_ + 1
    this%ZZZ(L + 1) = this%ZZZ(L) + ZZHT
    this%DDD(L) = (this%PPP(L) - this%PPP(L + 1))*MASFAC
    this%OOO(L) = this%DDD(L)*this%O3(L)*1.d-6
    this%CCC(L) = this%DDD(L)*this%CH4(L)*1.d-9
!-----------------------------------------------------------------------
!       call ACLIM_RH (PL, TL, QL, HHH, L1U)
!-----------------------------------------------------------------------
! quick fix Rel Humidity
    this%HHH(:) = 0.50d0

!!! set up clouds and aerosols
    this%AERSP(:, :) = 0.d0
    this%NDXAER(:, :) = 0
    do L = 1, L_
      this%NDXAER(L, 1) = NAA1(L)
      this%AERSP(L, 1) = AER1(L)
      this%NDXAER(L, 2) = NAA2(L)
      this%AERSP(L, 2) = AER2(L)
    end do
    this%LTOP = LWEPAR
    if (maxval(this%CLDFRW) .le. 0.005d0) then
      this%IWP(:) = 0.d0
      this%REFFI(:) = 0.d0
      this%LWP(:) = 0.d0
      this%REFFL(:) = 0.d0
    end if
    do L = 1, this%LTOP
      this%CLDIW(L) = 0
      CF = this%CLDFRW(L)
      if (CF .gt. 0.005d0) then
        this%CLF(L) = CF
        WLC(L) = CLDLWCW(L)/CF
        WIC(L) = CLDIWCW(L)/CF
        !  CLDIW is an integer flag: 1 = water cld, 2 = ice cloud, 3 = both
        if (WLC(L) .gt. 1.d-11) this%CLDIW(L) = 1
        if (WIC(L) .gt. 1.d-11) this%CLDIW(L) = this%CLDIW(L) + 2
      else
        this%CLF(L) = 0.d0
        WLC(L) = 0.d0
        WIC(L) = 0.d0
      end if
    end do
    !---derive R-effective for clouds:  the current UCI algorithm - use your own
    do L = 1, this%LTOP
      !---ice clouds
      if (WIC(L) .gt. 1.d-12) then
        PDEL = this%PPP(L) - this%PPP(L + 1)
        ZDEL = (this%ZZZ(L + 1) - this%ZZZ(L))*0.01d0  ! m
        this%IWP(L) = 1000.d0*WIC(L)*PDEL*G100/this%CLF(L)   ! g/m2
        ICWC = this%IWP(L)/ZDEL          ! g/m3
        this%REFFI(L) = 164.d0*(ICWC**0.23d0)
      else
        this%IWP(L) = 0.d0
        this%REFFI(L) = 0.d0
      end if
      !---water clouds
      if (WLC(L) .gt. 1.d-12) then
        PMID = 0.5d0*(this%PPP(L) + this%PPP(L + 1))
        PDEL = this%PPP(L) - this%PPP(L + 1)
        F1 = 0.005d0*(PMID - 610.d0)
        F1 = min(1.d0, max(0.d0, F1))
        this%LWP(L) = 1000.d0*WLC(L)*PDEL*G100     ! g/m2
        this%REFFL(L) = 9.6d0*F1 + 12.68d0*(1.d0 - F1)
      else
        this%LWP(L) = 0.d0
        this%REFFL(L) = 0.d0
      end if
    end do
    do L = 1, this%LTOP
      this%CLDFRW(L) = this%CLF(L)
    end do
      !!!  end of atmosphere setup
  end subroutine init

  subroutine run_cloudj(this, sza)
    !> the cloud-j interface object, needs to be initialized
    class(cloudj_interface_t), intent(inout) :: this
    !> solar zenith angle (deg)
    real(kind=dp), intent(in) :: SZA

! beware the OSA code uses single R*4 variables
    real(kind=sp)  :: OWAVEL, OWIND, OCHLR, OSA_dir(5)

    integer(kind=i_kind)  :: L, K, J

    do L = 1, this%LTOP
      this%CLF(L) = this%CLDFRW(L)
    end do

    this%IRAN = 1
    this%SOLF = 1.0
    this%U0 = cos(SZA*CPI180)

! beware the OSA code uses single R*4 variables
    ANGLES(1) = real(EMU(1), kind=sp)
    ANGLES(2) = real(EMU(2), kind=sp)
    ANGLES(3) = real(EMU(3), kind=sp)
    ANGLES(4) = real(EMU(4), kind=sp)
    ANGLES(5) = real(this%U0, kind=sp)
    OWIND = real(this%WIND, kind=sp)
    OCHLR = real(this%CHLR, kind=sp)
    do K = 1, NS2
      OWAVEL = real(WL(K), kind=sp)
      call OSA(OWAVEL, OWIND, OCHLR, ANGLES, OSA_dir)
      do J = 1, 5
        this%RFL(J, K) = real(OSA_dir(J), kind=dp)
! this overwrite the OSA with the readin albedo values
        this%RFL(J, K) = this%ALBEDO(J)
      end do
    end do
    this%LPRTJ = .false.
    if (this%LPRTJ) then
      write (6, '(a,f8.3,3f8.5)') 'SZA SOLF U0 albedo' &
        , SZA, this%SOLF, this%U0, this%RFL(5, 18)
      call JP_ATM0(this%PPP, this%TTT, this%DDD, this%OOO, this%ZZZ, L_)
      write (6, *) ' wvl  albedo u1:u4 & u0'
      do K = 1, NS2
        write (6, '(i5,f8.1,5f8.4)') K, WL(K), (this%RFL(J, K), J=1, 5)
      end do
    end if

    this%SKPERD(:, :) = 0.0
    this%SWMSQ(:) = 0.0
    this%OD18(:) = 0.0
    this%WTQCA(:) = 0.0

!=======================================================================
    call CLOUD_JX(this%U0, SZA, this%RFL, this%SOLF, this%LPRTJ, &
                  this%PPP, this%ZZZ, this%TTT, this%HHH, this%DDD, &
                  this%RRR, this%OOO, this%CCC, this%LWP, this%IWP, this%REFFL, &
                  this%REFFI, this%CLF, CLDCOR, this%CLDIW, &
                  this%AERSP, this%NDXAER, this%L1U, this%ANU, size(this%valjxx, 2), &
                  this%VALJXX, this%SKPERD, this%SWMSQ, this%OD18, &
                  CLDFLAG, NRANDO, this%IRAN, LNRG, this%NICA, &
                  this%JCOUNT, this%LDARK, this%WTQCA)
!=======================================================================

    do L = L_, 1, -1
      write (7, '(i3,1p, 72e9.2)') L, (this%VALJXX(L, K), K=1, NJX)
    end do
  end subroutine

end module boxmodel_cloudj_interface
