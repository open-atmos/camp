      PROGRAM tuv
*-----------------------------------------------------------------------------*
*=    Tropospheric Ultraviolet-Visible (TUV) radiation model                 =*
*=    Version 5.4                                                            =*
*=    November 2018                                                          =*
*-----------------------------------------------------------------------------*
*= Developed by Sasha Madronich with important contributions from:           =*
*= Chris Fischer, Siri Flocke, Julia Lee-Taylor, Bernhard Meyer,             =*
*= Irina Petropavlovskikh,  Xuexi Tie, and Jun Zen.                          =*
*= Special thanks to Knut Stamnes and co-workers for the development of the  =*
*= Discrete Ordinates code, and to Warren Wiscombe and co-workers for the    =*
*= development of the solar zenith angle subroutine. Citations for the many  =*
*= data bases (e.g. extraterrestrial irradiances, molecular spectra) may be  =*
*= found in the data files headers and/or in the subroutines that read them. =*
*=              To contact the author, write to:                             =*
*= Sasha Madronich, NCAR/ACD, P.O.Box 3000, Boulder, CO, 80307-3000, USA  or =*
*= send email to:  sasha@ucar.edu  or tuv@acd.ucar.edu                       =*
*-----------------------------------------------------------------------------*
*= This program is free software; you can redistribute it and/or modify      =*
*= it under the terms of the GNU General Public License as published by the  =*
*= Free Software Foundation;  either version 2 of the license, or (at your   =*
*= option) any later version.                                                =*
*= The TUV package is distributed in the hope that it will be useful, but    =*
*= WITHOUT ANY WARRANTY;  without even the implied warranty of MERCHANTIBI-  =*
*= LITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public     =*
*= License for more details.                                                 =*
*= To obtain a copy of the GNU General Public License, write to:             =*
*= Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.   =*
*-----------------------------------------------------------------------------*
*= Copyright (C) 1994-2018 by the University Corporation for Atmospheric     =*
*= Research, extending to all called subroutines, functions, and data unless =*
*= another source is specified.                                              =*
*-----------------------------------------------------------------------------*

      IMPLICIT NONE

* Include parameter file

      INCLUDE 'params'

* Wavelength grid:

      INTEGER nw, iw, nwint
      REAL wl(kw), wc(kw), wu(kw)
      REAL wstart, wstop

* Altitude grid

      INTEGER nz, nzm1, iz, izout
      REAL z(kz), zstart, zstop, zout

* Solar zenith angle and azimuth
* slant pathlengths in spherical geometry

      REAL sza(kt), zen, sznoon
      INTEGER nid(0:kz)
      REAL dsdh(0:kz,kz)

* Extra terrestrial solar flux and earth-Sun distance ^-2

      REAL f(kw), etf(kw)
      REAL esfact(kt)

* Ozone absorption cross section

      INTEGER mabs
      REAL o3xs(kz,kw)

* O2 absorption cross section

      REAL o2xs(kz,kw), o2xs1(kw)

* SO2 absorption cross section
     
      REAL so2xs(kw)

* NO2 absorption cross section
     
      REAL no2xs(kz,kw)

* Atmospheric optical parameters

      REAL tlev(kz), tlay(kz)
      REAL aircon(kz), aircol(kz), vcol(kz), scol(kz)
      REAL dtrl(kz,kw)
      REAL co3(kz)
      REAL dto3(kz,kw), dto2(kz,kw), dtso2(kz,kw), dtno2(kz,kw)
      REAL dtcld(kz,kw), omcld(kz,kw), gcld(kz,kw)
      REAL dtaer(kz,kw), omaer(kz,kw), gaer(kz,kw)
      REAL dtsnw(kz,kw), omsnw(kz,kw), gsnw(kz,kw)
      REAL albedo(kw)
      REAL dt_any(kz,kw), om_any(kz,kw), g_any(kz,kw)

* Spectral irradiance and actinic flux (scalar irradiance)

      REAL edir(kz), edn(kz), eup(kz)
      REAL sirrad(kz,kw)
      REAL fdir(kz), fdn(kz), fup(kz)
      REAL saflux(kz,kw)

* Spectral weighting functions and weighted radiation

      INTEGER ns, is
      REAL sw(ks,kw), rate(ks,kz), dose(ks)
      REAL drdw
      CHARACTER*50 slabel(ks)

* Photolysis coefficients (j-values)

      INTEGER nj, ij
      REAL sj(kj,kz,kw), valj(kj,kz)
      REAL djdw
      CHARACTER*50 jlabel(kj)
      INTEGER tpflag(kj)

**** Re-scaling factors (can be read from input file)
* New surface albedo and surface pressure (milli bar)
* Total columns of O3, SO2, NO2 (Dobson Units)
* Cloud optical depth, altitude of base and top
* Aerosol optical depth at 550 nm, single scattering albedo, Angstrom alpha

      REAL alsurf, psurf
      REAL o3_tc, so2_tc, no2_tc
      REAL taucld, zbase, ztop
      REAL tauaer, ssaaer, alpha

* Location: Lat and Lon (deg.), surface elev (km)
* Altitude, temperature and pressure for specific outputs

      REAL lat, lon
      REAL zaird, ztemp

* Time and/or solar zenith angle
      
      INTEGER iyear, imonth, iday
      INTEGER it, nt
      REAL t(kt), tstart, tstop
      REAL tmzone
      LOGICAL lzenit

* number of radiation streams

      INTEGER nstr

* input/output control

      LOGICAL intrct
      CHARACTER*6 inpfil, outfil

      INTEGER iout

      REAL dirsun, difdn, difup

      CHARACTER*1 again

* Save arrays for output:

      LOGICAL lirrad, laflux, lrates, ljvals, lmmech
      INTEGER isfix, ijfix, itfix, izfix, iwfix
      INTEGER nms, ims(ks), nmj, imj(kj)

      REAL svj_zj(kz,kj), svj_tj(kt,kj), svj_zt(kz,kt)
      REAL svr_zs(kz,ks), svr_ts(kt,ks), svr_zt(kz,kt)
      REAL svf_zw(kz,kw), svf_tw(kt,kw), svf_zt(kz,kt)
      REAL svi_zw(kz,kw), svi_tw(kt,kw), svi_zt(kz,kt)

* Planetary boundary layer height and pollutant concentrations

      INTEGER ipbl
      REAL zpbl
      REAL o3pbl, so2pbl, no2pbl, aod330

* WRF-Chem output control

      LOGICAL wrfchm

***** Surface waters (lakes, ocean)
*   sdom = spectral absorption by Dissolved Organic Matter (DOM) 
*          in lakes and ocean
*   h2oabs = sdom at specific wavenght

      INTEGER jdom, jd
      CHARACTER*50 dlabel(kdom)
      REAL sdom(kdom,kw)
      REAL h2oabs

*     ydepth = depth in meters
*   irradiances normalized to unity incidence:
*     se_0 = solar beam irradiance just below surface
*     de_0 = diffuse irradiance just below surface (sum over all angles)
*     se_y = solar beam irradiance at ydepth
*     de_y = diffuse irradiance at ydepth  (sum over all angles)
*     se_int = integral of solar irradiance from surface to ydepth 
*     de_int = integral of diffuse irradiance from surface to ydepth
*   spectral irradiances (total = direct + diffuse-dn) in units of W m-2 nm-1:
*     we_0 = just below air-water interface
*     we_int = from 0 to ydepth

      REAL ydepth
      REAL se_0, de_0, se_y, de_y, se_int, de_int
      REAL we_0, we_int

*  in-water dose rates, doses - DNA weighted unless otherwise specified
*     dose in air just above surface, computed by time integration of rate(13,1)
*     wrate0, wdose0 = dose rate, dose, just below surface
*     wratei, wdosei = dose rate, dose, integrated from surface to ydepth

      REAL adose
      REAL wratei, wrate0
      REAL wdosei, wdose0

****** Other user-defined variables here:

* spectrum locator indices:

      integer js_dna, js_uvi, jd_dom
      integer id


* --- END OF DECLARATIONS ---------------------------------------------

* re-entry point

 1000 CONTINUE

* Open log file:

c      OPEN(UNIT=kout,FILE='tuvlog',STATUS='UNKNOWN')
      OPEN(UNIT=kout,FILE='../'//'tuvlog'//'.txt',STATUS='UNKNOWN')

* ___ SECTION 1: SIMPLE INPUT VARIABLES --------------------------------
******* Read simple input variables from a file:

* can read interactively (intrct = .TRUE.) 
* or in batch mode (intrct = .FALSE.)

      intrct = .TRUE.
c      intrct = .FALSE.
      IF ( .NOT. intrct) inpfil = 'usrinp'

      CALL rdinp(intrct, 
     $     inpfil, outfil, nstr,   lat,    lon,    tmzone,
     $     iyear,  imonth, iday,   zstart, zstop,  nz,
     $     wstart, wstop,  nwint,  tstart, tstop,  nt,
     $     lzenit, alsurf, psurf,  o3_tc,  so2_tc, no2_tc,
     $     taucld, zbase,  ztop,   tauaer, ssaaer, alpha,
     $     dirsun, difdn,  difup,  zout,   zaird,  ztemp,
     $     lirrad, laflux, lmmech, lrates, isfix,  nms,
     $     ljvals, ijfix,  nmj,    iwfix,  itfix,  izfix,
     $     ims,    slabel, imj,    jlabel)

      IF(outfil .EQ. 'screen') THEN
         iout = 6
      ELSE
         iout = 30
      ENDIF         

************* Can overwrite basic inputs here manually:
* Input and output files:
*   inpfil = input file name
*   outfil = output file name
* Radiative transfer scheme:
*   nstr = number of streams
*          If nstr < 2, will use 2-stream Delta Eddington
*          If nstr > 1, will use nstr-stream discrete ordinates
* Location (geographic):
*   lat = LATITUDE (degrees, North = positive)
*   lon = LONGITUDE (degrees, East = positive)
*   tmzone = Local time zone difference (hrs) from Universal Time (ut):  
*            ut = timloc - tmzone
* Date:
*   iyear = year (1950 to 2050)
*   imonth = month (1 to 12)
*   iday = day of month
* Time of day grid:
*   tstart = starting time, local hours
*   tstop = stopping time, local hours
*   nt = number of time steps
*   lzenit = switch for solar zenith angle (sza) grid rather than time 
*             grid. If lzenit = .TRUE. then 
*                tstart = first sza in deg., 
*                tstop = last sza in deg., 
*                nt = number of sza steps. 
*                esfact = 1. (Earth-sun distance = 1.000 AU)
* Vertical grid:
*   zstart = surface elevation above sea level, km
*   zstop = top of the atmosphere (exospheric), km
*   nz = number of vertical levels, equally spaced
*        (nz will increase by +1 if zout does not match altitude grid)
* Wavlength grid:
*   wstart = starting wavelength, nm
*   wstop  = final wavelength, nm
*   nwint = number of wavelength intervals, equally spaced
*           if nwint < 0, the standard atmospheric wavelength grid, not
*           equally spaced, from 120 to 735 nm, will be used. In this
*           case, wstart and wstop values are ignored.
* Surface condition:
*   alsurf = surface albedo, wavelength independent
*   psurf = surface pressure, mbar.  Set to negative value to use
*           US Standard Atmosphere, 1976 (USSA76)
* Column amounts of absorbers (in Dobson Units, from surface to space):
*          Vertical profile for O3 from USSA76.  For SO2 and NO2, vertical
*          concentration profile is 2.69e10 molec cm-3 between 0 and 
*          1 km above sea level, very small residual (10/largest) above 1 km.
*   o3_tc = ozone (O3)
*   so2_tc = sulfur dioxide (SO2)
*   no2_tc = nitrogen dioxide (NO2)
* Cloud, assumed horizontally uniform, total coverage, single scattering
*         albedo = 0.9999, asymmetry factor = 0.85, indep. of wavelength,
*         and also uniform vertically between zbase and ztop:
*   taucld = vertical optical depth, independent of wavelength
*   zbase = altitude of base, km above sea level
*   ztop = altitude of top, km above sea level
* Aerosols, assumed vertical provile typical of continental regions from
*         Elterman (1968):
*   tauaer = aerosol vertical optical depth at 550 nm, from surface to space. 
*           If negative, will default to Elterman's values (ca. 0.235 
*           at 550 nm).
*   ssaaer = single scattering albedo of aerosols, wavelength-independent.
*   alpha = Angstrom coefficient = exponent for wavelength dependence of 
*           tauaer, so that  tauaer1/tauaer2  = (w2/w1)**alpha.
* Directional components of radiation, weighting factors:
*   dirsun = direct sun
*   difdn = down-welling diffuse
*   difup = up-welling diffuse
*        e.g. use:
*        dirsun = difdn = 1.0, difup = 0 for total down-welling irradiance
*        dirsun = difdn = difup = 1.0 for actinic flux from all directions
*        dirsun = difdn = 1.0, difup = -1 for net irradiance
* Output altitude:
*   zout = altitude, km, for desired output.
*        If not within 1 m of altitude grid, an additional
*        level will be inserted and nz will be increased by +1.
*   zaird = air density (molec. cm-3) at zout.  Set to negative value for
*        default USSA76 value interpolated to zout.
*   ztemp = air temperature (K) at zout.  Set to negative value for
*        default USSA76 value interpolated to zout.
* Output options, logical switches:
*   lirrad = output spectral irradiance
*   laflux = output spectral actinic flux
*   lmmech = output for NCAR Master Mechanism use
*   lrates = output dose rates (UVB, UVA, CIE/erythema, etc.)
* Output options, integer selections:
*   isfix:  if > 0, output dose rate for action spectrum is=isfix, tabulated
*           for different times and altitudes.
*   ijfix:  if > 0, output j-values for reaction ij=ijfix, tabulated
*           for different times and altitudes.
*   iwfix:  if > 0, output spectral irradiance and/or spectral actinic
*           flux at wavelength iw=iwfix, tabulated for different times
*           and altitudes.
*   itfix:  if > 0, output spectral irradiance and/or spectral actinic
*           flux at time it=itfix, tabulated for different altitudes
*           and wavelengths.
*   izfix:  if > 0, output spectral irradiance and/or spectral actinic
*           flux at altitude iz=izfix, tabulated for different times
*           and wavelengths.
*   nms:    number of dose rates that will be reported. Selections must be 
*           made interactively, or by editing input file.
*   nmj:    number of j-values that will be reported. Selections must be 
*           made interactively, or by editing input file.
* The following default settings are also found in the input file 'defin1':

c      inpfil = defin1
c      outfil = usrout
c      nstr = -2
c      lat = 0.
c      lon = 0.
c      tmzone = 0.
c      iyear = 2002
c      imonth = 3
c      iday = 21
c      zstart = 0.
c      zstop = 80.
c      nz = 80
c      wstart = 280.
c      wstop = 420.
c      nwint = 140
c      tstart = 12.
c      tstop = 20.
c      nt = 5
c      lzenit = .FALSE.
c      alsurf = 0.1
c      psurf = -999.
c      o3_tc = 300.
c      so2_tc = 0.
c      no2_tc = 0.
c      tcloud = 0.
c      zbase = 4.
c      ztop = 5.
c      tauaer = 0.235
c      ssaaer = 0.99
c      alpha = 1.
c      dirsun = 1.
c      difdn = 1.
c      difup = 0.
c      zout = 0.
c      zaird = -999.
c      ztemp = -999.
c      lirrad = .TRUE.
c      laflux = .FALSE.
c      lmmech = .FALSE.
c      lrates = .TRUE.
c      isfix = 0
*      nms cannot be set here
c      ljvals = .FALSE.
c      ijfix = 0
*      nmj cannot be set here
c      iwfix = 0
c      itfix = 0
c      izfix = 0

      IF(nstr .LT. 2) THEN
         WRITE(kout,*) 'Delta-Eddington 2-stream radiative transfer' 
      ELSE
         WRITE(kout,*) 'Discrete ordinates ', 
     $        nstr, '-stream radiative transfer' 
      ENDIF

      WRITE(*,*) 'calculating....'

* ___ SECTION 2: SET GRIDS _________________________________________________

* altitudes (creates altitude grid, locates index for selected output, izout)

      CALL gridz(zstart, zstop, nz, z, zout, izout)
      IF(izfix .GT. 0) izout = izfix

* time/zenith (creates time/zenith angle grid, starting at tstart)

      CALL gridt(lat, lon, tmzone,
     $     iyear, imonth, iday,
     $     lzenit, tstart, tstop,
     $     nt, t, sza, sznoon, esfact)

* wavelength grid, user-set range and spacing. 
* NOTE:  Wavelengths are in vacuum, and therefore independent of altitude.
* To use wavelengths in air, see options in subroutine gridw

      CALL gridw(wstart, wstop, nwint,
     $     nw, wl, wc, wu)

* ___ SECTION 3: SET UP VERTICAL PROFILES OF TEMPERATURE, AIR DENSITY, and OZONE

***** Temperature vertical profile, Kelvin 
*   can overwrite temperature at altitude z(izout)

      CALL vptmp(nz,z, tlev,tlay)
      IF(ztemp .GT. nzero) tlev(izout) = ztemp

*****  Air density (molec cm-3) vertical profile 
*   can overwrite air density at altitude z(izout)

      CALL vpair(psurf, nz, z,
     $     aircon, aircol)
      IF(zaird .GT. nzero) aircon(izout) = zaird

*****
*! PBL pollutants will be added if zpbl > 0.
* CAUTIONS:  
* 1. The top of the PBL, zpbl in km, should be on one of the z-grid altitudes.
* 2. Concentrations, column increments, and optical depths
*       will be overwritten between surface and zpbl.
* 3. Inserting PBL constituents may change their total column amount.
* 4. Above pbl, the following are used:
*       for O3:  USSA or other profile
*       for NO2 and SO2: set to zero.
*       for aerosols: Elterman
* Turning on pbl will affect subroutines:
* vpo3, setno2, setso2, and setaer. See there for details

      zpbl = -999.
C      zpbl = 3.

* locate z-index for top of pbl

      ipbl = 0
      IF(zpbl. GT. 0.) THEN
         DO iz = 1, nz-1
            IF(z(iz+1) .GT. z(1) + zpbl*1.00001) GO TO 19
         ENDDO
 19      CONTINUE
         ipbl = iz - 1
         write(*,*) 'top of PBL index, height (km) ', ipbl, z(ipbl)

* specify pbl concetrations, in parts per billion

         o3pbl = 100.
         so2pbl = 10.
         no2pbl = 50.

* PBL aerosol optical depth at 330 nm
* (to change ssa and g of pbl aerosols, go to subroutine setair.f)

         aod330 = 0.8

      ENDIF

***** Ozone vertical profile

      CALL vpo3(ipbl, zpbl, o3pbl, 
     $       o3_tc, nz, z, aircol, co3)

* ___ SECTION 4: READ SPECTRAL DATA ____________________________

* read (and grid) extra terrestrial flux data:
      
      CALL rdetfl(nw,wl, f)

* read cross section data for 
*    O2 (will overwrite at Lyman-alpha and SRB wavelengths
*            see subroutine la_srb.f)
*    O3 (temperature-dependent)
*    SO2 
*    NO2

      nzm1 = nz - 1
      CALL rdo2xs(nw,wl, o2xs1)
      mabs = 1
      CALL rdo3xs(mabs,nzm1,tlay,nw,wl, o3xs)
      CALL rdso2xs(nw,wl, so2xs)
      CALL rdno2xs(nz,tlay,nw,wl, no2xs)

****** Spectral weighting functions 
* (Some of these depend on temperature T and pressure P, and therefore
*  on altitude z.  Therefore they are computed only after the T and P profiles
*  are set above with subroutines settmp and setair.)
* Photo-physical   set in swphys.f (transmission functions)
* Photo-biological set in swbiol.f (action spectra)
* Photo-chemical   set in swchem.f (cross sections x quantum yields)* Physical 
*   and biological weigthing functions are assumed to depend
*   only on wavelength.
* Chemical weighting functions (product of cross-section x quantum yield)
*   for many photolysis reactions are known to depend on temperature
*   and/or pressure, and therefore are functions of wavelength and altitude.
* Output:
* from swphys & swbiol:  sw(ks,kw) - for each weighting function slabel(ks)
* from swchem:  sj(kj,kz,kw) - for each reaction jlabel(kj)
* For swchem, need to know temperature and pressure profiles.

      CALL swphys(nw,wl,wc, ns,sw,slabel)
      CALL swbiol(nw,wl,wc, ns,sw,slabel)
      CALL swchem(nw,wl,nz,tlev,aircon, nj,sj,jlabel,tpflag)

** Read other spectral data
* absorption coefficients for Dissolved Organic Matter (DOM) in surface waters

      CALL swdom(nw,wl,wc, jdom,dlabel,sdom)

* locate indices for some specific spectra:

      js_dna = 0
      js_uvi = 0
      DO is = 1, ns
         IF(slabel(is) .EQ. 
     $        'DNA damage, in vitro (Setlow, 1974)               ') 
     $        js_dna = is

         IF(slabel(is) .EQ. 
     $        'UV index (WMO, 1994; Webb et al., 2011)')           
     $        js_uvi = is
      ENDDO

      jd_dom = 0
      DO jd = 1, jdom
         if(dlabel(jd) .eq. 
     $        'Generic DOM absorption')
     $        jd_dom = jd
      ENDDO

c      write(*,*) js_dna, js_uvi, jd_dom

**** The following CALL is normally commented out.
* Subroutine newlst regenerates the list of weighting functions 
* (molecular and biological spectra) when new ones are added, to 
* update the default input files (defin1, defin2. etc.).  User
* input files, e.g. usrinp, should be similarly updated. 
* The program STOPS at the completion of newlst.
* If not in use, newlst.o can be safely removed from Makefile.

c      CALL newlst(ns,slabel,nj,jlabel)

**** Option for writing look-up tables of 
* (molecular cross sections x quantum yields) 
* for WRF-Chem, at selected temperatures and pressures. 
* STOPs after tables are written.

      wrfchm = .FALSE.
      IF (inpfil .EQ. 'defin5') wrfchm = .TRUE.
      IF (wrfchm) CALL wrflut(nw, wl, nz, tlev, aircon)

* ___ SECTION 5: SET ATMOSPHERIC OPTICAL DEPTH INCREMENTS _____________________

* Rayleigh optical depth increments:

      CALL odrl(nz, z, nw, wl, aircol, dtrl)
      
* O2 vertical profile and O2 absorption optical depths
*   For now, O2 densitiy assumed as 20.95% of air density, can be changed
*   in subroutine.
*   Optical depths in Lyman-alpha and SRB will be over-written
*   in subroutine la_srb.f

      CALL seto2(nz,z,nw,wl,aircol,o2xs1, dto2)

* Ozone optical depths

      CALL odo3(nz,z,nw,wl,o3xs,co3, dto3)

* SO2 vertical profile and optical depths

      CALL setso2(ipbl, zpbl, so2pbl,
     $     so2_tc, nz, z, nw, wl, so2xs,
     $     tlay, aircol,
     $     dtso2)

* NO2 vertical profile and optical depths

      CALL setno2(ipbl, zpbl, no2pbl, 
     $     no2_tc, nz, z, nw, wl, no2xs,
     $     tlay, aircol,
     $     dtno2)

* Cloud vertical profile, optical depths, single scattering albedo, asymmetry factor

      CALL setcld(taucld,zbase,ztop,
     $     nz,z,nw,wl,
     $     dtcld,omcld,gcld)

* Aerosol vertical profile, optical depths, single scattering albedo, asymmetry factor

      CALL setaer(ipbl, zpbl, aod330,
     $     tauaer, ssaaer, alpha,
     $     nz, z, nw, wl,
     $     dtaer, omaer, gaer)

* Snowpack physical and optical depths, single scattering albedo, asymmetry factor

      CALL setsnw(
     $     nz,z,nw,wl,
     $     dtsnw,omsnw,gsnw)

* Surface albedo

      CALL setalb(alsurf,nw,wl,
     $     albedo)

* Set any additional absorber or scatterer:
* Must populate dt_any(kz,kw), om_any(kz,kw), g_any(kz,kw) manually
* This allows user to put in arbitrary absorber or scatterer
* could write a subroutine, e.g.:
C      CALL setany(nz,z,nw,wl,aircol, dt_any,om_any, g_any)
* or write manually here.

      DO iz = 1, nz-1
         DO iw = 1, nw-1
c            dt_any(iz,iw) = 0.79*aircol(iz) * 2.e-17 ! N2 VUV absorption
            dt_any(iz,iw) = 0.
            om_any(iz,iw) = 0.
            g_any(iz,iw) = 0.
         ENDDO
      ENDDO

* ___ SECTION 6: TIME/SZA LOOP  _____________________________________

* Initialize any time-integrated quantities here

      adose = 0.
      wdose0 = 0.
      wdosei = 0.
      CALL zero1(dose,ks)

* Loop over time or solar zenith angle (zen):

      
      write(*,*) 'Date, Lat, Lon, Min_SZA'
      write(*,222) iyear,imonth,iday,lat,lon,sznoon
 222  format(i4,'/',i2,'/',i2,3(1x,F7.3))
      
      DO 20, it = 1, nt

         zen = sza(it)

         WRITE(*,200) it, zen, esfact(it)
         WRITE(kout,200) it, zen, esfact(it)
 200     FORMAT('step = ', I4,' sza = ', F9.3, 
     $        ' Earth-sun factor = ', F10.7)


* correction for earth-sun distance

         DO iw = 1, nw - 1
            etf(iw) = f(iw) * esfact(it)
         ENDDO

* ____ SECTION 7: CALCULATE ZENITH ANGLE-DEPENDENT QUANTITIES __________

* slant path lengths for spherical geometry

         CALL sphers(nz,z,zen, dsdh,nid)
         CALL airmas(nz, dsdh,nid, aircol,vcol,scol)

* Recalculate effective O2 optical depth and cross sections for Lyman-alpha
* and Schumann-Runge bands, must know zenith angle
* Then assign O2 cross section to sj(1,*,*)

         CALL la_srb(nz,z,tlev,nw,wl,vcol,scol,o2xs1,
     $        dto2,o2xs)
         CALL sjo2(nz,nw,o2xs,1, sj)

* ____ SECTION 8: WAVELENGTH LOOP ______________________________________

* initialize for wavelength integration

         CALL zero2(rate,ks,kz)
         CALL zero2(valj,kj,kz)
         wrate0 = 0.
         wratei = 0.

***** Main wavelength loop:

         DO 10, iw = 1, nw-1

** monochromatic radiative transfer. Outputs are:
*  normalized irradiances     edir(iz), edn(iz), eup(iz) 
*  normalized actinic fluxes  fdir(iz), fdn(zi), fup(iz)
*  where 
*  dir = direct beam, dn = down-welling diffuse, up = up-welling diffuse

            CALL rtlink(nstr, nz,
     $           iw, albedo(iw), zen,
     $           dsdh,nid,
     $           dtrl,
     $           dto3,
     $           dto2,
     $           dtso2,
     $           dtno2,
     $           dtcld, omcld, gcld,
     $           dtaer,omaer,gaer,
     $           dtsnw,omsnw,gsnw,
     $           dt_any,om_any,g_any,
     $           edir, edn, eup, fdir, fdn, fup)

* Spectral irradiance, W m-2 nm-1
* for downwelling only, use difup = 0.

            DO iz = 1, nz
               sirrad(iz,iw) = etf(iw) * 
     $           (dirsun*edir(iz) + difdn*edn(iz) + difup*eup(iz))
            ENDDO

* Spectral actinic flux, quanta s-1 nm-1 cm-2, all directions:
*    units conversion:  1.e-4 * (wc*1e-9) / hc

            DO iz = 1, nz
               saflux(iz,iw) = etf(iw) * (1.e-13 * wc(iw) / hc) *
     $              (dirsun*fdir(iz) + difdn*fdn(iz) + difup*fup(iz))
            ENDDO

*** Accumulate weighted integrals over wavelength, at all altitudes:

            DO iz = 1, nz

* Weighted irradiances (dose rates) W m-2

               DO is = 1, ns
                  drdw = sirrad(iz,iw) * sw(is,iw) 
                  rate(is,iz) = rate(is,iz) + drdw * (wu(iw) - wl(iw))
               ENDDO

* Photolysis rate coefficients (J-values) s-1

               DO ij = 1, nj
                  djdw = saflux(iz,iw) * sj(ij,iz,iw)
                  valj(ij,iz) = valj(ij,iz) + djdw * (wu(iw) - wl(iw))
               ENDDO

            ENDDO

************ In-water radiation:
*   Input:  
*     ydepth, in meters, for which radiation field is desired
*     h2oabs = absorption coeff of DOM in H2O at this wavelength, 1/meter
*     zen = solar zenith angle
*   Output from subroutine waters:
*     se_0 = solar beam irradiance just below surface
*     de_0 = diffuse irradiance just below surface (sum over all angles)
*     se_y = solar beam irradiance at ydepth
*     de_y = diffuse irradiance at ydepth  (sum over all angles)
*     se_i = integral of solar beam irradiance from surface to ydepth 
*     de_i = integral of diffuse irradiance from surface to ydepth

            ydepth = 1.
            h2oabs = sdom(jd_dom,iw)
            CALL waters(zen,h2oabs,ydepth, 
     $           se_0,de_0,se_y,de_y,se_int,de_int)
            
* calculate spectral irradiances in water:
* irradiance just below air-water interface:

            we_0 = etf(iw) * (se_0*edir(1)*dirsun + 
     $           de_0*edn(1)*difdn)

* average spectral irradiance in water, from 0 to ydepth = integral / ydepth

            we_int = etf(iw) * (se_int*edir(1)*dirsun + 
     $           de_int*edn(1)*difdn) / ydepth

* calculate DNA-weighted irradiance, just below surface and
* averaged from 0 to ydepth

            drdw = we_0 * sw(js_dna,iw)
            wrate0 = wrate0 + drdw * (wu(iw)-wl(iw))

            drdw = we_int * sw(js_dna,iw)
            wratei = wratei + drdw * (wu(iw)-wl(iw))
               

**** Save irradiances and actinic fluxes for output

            CALL saver1(it, itfix, iw, iwfix,  nz, izout,
     $           sirrad, saflux,
     $           svi_zw, svf_zw, svi_zt, svf_zt, svi_tw, svf_tw)
 10      CONTINUE

*^^^^^^^^^^^^^^^^ end wavelength loop

**** integrate doses over time: 
* adose = dose in air just above surface
* wdose0 = dose in water just below surface
* wdosei = dose averaged over ydepth

         adose = adose + 
     $        rate(13,1) * 3600.* (tstop - tstart)/float(nt-1)
         wdose0 = wdose0 + 
     $        wrate0 * 3600.* (tstop - tstart)/float(nt-1)
         wdosei = wdosei + 
     $        wratei * 3600.* (tstop - tstart)/float(nt-1)

* Save dose rates and j-values for output

         CALL saver2(it,itfix, nz,izout, ns,isfix,ims, nj,ijfix,imj,
     $        rate, valj,
     $        svr_zs, svj_zj, svr_zt, svj_zt, svr_ts, svj_tj)

 20   CONTINUE

* output in-water doses

c      write(*,222) adose, wdose0, wdosei, dlabel(jd_dom)
c 222  format(3(0pf10.4,1x),a50)

**output all Js at zout

c      do iz = 1, nz
c         if(z(iz) .eq. zout) then
c            do ij = 1, nj
c               write(44,444) valj(ij,iz), jlabel(ij)
c            enddo
c         endif
c      enddo
c 444  format(1pe11.4,1x,a50)
*^^^^^^^^^^^^^^^^ end time/zenith loop

* ____ SECTION 9: OUTPUT ______________________________________________

      call outpt1( outfil, iout, 
     $     lirrad, laflux, lrates, ljvals, lmmech, lzenit,
     $     nms, ims, nmj, imj,
     $     nz, z, tlev, aircon, izout,
     $     nw, wl, etf, iwfix,
     $     nt, t, sza, itfix,
     $     ns, slabel, isfix, nj, jlabel, ijfix,
     $     svj_zj, svj_tj, svj_zt,
     $     svr_zs, svr_ts, svr_zt,
     $     svf_zw, svf_tw, svf_zt,
     $     svi_zw, svi_tw, svi_zt )

 30   continue

*_______________________________________________________________________

      IF(intrct) THEN
         WRITE(*,*) 'do you want to do another calculation?'
         WRITE(*,*) 'y = yes'
         WRITE(*,*) 'any other key = no'
         READ(*,1001) again
 1001    FORMAT(A1)
         IF(again .EQ. 'y' .OR. again .EQ. 'Y') GO TO 1000
      ENDIF

      CLOSE(iout)
      CLOSE(kout)
      END



