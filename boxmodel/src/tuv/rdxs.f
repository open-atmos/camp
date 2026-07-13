* This file contains subroutines related to reading the
* absorption cross sections of gases that contribute to atmospheric transmission:
* Some of these subroutines are also called from rxn.f when loading photolysis cross sections
* for these same gases. It is possible to have different cross sections for 
* transmission and for photolysis, e.g. for ozone, Bass et al. could be used
* for transmission while Molina and Molina could be used for photolysis.  
* This flexibility can be useful but users should be aware.
* For xsections that are temperature dependent, caution should be used in passing the proper 
* temperature to the data routines.  Usually, transmission is for layers, TLAY(NZ-1), while
* photolysis is at levels, T(NZ).
* The following subroutines are her: 
*     rdo3xs
*       o3_mol
*       o3_rei
*       o3_bas
*       o3_wmo
*       o3_jpl
*     rdo2xs
*     rdno2xs
*       no2xs_d
*       no2xs_jpl94
*       no2xs_har
*     rdso2xs
*=============================================================================*

      SUBROUTINE rdo3xs(mabs, nz,t,nw,wl, xs)

*-----------------------------------------------------------------------------*
*=  PURPOSE:                                                                 =*
*=  Read ozone molecular absorption cross section.  Re-grid data to match    =*
*=  specified wavelength working grid. Interpolate in temperature as needed  =*
*-----------------------------------------------------------------------------*
*=  PARAMETERS:                                                              =*
*=  MABS   - INTEGER, option for splicing different combinations of       (I)=*
*=           absorption cross secttions                                      =*
*=  NZ     - INTEGER, number of altitude levels or layers                 (I)=*
*=  T      - REAL, temperature of levels or layers                        (I)=*
*=  NW     - INTEGER, number of specified intervals + 1 in working        (I)=*
*=           wavelength grid                                                 =*
*=  WL     - REAL, vector of lower limits of wavelength intervals in      (I)=*
*=           working wavelength grid. In vacuum, nm                          =*
*=  XS     - REAL, molecular absoprtion cross section (cm^2) of O3 at     (O)=*
*=           each specified wavelength (WMO value at 273)                    =*
*-----------------------------------------------------------------------------*
      use tuv_params
      IMPLICIT NONE

* input: (altitude working grid)

      INTEGER iw, nw
      REAL wl(kw)

      INTEGER iz, nz
      REAL t(kz)

* internal

      INTEGER mabs
      REAL tc

* output:
* ozone absorption cross sections interpolated to 
*   working wavelength grid (iw)
*   working altitude grid (iz) for temperature of layer or level (specified in call)
* Units are cm2 molecule-1 in vacuum

      REAL xs(kz,kw)

* wavelength-interpolated values from different O3 data sources 
* also values of significant wavelengths, converted to vacuum

      REAL rei218(kw), rei228(kw), rei243(kw), rei295(kw)
      REAL v195, v345, v830

      REAL wmo203(kw), wmo273(kw)
      REAL v176, v850

      REAL jpl295(kw), jpl218(kw)
      REAL v186, v825

      REAL mol226(kw), mol263(kw), mol298(kw)
      REAL v185, v240, v350

      REAL c0(kw), c1(kw), c2(kw)
      REAL vb245, vb342

*_______________________________________________________________________
* read data from different sources
* rei = Reims group (Malicet et al., Brion et al.)
* jpl = JPL 2006 evaluation
* wmo = WMO 1985 O3 assessment
* mol = Molina and Molina
* bas = Bass et al.

      CALL o3_rei(nw,wl, rei218,rei228,rei243,rei295, v195,v345,v830)

      CALL o3_jpl(nw,wl, jpl218,jpl295, v186,v825)

      CALL o3_wmo(nw,wl, wmo203,wmo273, v176,v850)

      CALL o3_mol(nw,wl, mol226,mol263,mol298, v185,v240,v350)

      CALL o3_bas(nw,wl, c0,c1,c2, vb245,vb342)

****** option 1:

      IF(mabs. EQ. 1) THEN

* assign according to wavelength range:
*  175.439 - 185.185  1985WMO (203, 273 K)
*  185.185 - 195.00   2006JPL_O3 (218, 295 K)
*  195.00  - 345.00   Reims group (218, 228, 243, 295 K)
*  345.00  - 830.00   Reims group (295 K)
*  no extrapolations in temperature allowed

         DO 10 iw = 1, nw-1
         DO 20 iz = 1, nz

         IF(wl(iw) .LT. v185) THEN
            xs(iz,iw) = wmo203(iw) + 
     $           (wmo273(iw) - wmo203(iw))*(t(iz) - 203.)/(273. - 203.)
            IF (t(iz) .LE. 203.) xs(iz,iw) = wmo203(iw)
            IF (t(iz) .GE. 273.) xs(iz,iw) = wmo273(iw)
         ENDIF

         IF(wl(iw) .GE. v185 .AND. wl(iw) .LE. v195) THEN
            xs(iz,iw) = jpl218(iw) + 
     $           (jpl295(iw) - jpl218(iw))*(t(iz) - 218.)/(295. - 218.)
            IF (t(iz) .LE. 218.) xs(iz,iw) = jpl218(iw)
            IF (t(iz) .GE. 295.) xs(iz,iw) = jpl295(iw)
         ENDIF

         IF(wl(iw) .GE. v195 .AND. wl(iw) .LT. v345) THEN
            IF (t(iz) .GE. 218. .AND. t(iz) .LT. 228.) THEN
               xs(iz,iw) = rei218(iw) + 
     $              (t(iz)-218.)*(rei228(iw)-rei218(iw))/(228.-218.)
            ELSEIF (t(iz) .GE. 228. .AND. t(iz) .LT. 243.) THEN
               xs(iz,iw) = rei228(iw) +
     $              (t(iz)-228.)*(rei243(iw)-rei228(iw))/(243.-228.)
            ELSEIF (t(iz) .GE. 243. .AND. t(iz) .LT. 295.) THEN
               xs(iz,iw) = rei243(iw) +
     $              (t(iz)-243.)*(rei295(iw)-rei243(iw))/(295.-243.)
            ENDIF
            IF (t(iz) .LT. 218.) xs(iz,iw) = rei218(iw)
            IF (t(iz) .GE. 295.) xs(iz,iw) = rei295(iw)
         ENDIF

         IF(wl(iw) .GE. v345) THEN
            xs(iz,iw) = rei295(iw)
         ENDIF

 20      CONTINUE
 10      CONTINUE

      ELSEIF(mabs .EQ. 2) THEN

* use exclusively JPL-2006

         DO iw = 1, nw-1
         DO iz = 1, nz

            xs(iz,iw) = jpl218(iw) + 
     $           (jpl295(iw) - jpl218(iw))*(t(iz) - 218.)/(295. - 218.)
            IF (t(iz) .LE. 218.) xs(iz,iw) = jpl218(iw)
            IF (t(iz) .GE. 295.) xs(iz,iw) = jpl295(iw)

         ENDDO
         ENDDO

      ELSEIF(mabs .EQ. 3) THEN

* use exclusively Molina and Molina

         DO iw = 1, nw-1
         DO iz = 1, nz
            
            IF(wl(iw) .LT. v240) THEN
               xs(iz,iw) = mol226(iw) + 
     $              (t(iz)-226.)*(mol298(iw)-mol226(iw))/(298.-226.)
            ELSE
               IF(t(iz) .LT. 263.) THEN
                  xs(iz,iw) = mol226(iw) + 
     $                 (t(iz)-226.)*(mol263(iw)-mol226(iw))/(263.-226.)
               ELSE
                  xs(iz,iw) = mol263(iw) + 
     $                 (t(iz)-263.)*(mol298(iw)-mol263(iw))/(298.-263.)
               ENDIF
            ENDIF
            IF (t(iz) .LE. 226.) xs(iz,iw) = mol226(iw)
            IF (t(iz) .GE. 298.) xs(iz,iw) = mol298(iw)

         ENDDO
         ENDDO

      ELSEIF(mabs .EQ. 4) THEN

* use exclusively Bass et al.
* note limited wavelength range 245-342

         DO iw = 1, nw-1
         DO iz = 1, nz

            tc = t(iz) - 273.15
            xs(iz,iw) = c0(iw) + c1(iw)*tc + c2(iw)*tc*tc

         ENDDO
         ENDDO

      ELSE
         STOP 'mabs not set in rdxs.f'
      ENDIF

      RETURN
      END

*=============================================================================*

      SUBROUTINE o3_rei(nw,wl, 
     $     rei218,rei228,rei243,rei295, v195,v345,v830)

*-----------------------------------------------------------------------------*
*=  PURPOSE:                                                                 =*
*=  Read and interpolate the O3 cross section from Reims group               =*
*-----------------------------------------------------------------------------*
*=  PARAMETERS:                                                              =*
*=  NW     - INTEGER, number of specified intervals + 1 in working        (I)=*
*=           wavelength grid                                                 =*
*=  WL     - REAL, vector of lower limits of wavelength intervals in      (I)=*
*=           working wavelength grid                                         =*
*=  REI218 - REAL, cross section (cm^2) for O3 at 218K                    (O)=*
*=  REI228 - REAL, cross section (cm^2) for O3 at 218K                    (O)=*
*=  REI243 - REAL, cross section (cm^2) for O3 at 218K                    (O)=*
*=  REI295 - REAL, cross section (cm^2) for O3 at 218K                    (O)=*
*=  V195   - REAL, exact wavelength in vacuum for data breaks             (O)=*
*=              e.g. start, stop, or other change                            =*
*=  V345   - REAL, exact wavelength in vacuum for data breaks             (O)=*
*=  V830   - REAL, exact wavelength in vacuum for data breaks             (O)=*
*-----------------------------------------------------------------------------*
      use tuv_params
      IMPLICIT NONE

*  input

      INTEGER nw, iw
      REAL wl(kw)

** internal

      INTEGER kdata
      PARAMETER (kdata = 70000)

      INTEGER n1, n2, n3, n4
      REAL x1(kdata), x2(kdata), x3(kdata), x4(kdata)
      REAL y1(kdata), y2(kdata), y3(kdata), y4(kdata)

      INTEGER i
      INTEGER ierr

* used for air-to-vacuum wavelength conversion

      REAL refrac, ri(kdata)
      EXTERNAL refrac

* output:

      REAL rei218(kw), rei228(kw), rei243(kw), rei295(kw)
      REAL v195, v345, v830

* data from the Reims group:
*=  For Hartley and Huggins bands, use temperature-dependent values from     =*
*=  Malicet et al., J. Atmos. Chem.  v.21, pp.263-273, 1995.                 =*
*=  over 345.01 - 830.00, use values from Brion, room temperature only

      OPEN(UNIT=kin,FILE='DATAE1/O3/1995Malicet_O3.txt',STATUS='old')
      DO i = 1, 2
         READ(kin,*)
      ENDDO
      n1 = 15001
      n2 = 15001
      n3 = 15001
      n4 = 15001
      DO i = 1, n1
         READ(kin,*) x1(i), y1(i), y2(i), y3(i), y4(i)
         x2(i) = x1(i)
         x3(i) = x1(i)
         x4(i) = x1(i)
      ENDDO
      CLOSE (kin)

*=  over 345.01 - 830.00, use values from Brion, room temperature only
* skip datum at 345.00 because already read in from 1995Malicet

      OPEN(UNIT=kin,FILE='DATAE1/O3/1998Brion_295.txt',STATUS='old')
      DO i = 1, 15
         READ(kin,*)
      ENDDO
      DO i = 1, 48515-15
         n1 = n1 + 1
         READ(kin,*) x1(n1), y1(n1)
      ENDDO
      CLOSE (kin)

      DO i = 1, n1
         ri(i) = refrac(x1(i), 2.45E19)
      ENDDO
      DO i = 1, n1
         x1(i) = x1(i) * ri(i)
      ENDDO

      CALL addpnt(x1,y1,kdata,n1,x1(1)*(1.-deltax),0.)
      CALL addpnt(x1,y1,kdata,n1,               0.,0.)
      CALL addpnt(x1,y1,kdata,n1,x1(n1)*(1.+deltax),0.)
      CALL addpnt(x1,y1,kdata,n1,            1.e+38,0.)
      CALL inter2(nw,wl,rei295,n1,x1,y1,ierr)
      IF (ierr .NE. 0) THEN
         WRITE(*,*) ierr, 'O3 xsect - Reims 295K'
         STOP
      ENDIF

      DO i = 1, n2
         ri(i) = refrac(x2(i), 2.45E19)
      ENDDO
      DO i = 1, n2
         x2(i) = x2(i) * ri(i)
         x3(i) = x2(i)
         x4(i) = x2(i)
      ENDDO

      CALL addpnt(x2,y2,kdata,n2,x2(1)*(1.-deltax),0.)
      CALL addpnt(x2,y2,kdata,n2,               0.,0.)
      CALL addpnt(x2,y2,kdata,n2,x2(n2)*(1.+deltax),0.)
      CALL addpnt(x2,y2,kdata,n2,            1.e+38,0.)
      CALL inter2(nw,wl,rei243,n2,x2,y2,ierr)
      IF (ierr .NE. 0) THEN
         WRITE(*,*) ierr, 'O3 xsect - Reims 243K'
         STOP
      ENDIF

      CALL addpnt(x3,y3,kdata,n3,x3(1)*(1.-deltax),0.)
      CALL addpnt(x3,y3,kdata,n3,               0.,0.)
      CALL addpnt(x3,y3,kdata,n3,x3(n3)*(1.+deltax),0.)
      CALL addpnt(x3,y3,kdata,n3,            1.e+38,0.)
      CALL inter2(nw,wl,rei228,n3,x3,y3,ierr)
      IF (ierr .NE. 0) THEN
         WRITE(*,*) ierr, 'O3 xsect - Reims 228K'
         STOP
      ENDIF

      CALL addpnt(x4,y4,kdata,n4,x4(1)*(1.-deltax),0.)
      CALL addpnt(x4,y4,kdata,n4,               0.,0.)
      CALL addpnt(x4,y4,kdata,n4,x4(n4)*(1.+deltax),0.)
      CALL addpnt(x4,y4,kdata,n4,            1.e+38,0.)
      CALL inter2(nw,wl,rei218,n4,x4,y4,ierr)
      IF (ierr .NE. 0) THEN
         WRITE(*,*) ierr, 'O3 xsect - Reims 218K'
         STOP
      ENDIF

* wavelength breaks must be converted to vacuum:

      v195 = 195.00 * refrac(195.00, 2.45E19)
      v345 = 345.00 * refrac(345.00, 2.45E19)
      v830 = 830.00 * refrac(830.00, 2.45E19)

      RETURN
      END

*=============================================================================*

      SUBROUTINE o3_wmo(nw,wl, wmo203,wmo273, v176,v850)

*-----------------------------------------------------------------------------*
*=  PURPOSE:                                                                 =*
*=  Read and interpolate the O3 cross section                                =*
*=  data from WMO 85 Ozone Assessment                                        =*
*-----------------------------------------------------------------------------*
*=  PARAMETERS:                                                              =*
*=  NW     - INTEGER, number of specified intervals + 1 in working        (I)=*
*=           wavelength grid                                                 =*
*=  WL     - REAL, vector of lower limits of wavelength intervals in      (I)=*
*=           working wavelength grid                                         =*
*=  WMO203 - REAL, cross section (cm^2) for O3 at 203K                    (O)=*
*=  WMO273 - REAL, cross section (cm^2) for O3 at 273K                    (O)=*
*=  V176   - REAL, exact wavelength in vacuum for data breaks             (O)=*
*=              e.g. start, stop, or other change                            =*
*=  V850   - REAL, exact wavelength in vacuum for data breaks             (O)=*
*-----------------------------------------------------------------------------*
      use tuv_params
      IMPLICIT NONE

*  input

      INTEGER nw, iw
      REAL wl(kw)

* internal

      INTEGER kdata
      PARAMETER (kdata = 200)

      INTEGER n1, n2
      REAL x1(kdata), x2(kdata)
      REAL y1(kdata), y2(kdata)

      INTEGER i, idum
      REAL a1, a2, dum
      INTEGER ierr

* used for air-to-vacuum wavelength conversion

      REAL refrac, ri(kdata)
      EXTERNAL refrac

* output

      REAL wmo203(kw), wmo273(kw)
      REAL v176, v850

*----------------------------------------------------------
* cross sections from WMO 1985 Ozone Assessment
* from 175.439 to 847.500 nm

      OPEN(UNIT=kin,FILE='DATAE1/wmo85',STATUS='old')
      DO i = 1, 3
         read(kin,*)
      ENDDO
      n1 = 158
      n2 = 158
      DO i = 1, n1
         READ(kin,*) idum, a1, a2, dum, dum, dum, y1(i), y2(i)
         x1(i) = (a1+a2)/2.
         x2(i) = (a1+a2)/2.
      ENDDO
      CLOSE (kin)

* convert wavelengths to vacuum

      DO i = 1, n1
         ri(i) = refrac(x1(i), 2.45E19)
      ENDDO
      DO i = 1, n1
         x1(i) = x1(i) * ri(i)
         x2(i) = x2(i) * ri(i)
      ENDDO

      CALL addpnt(x1,y1,kdata,n1,x1(1)*(1.-deltax),0.)
      CALL addpnt(x1,y1,kdata,n1,               0.,0.)
      CALL addpnt(x1,y1,kdata,n1,x1(n1)*(1.+deltax),0.)
      CALL addpnt(x1,y1,kdata,n1,           1.e+38,0.)
      CALL inter2(nw,wl,wmo203,n1,x1,y1,ierr)
      IF (ierr .NE. 0) THEN
         WRITE(*,*) ierr, 'O3 cross section - WMO - 203K'
         STOP
      ENDIF

      CALL addpnt(x2,y2,kdata,n2,x2(1)*(1.-deltax),0.)
      CALL addpnt(x2,y2,kdata,n2,               0.,0.)
      CALL addpnt(x2,y2,kdata,n2,x2(n2)*(1.+deltax),0.)
      CALL addpnt(x2,y2,kdata,n2,           1.e+38,0.)
      CALL inter2(nw,wl,wmo273,n2,x2,y2,ierr)
      IF (ierr .NE. 0) THEN
         WRITE(*,*) ierr, 'O3 cross section - WMO - 273K'
         STOP
      ENDIF

* wavelength breaks must be converted to vacuum:
      
      a1 = (175.438 + 176.991) / 2.
      v176 = a1 * refrac(a1,2.45E19)

      a1 = (847.5 + 852.5) / 2.
      v850 = a1 * refrac(a1, 2.45E19)

      RETURN
      END

*=============================================================================*

      SUBROUTINE o3_jpl(nw,wl, jpl218,jpl295, v186,v825)

*-----------------------------------------------------------------------------*
*=  PURPOSE:                                                                 =*
*=  Read and interpolate the O3 cross section from JPL 2006                  =*
*-----------------------------------------------------------------------------*
*=  PARAMETERS:                                                              =*
*=  NW     - INTEGER, number of specified intervals + 1 in working        (I)=*
*=           wavelength grid                                                 =*
*=  WL     - REAL, vector of lower limits of wavelength intervals in      (I)=*
*=           working wavelength grid                                         =*
*=  JPL218 - REAL, cross section (cm^2) for O3 at 218K                    (O)=*
*=  JPL295 - REAL, cross section (cm^2) for O3 at 295K                    (O)=*
*=  V186   - REAL, exact wavelength in vacuum for data breaks             (O)=*
*=              e.g. start, stop, or other change                            =*
*=  V825   - REAL, exact wavelength in vacuum for data breaks             (O)=*
*-----------------------------------------------------------------------------*
      use tuv_params
      IMPLICIT NONE

*  input

      INTEGER nw, iw
      REAL wl(kw)

* internal

      INTEGER kdata
      PARAMETER (kdata = 200)

      INTEGER n1, n2
      REAL x1(kdata), x2(kdata)
      REAL y1(kdata), y2(kdata)

      INTEGER i
      REAL dum
      INTEGER ierr

* used for air-to-vacuum wavelength conversion

      REAL refrac, ri(kdata)
      EXTERNAL refrac

* output

      REAL jpl295(kw), jpl218(kw)
      REAL v186, v825

***********

      OPEN(UNIT=kin,FILE='DATAE1/O3/2006JPL_O3.txt',STATUS='old')
      DO i = 1, 2
         read(kin,*)
      ENDDO
      n1 = 167
      n2 = 167
      DO i = 1, n1
         READ(kin,*) dum, dum, x1(i), y1(i), y2(i)
         y1(i) = y1(i) * 1.e-20
         y2(i) = y2(i) * 1.e-20
      ENDDO
      CLOSE (kin)

* convert wavelengths to vacuum

      DO i = 1, n1
         ri(i) = refrac(x1(i), 2.45E19)
      ENDDO
      DO i = 1, n1
         x1(i) = x1(i) * ri(i)
         x2(i) = x1(i)
      ENDDO

      CALL addpnt(x1,y1,kdata,n1,x1(1)*(1.-deltax),0.)
      CALL addpnt(x1,y1,kdata,n1,               0.,0.)
      CALL addpnt(x1,y1,kdata,n1,x1(n1)*(1.+deltax),0.)
      CALL addpnt(x1,y1,kdata,n1,           1.e+38,0.)
      CALL inter2(nw,wl,jpl295,n1,x1,y1,ierr)
      IF (ierr .NE. 0) THEN
         WRITE(*,*) ierr, 'O3 cross section - WMO - 295K'
         STOP
      ENDIF

      CALL addpnt(x2,y2,kdata,n2,x2(1)*(1.-deltax),0.)
      CALL addpnt(x2,y2,kdata,n2,               0.,0.)
      CALL addpnt(x2,y2,kdata,n2,x2(n2)*(1.+deltax),0.)
      CALL addpnt(x2,y2,kdata,n2,           1.e+38,0.)
      CALL inter2(nw,wl,jpl218,n2,x2,y2,ierr)
      IF (ierr .NE. 0) THEN
         WRITE(*,*) ierr, 'O3 cross section - WMO - 218K'
         STOP
      ENDIF

* wavelength breaks must be converted to vacuum:

      v186 = 186.051 * refrac(186.051, 2.45E19)
      v825 = 825.    * refrac(825.   , 2.45E19)


      RETURN
      END


*=============================================================================*

      SUBROUTINE o3_mol(nw,wl, mol226,mol263,mol298, v185,v240,v350)

*-----------------------------------------------------------------------------*
*=  PURPOSE:                                                                 =*
*=  Read and interpolate the O3 cross section from Molina and Molina 1986    =*
*-----------------------------------------------------------------------------*
*=  PARAMETERS:                                                              =*
*=  NW     - INTEGER, number of specified intervals + 1 in working        (I)=*
*=           wavelength grid                                                 =*
*=  WL     - REAL, vector of lower limits of wavelength intervals in      (I)=*
*=           working wavelength grid                                         =*
*=  MOL226 - REAL, cross section (cm^2) for O3 at 226 K                   (O)=*
*=  MOL263 - REAL, cross section (cm^2) for O3 at 263 K                   (O)=*
*=  MOL298 - REAL, cross section (cm^2) for O3 at 298 K                   (O)=*
*=  V185   - REAL, exact wavelength in vacuum for data breaks             (O)=*
*=              e.g. start, stop, or other change                            =*
*=  V240   - REAL, exact wavelength in vacuum for data breaks             (O)=*
*=  V350   - REAL, exact wavelength in vacuum for data breaks             (O)=*
*-----------------------------------------------------------------------------*
      use tuv_params
      IMPLICIT NONE

*  input

      INTEGER nw, iw
      REAL wl(kw)

* internal

      INTEGER i
      INTEGER ierr

      INTEGER kdata
      PARAMETER (kdata = 335)
      INTEGER n1, n2, n3
      REAL x1(kdata), x2(kdata), x3(kdata)
      REAL y1(kdata), y2(kdata), y3(kdata)

* used for air-to-vacuum wavelength conversion

      REAL refrac, ri(kdata)
      EXTERNAL refrac

* output

      REAL mol226(kw), mol263(kw), mol298(kw)
      REAL v185, v240, v350

*----------------------------------------------------------

      OPEN(UNIT=kin,FILE='DATAE1/O3/1986Molina.txt',STATUS='old')
      DO i = 1, 10
         READ(kin,*)
      ENDDO
      n1 = 0
      n2 = 0
      n3 = 0
      DO i = 1, 121-10
         n1 = n1 + 1
         n3 = n3 + 1
         READ(kin,*) x1(n1), y1(n1),  y3(n3)
         x3(n3) = x1(n1)
      ENDDO
      DO i = 1, 341-122
         n1 = n1 + 1
         n2 = n2 + 1
         n3 = n3 + 1
         READ(kin,*) x1(n1), y1(n1), y2(n2), y3(n3)
         x2(n2) = x1(n1)
         x3(n3) = x1(n1)
      ENDDO
      CLOSE (kin)

* convert all wavelengths from air to vacuum

      DO i = 1, n1
         ri(i) = refrac(x1(i), 2.45E19)
      ENDDO
      DO i = 1, n1
         x1(i) = x1(i) * ri(i)
      ENDDO

      DO i = 1, n2
         ri(i) = refrac(x2(i), 2.45E19)
      ENDDO
      DO i = 1, n2
         x2(i) = x2(i) * ri(i)
      ENDDO

      DO i = 1, n3
         ri(i) = refrac(x3(i), 2.45E19)
      ENDDO
      DO i = 1, n3
         x3(i) = x3(i) * ri(i)
      ENDDO

* convert wavelength breaks from air to vacuum

      v185 = 185.  * refrac(185. , 2.45E19)
      v240 = 240.5 * refrac(240.5, 2.45E19)
      v350 = 350.  * refrac(350. , 2.45E19)

* interpolate to working grid

      CALL addpnt(x1,y1,kdata,n1,x1(1)*(1.-deltax),0.)
      CALL addpnt(x1,y1,kdata,n1,               0.,0.)
      CALL addpnt(x1,y1,kdata,n1,x1(n1)*(1.+deltax),0.)
      CALL addpnt(x1,y1,kdata,n1,            1.e+38,0.)
      CALL inter2(nw,wl,mol226,n1,x1,y1,ierr)
      IF (ierr .NE. 0) THEN
         WRITE(*,*) ierr, 'O3 xsect - 226K Molina'
         STOP
      ENDIF

      CALL addpnt(x2,y2,kdata,n2,x2(1)*(1.-deltax),0.)
      CALL addpnt(x2,y2,kdata,n2,               0.,0.)
      CALL addpnt(x2,y2,kdata,n2,x2(n2)*(1.+deltax),0.)
      CALL addpnt(x2,y2,kdata,n2,            1.e+38,0.)
      CALL inter2(nw,wl,mol263,n2,x2,y2,ierr)
      IF (ierr .NE. 0) THEN
         WRITE(*,*) ierr, 'O3 xsect - 263K Molina'
         STOP
      ENDIF

      CALL addpnt(x3,y3,kdata,n3,x3(1)*(1.-deltax),0.)
      CALL addpnt(x3,y3,kdata,n3,               0.,0.)
      CALL addpnt(x3,y3,kdata,n3,x3(n3)*(1.+deltax),0.)
      CALL addpnt(x3,y3,kdata,n3,            1.e+38,0.)
      CALL inter2(nw,wl,mol298,n3,x3,y3,ierr)
      IF (ierr .NE. 0) THEN
         WRITE(*,*) ierr, 'O3 xsect - 298K Molina'
         STOP
      ENDIF

      RETURN
      END

*=============================================================================*

      SUBROUTINE o3_bas(nw,wl, c0,c1,c2, vb245,vb342)

*-----------------------------------------------------------------------------*
*=  PURPOSE:                                                                 =*
*=  Read and interpolate the O3 cross section from Bass 1985                 =*
*-----------------------------------------------------------------------------*
*=  PARAMETERS:                                                              =*
*=  NW     - INTEGER, number of specified intervals + 1 in working        (I)=*
*=           wavelength grid                                                 =*
*=  WL     - REAL, vector of lower limits of wavelength intervals in      (I)=*
*=           working wavelength grid                                         =*
*=  c0     - REAL, coefficint for polynomial fit to cross section (cm^2)  (O)=*
*=  c1     - REAL, coefficint for polynomial fit to cross section (cm^2)  (O)=*
*=  c2     - REAL, coefficint for polynomial fit to cross section (cm^2)  (O)=*
*=  Vb245   - REAL, exact wavelength in vacuum for data breaks            (O)=*
*=              e.g. start, stop, or other change                            =*
*=  Vb342   - REAL, exact wavelength in vacuum for data breaks            (O)=*
*-----------------------------------------------------------------------------*
      use tuv_params
      IMPLICIT NONE

* input:

      INTEGER nw, iw
      REAL wl(kw)

* internal:

      INTEGER kdata
      PARAMETER (kdata = 2000)

      INTEGER i
      INTEGER ierr

      INTEGER n1, n2, n3
      REAL x1(kdata), x2(kdata), x3(kdata)
      REAL y1(kdata), y2(kdata), y3(kdata)

* used for air-to-vacuum wavelength conversion

      REAL refrac, ri(kdata)
      EXTERNAL refrac

* output:

      REAL c0(kw), c1(kw), c2(kw)
      REAL vb245, vb342

*******************

      OPEN(UNIT=kin,FILE='DATAE1/O3/1985Bass_O3.txt',STATUS='old')
      DO i = 1, 8
         READ(kin,*)
      ENDDO
      n1 = 1915
      n2 = 1915
      n3 = 1915
      DO i = 1, n1
         READ(kin,*) x1(i), y1(i), y2(i), y3(i)
         y1(i) = 1.e-20 * y1(i)
         y2(i) = 1.e-20 * y2(i)
         y3(i) = 1.e-20 * y3(i)
      ENDDO
      CLOSE (kin)

* convert all wavelengths from air to vacuum

      DO i = 1, n1
         ri(i) = refrac(x1(i), 2.45E19)
      ENDDO
      DO i = 1, n1
         x1(i) = x1(i) * ri(i)
         x2(i) = x1(i)
         x3(i) = x1(i)
      ENDDO

* convert wavelength breaks to vacuum

      vb245 = 245.018 * refrac(245.018, 2.45E19)
      vb342 = 341.981 * refrac(341.981, 2.45E19)

* interpolate to working grid

      CALL addpnt(x1,y1,kdata,n1,x1(1)*(1.-deltax),0.)
      CALL addpnt(x1,y1,kdata,n1,               0.,0.)
      CALL addpnt(x1,y1,kdata,n1,x1(n1)*(1.+deltax),0.)
      CALL addpnt(x1,y1,kdata,n1,            1.e+38,0.)
      CALL inter2(nw,wl,c0,n1,x1,y1,ierr)
      IF (ierr .NE. 0) THEN
         WRITE(*,*) ierr, 'O3 xsect - c0 Bass'
         STOP
      ENDIF

      CALL addpnt(x2,y2,kdata,n2,x2(1)*(1.-deltax),0.)
      CALL addpnt(x2,y2,kdata,n2,               0.,0.)
      CALL addpnt(x2,y2,kdata,n2,x2(n2)*(1.+deltax),0.)
      CALL addpnt(x2,y2,kdata,n2,            1.e+38,0.)
      CALL inter2(nw,wl,c1,n2,x2,y2,ierr)
      IF (ierr .NE. 0) THEN
         WRITE(*,*) ierr, 'O3 xsect - c1 Bass'
         STOP
      ENDIF

      CALL addpnt(x3,y3,kdata,n3,x3(1)*(1.-deltax),0.)
      CALL addpnt(x3,y3,kdata,n3,               0.,0.)
      CALL addpnt(x3,y3,kdata,n3,x3(n3)*(1.+deltax),0.)
      CALL addpnt(x3,y3,kdata,n3,            1.e+38,0.)
      CALL inter2(nw,wl,c2,n3,x3,y3,ierr)
      IF (ierr .NE. 0) THEN
         WRITE(*,*) ierr, 'O3 xsect - c2 Bass'
         STOP
      ENDIF

      RETURN
      END

*=============================================================================*

      SUBROUTINE rdo2xs(nw,wl,o2xs1)

*-----------------------------------------------------------------------------*
*=  PURPOSE:                                                                 =*
*=  Compute equivalent O2 cross section, except                              =*
*=  the SR bands and the Lyman-alpha line.                                   =*
*-----------------------------------------------------------------------------* 
*=  PARAMETERS:                                   
*=  NW      - INTEGER, number of specified intervals + 1 in working       (I)=*
*=            wavelength grid                                                =*
*=  WL      - REAL, vector of lower limits of wavelength intervals in     (I)=*
*=            working wavelength grid           
*=            vertical layer at each specified wavelength                    =*
*=  O2XS1   - REAL, O2 molecular absorption cross section                    =*
*=
*-----------------------------------------------------------------------------*
      use tuv_params
      IMPLICIT NONE

* Input

      INTEGER nw
      REAL wl(kw)

* Output O2 xsect, temporary, will be over-written in Lyman-alpha and 
*   Schumann-Runge wavelength bands.

      REAL o2xs1(kw)

* Internal

      INTEGER i, n, kdata
      PARAMETER (kdata = 200)
      REAL x1(kdata), y1(kdata)
      REAL x, y
      INTEGER ierr

*-----------------------------------------------------

* Read O2 absorption cross section data:
*  116.65 to 203.05 nm = from Brasseur and Solomon 1986
*  205 to 240 nm = Yoshino et al. 1988

* Note that subroutine la_srb.f will over-write values in the spectral regions
*   corresponding to:
* - Lyman-alpha (LA: 121.4-121.9 nm, Chabrillat and Kockarts parameterization) 
* - Schumann-Runge bands (SRB: 174.4-205.8 nm, Koppers parameteriaztion)

      n = 0

      OPEN(UNIT=kin,FILE='DATAE1/O2/O2_brasseur.abs')
      DO i = 1, 7
         READ(kin,*)
      ENDDO
      DO i = 1, 78
         READ(kin,*) x, y
         IF (x .LE. 204.) THEN
            n = n + 1
            x1(n) = x
            y1(n) = y
         ENDIF
      ENDDO
      CLOSE(kin)

      OPEN(UNIT=kin,FILE='DATAE1/O2/O2_yoshino.abs',STATUS='old')
      DO i = 1, 8
         READ(kin,*)
      ENDDO
      DO i = 1, 36
         n = n + 1
         READ(kin,*) x, y
         y1(n) = y*1.E-24
         x1(n) = x
      END DO
      CLOSE (kin)

* Add termination points and interpolate onto the 
*  user grid (set in subroutine gridw):

      CALL addpnt(x1,y1,kdata,n,x1(1)*(1.-deltax),y1(1))
      CALL addpnt(x1,y1,kdata,n,0.               ,y1(1))
      CALL addpnt(x1,y1,kdata,n,x1(n)*(1.+deltax),0.)
      CALL addpnt(x1,y1,kdata,n,              1.E+38,0.)
      CALL inter2(nw,wl,o2xs1, n,x1,y1, ierr)
      IF (ierr .NE. 0) THEN
         WRITE(*,*) ierr, 'O2 -> O + O'
         STOP
      ENDIF

*------------------------------------------------------

      RETURN
      END

*=============================================================================*

      SUBROUTINE rdno2xs(nz,tlay,nw,wl,no2xs)

*-----------------------------------------------------------------------------*
*=  PURPOSE:                                                                 =*
*=  Read NO2 molecular absorption cross section.  Re-grid data to match      =*
*=  specified wavelength working grid.                                       =*
*-----------------------------------------------------------------------------*
*=  PARAMETERS:                                                              =*
*=  NW     - INTEGER, number of specified intervals + 1 in working        (I)=*
*=           wavelength grid                                                 =*
*=  WL     - REAL, vector of lower limits of wavelength intervals in      (I)=*
*=           working wavelength grid                                         =*
*=  NO2XS  - REAL, molecular absoprtion cross section (cm^2) of NO2 at    (O)=*
*=           each specified wavelength                                       =*
*-----------------------------------------------------------------------------*
      use tuv_params
      IMPLICIT NONE

* input: (altitude working grid)

      INTEGER nz
      REAL tlay(kz)

      INTEGER nw
      REAL wl(kw)

      INTEGER mabs

* output:

      REAL no2xs(kz,kw)

*_______________________________________________________________________

* options for NO2 cross section:
* 1 = Davidson et al. (1988), indepedent of T
* 2 = JPL 1994 (same as JPL 1997, JPL 2002)
* 3 = Harder et al.
* 4 = JPL 2006, interpolating between midpoints of bins
* 5 = JPL 2006, bin-to-bin interpolation

      mabs = 4

      IF (mabs. EQ. 1) CALL no2xs_d(nz,tlay,nw,wl, no2xs)
      IF (mabs .EQ. 2) CALL no2xs_jpl94(nz,tlay,nw,wl, no2xs)
      IF (mabs .EQ. 3) CALL no2xs_har(nz,tlay,nw,wl, no2xs)
      IF (mabs .EQ. 4) CALL no2xs_jpl06a(nz,tlay,nw,wl, no2xs)
      IF (mabs .EQ. 5) CALL no2xs_jpl06b(nz,tlay,nw,wl, no2xs)

*_______________________________________________________________________

      RETURN
      END

*=============================================================================*

      SUBROUTINE no2xs_d(nz,t,nw,wl, no2xs)
      use tuv_params
      IMPLICIT NONE

* read and interpolate NO2 xs from Davidson et al. (1988).

* input:

      INTEGER nz, iz
      REAL t(nz)

      INTEGER nw, iw
      REAL wl(nw)

* output:

      REAL no2xs(kz,kw)

* local:

      INTEGER kdata
      PARAMETER (kdata=1000)
      REAL x1(kdata)
      REAL y1(kdata)
      REAL yg(kw)
      REAL dum
      INTEGER ierr
      INTEGER i, n, idum
      CHARACTER*40 fil

************* NO2 absorption cross sections
*     measurements by:
* Davidson, J. A., C. A. Cantrell, A. H. McDaniel, R. E. Shetter,
* S. Madronich, and J. G. Calvert, Visible-ultraviolet absorption
* cross sections for NO2 as a function of temperature, J. Geophys.
* Res., 93, 7105-7112, 1988.
*  Values at 273K from 263.8 to 648.8 nm in approximately 0.5 nm intervals

      fil = 'DATAE1/NO2/NO2_ncar_00.abs'
      OPEN(UNIT=kin,FILE=fil,STATUS='old')
      n = 750
      DO i = 1, n
         READ(kin,*) x1(i), y1(i), dum, dum, idum
      ENDDO
      CLOSE(kin)

* interpolate to wavelength grid

      CALL addpnt(x1,y1,kdata,n,x1(1)*(1.-deltax),0.)
      CALL addpnt(x1,y1,kdata,n,          0.,0.)
      CALL addpnt(x1,y1,kdata,n,x1(n)*(1.+deltax),0.)
      CALL addpnt(x1,y1,kdata,n,      1.e+38,0.)
      CALL inter2(nw,wl,yg,n,x1,y1,ierr)
      IF (ierr .NE. 0) THEN
         WRITE(*,*) ierr, fil
         STOP
      ENDIF

* assign, same at all altitudes (no temperature effect)     

      DO iz = 1, nz
         DO iw = 1, nw-1
            no2xs(iz,iw) = yg(iw)
         ENDDO
      ENDDO

*_______________________________________________________________________

      RETURN
      END

*=============================================================================*

      SUBROUTINE no2xs_jpl94(nz,t,nw,wl, no2xs)
      use tuv_params
      IMPLICIT NONE

* read and interpolate NO2 xs from JPL 1994

* input:

      INTEGER nz, iz
      REAL t(nz)

      INTEGER nw, iw
      REAL wl(nw)

* output:

      REAL no2xs(kz,kw)

* local:

      INTEGER kdata
      PARAMETER (kdata=100)
      INTEGER i, idum, n, n1
      REAL x1(kdata), x2(kdata), x3(kdata)
      REAL y1(kdata), y2(kdata), y3(kdata)
      REAL dum
      REAL yg1(nw), yg2(nw)

* cross section data from JPL 94 recommendation
* JPL 97 and JPL 2002 recommendations are identical

      OPEN(UNIT=kin,FILE='DATAE1/NO2/NO2_jpl94.abs',STATUS='old')
      READ(kin,*) idum, n
      DO i = 1, idum-2
         READ(kin,*)
      ENDDO 

* read in wavelength bins, cross section at T0 and temperature correction
* coefficient a;  see input file for details.
* data need to be scaled to total area per bin so that they can be used with
* inter3

      DO i = 1, n
         READ(kin,*) x1(i), x3(i), y1(i), dum, y2(i)
         y1(i) = (x3(i)-x1(i)) * y1(i)*1.E-20
         y2(i) = (x3(i)-x1(i)) * y2(i)*1.E-22
         x2(i) = x1(i) 
      ENDDO
      CLOSE(kin)

      x1(n+1) = x3(n)
      x2(n+1) = x3(n)
      n = n+1
      n1 = n

      CALL inter3(nw,wl,yg1,n,x1,y1,0)
      CALL inter3(nw,wl,yg2,n1,x2,y2,0)

* yg1, yg2 are per nm, so rescale by bin widths

      DO iw = 1, nw-1
         yg1(iw) = yg1(iw)/(wl(iw+1)-wl(iw))
         yg2(iw) = yg2(iw)/(wl(iw+1)-wl(iw))
      ENDDO

      DO iw = 1, nw-1
         DO iz = 1, nz
            no2xs(iz,iw) = yg1(iw) + yg2(iw)*(t(iz)-273.15)
         ENDDO
      ENDDO 

      RETURN
      END

*=============================================================================*

      SUBROUTINE no2xs_har(nz,t,nw,wl, no2xs)
      use tuv_params
      IMPLICIT NONE

* read and interpolate NO2 xs from Harder et al.

* input:

      INTEGER nz, iz
      REAL t(nz)

      INTEGER nw, iw
      REAL wl(nw)

* output:

      REAL no2xs(kz,kw)

* local:

      INTEGER kdata
      PARAMETER (kdata=150)
      INTEGER i, n, idum, ierr
      REAL x1(kdata), y1(kdata)
      REAL yg1(kw)

***

      OPEN(UNIT=kin,FILE='DATAE1/NO2/NO2_Har.abs',status='old')
      DO i = 1, 9
         READ(kin,*)
      ENDDO
      n = 135
      DO i = 1, n
         READ(kin,*) idum, y1(i)
         x1(i) = FLOAT(idum)
      ENDDO

      CALL addpnt(x1,y1,kdata,n,x1(1)*(1.-deltax),y1(1))
      CALL addpnt(x1,y1,kdata,n,               0.,y1(1))
      CALL addpnt(x1,y1,kdata,n,x1(n)*(1.+deltax),   0.)
      CALL addpnt(x1,y1,kdata,n,           1.e+38,   0.)
      CALL inter2(nw,wl,yg1,n,x1,y1,ierr)

      DO iw = 1, nw-1
         DO i = 1, nz
            no2xs(i,iw) = yg1(iw)
         ENDDO
      ENDDO 

      RETURN
      END

*=============================================================================*

      SUBROUTINE no2xs_jpl06a(nz,t,nw,wl, no2xs)

* read and interpolate NO2 xs from JPL2006
      use tuv_params
      IMPLICIT NONE

* input:

      INTEGER nz, iz
      REAL t(nz)

      INTEGER nw, iw
      REAL wl(nw)

* output:

      REAL no2xs(kz,kw)

* local

      INTEGER kdata
      PARAMETER (kdata=100)
      INTEGER i, n1, n2, ierr
      REAL x1(kdata), x2(kdata), y1(kdata), y2(kdata)
      REAL dum1, dum2
      REAL yg1(kw), yg2(kw)

* NO2 absorption cross section from JPL2006
* with interpolation of bin midpoints

      OPEN(UNIT=kin,FILE='DATAE1/NO2/NO2_jpl2006.abs',STATUS='old')
      DO i = 1, 3
         READ(kin,*)
      ENDDO 
      n1 = 81
      DO i = 1, n1
         READ(kin,*) dum1, dum2, y1(i), y2(i)
         x1(i) = 0.5 * (dum1 + dum2)
         x2(i) = x1(i) 
         y1(i) = y1(i)*1.E-20
         y2(i) = y2(i)*1.E-20
      ENDDO
      CLOSE(kin)
      n2 = n1

      CALL addpnt(x1,y1,kdata,n1,x1(1)*(1.-deltax),0.)
      CALL addpnt(x1,y1,kdata,n1,               0.,0.)
      CALL addpnt(x1,y1,kdata,n1,x1(n1)*(1.+deltax),   0.)
      CALL addpnt(x1,y1,kdata,n1,            1.e+38,   0.)
      CALL inter2(nw,wl,yg1,n1,x1,y1,ierr)
      
      CALL addpnt(x2,y2,kdata,n2,x2(1)*(1.-deltax),0.)
      CALL addpnt(x2,y2,kdata,n2,               0.,0.)
      CALL addpnt(x2,y2,kdata,n2,x2(n2)*(1.+deltax),   0.)
      CALL addpnt(x2,y2,kdata,n2,            1.e+38,   0.)
      CALL inter2(nw,wl,yg2,n2,x2,y2,ierr)
      
      DO iw = 1, nw-1
         DO iz = 1, nz
            no2xs(iz,iw) = yg1(iw) + 
     $           (yg2(iw)-yg1(iw))*(t(iz)-220.)/74.
         ENDDO
      ENDDO 

      RETURN
      END

*=============================================================================*

      SUBROUTINE no2xs_jpl06b(nz,t,nw,wl, no2xs)

* read and interpolate NO2 xs from Harder et al.
      use tuv_params
      IMPLICIT NONE

* input:

      INTEGER nz, iz
      REAL t(nz)

      INTEGER nw, iw
      REAL wl(nw)

* output:

      REAL no2xs(kz,kw)

* local

      INTEGER kdata
      PARAMETER (kdata=100)
      INTEGER i, n, n1, n2, ierr
      REAL x1(kdata), x2(kdata), y1(kdata), y2(kdata)
      REAL x3(kdata), y3(kdata)
      REAL dum1, dum2
      REAL yg1(kw), yg2(kw)

      OPEN(UNIT=kin,FILE='DATAE1/NO2/NO2_jpl2006.abs',STATUS='old')
      DO i = 1, 3
         READ(kin,*)
      ENDDO 
      n = 81
      do i = 1, n
         read(kin,*) x1(i), x3(i), y1(i), y2(i)
         y1(i) = (x3(i)-x1(i)) * y1(i)*1.E-20
         y2(i) = (x3(i)-x1(i)) * y2(i)*1.E-20
         x2(i) = x1(i) 
      ENDDO
      CLOSE(kin)
         
      x1(n+1) = x3(n)
      x2(n+1) = x3(n)
      n = n+1
      n1 = n

      CALL inter3(nw,wl,yg1,n,x1,y1,0)
      CALL inter3(nw,wl,yg2,n1,x2,y2,0)

* yg1, yg2 are per nm, so rescale by bin widths

      DO iw = 1, nw-1
         yg1(iw) = yg1(iw)/(wl(iw+1)-wl(iw))
         yg2(iw) = yg2(iw)/(wl(iw+1)-wl(iw))
      ENDDO

      DO iw = 1, nw-1
         DO iz = 1, nz
            no2xs(iz,iw) = yg1(iw) + 
     $           (yg2(iw)-yg1(iw))*(t(iz)-220.)/74.
         ENDDO

      ENDDO 

      RETURN
      END

*=============================================================================*

      SUBROUTINE rdso2xs(nw,wl,so2xs)

*-----------------------------------------------------------------------------*
*=  PURPOSE:                                                                 =*
*=  Read SO2 molecular absorption cross section.  Re-grid data to match      =*
*=  specified wavelength working grid.                                       =*
*-----------------------------------------------------------------------------*
*=  PARAMETERS:                                                              =*
*=  NW     - INTEGER, number of specified intervals + 1 in working        (I)=*
*=           wavelength grid                                                 =*
*=  WL     - REAL, vector of lower limits of wavelength intervals in      (I)=*
*=           working wavelength grid                                         =*
*=  SO2XS  - REAL, molecular absoprtion cross section (cm^2) of SO2 at    (O)=*
*=           each specified wavelength                                       =*
*-----------------------------------------------------------------------------*
*=  EDIT HISTORY:                                                            =*
*=  02/97  Changed offset for grid-end interpolation to relative number      =*
*=         (x * (1 +- deltax)                                                =*
*-----------------------------------------------------------------------------*
*= This program is free software;  you can redistribute it and/or modify     =*
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
*= To contact the authors, please mail to:                                   =*
*= Sasha Madronich, NCAR/ACD, P.O.Box 3000, Boulder, CO, 80307-3000, USA  or =*
*= send email to:  sasha@ucar.edu                                            =*
*-----------------------------------------------------------------------------*
      use tuv_params
      IMPLICIT NONE

      INTEGER kdata
      PARAMETER(kdata=1000)

* input: (altitude working grid)
      INTEGER nw
      REAL wl(kw)

* output:

      REAL so2xs(kw)

* local:
      REAL x1(kdata)
      REAL y1(kdata)
      REAL yg(kw)
      REAL dum
      INTEGER ierr
      INTEGER i, l, n, idum
      CHARACTER*40 fil
*_______________________________________________________________________

************* absorption cross sections:
* SO2 absorption cross sections from J. Quant. Spectrosc. Radiat. Transfer
* 37, 165-182, 1987, T. J. McGee and J. Burris Jr.
* Angstrom vs. cm2/molecule, value at 221 K

      fil = 'DATA/McGee87'
      OPEN(UNIT=kin,FILE='DATAE1/SO2/SO2xs.all',STATUS='old')
      DO 11, i = 1,3 
         read(kin,*)
   11 CONTINUE
c      n = 681 
      n = 704 
      DO 12, i = 1, n
         READ(kin,*) x1(i), y1(i)
         x1(i) = x1(i)/10.
   12 CONTINUE
      CLOSE (kin)

      CALL addpnt(x1,y1,kdata,n,x1(1)*(1.-deltax),0.)
      CALL addpnt(x1,y1,kdata,n,          0.,0.)
      CALL addpnt(x1,y1,kdata,n,x1(n)*(1.+deltax),0.)
      CALL addpnt(x1,y1,kdata,n,      1.e+38,0.)
      CALL inter2(nw,wl,yg,n,x1,y1,ierr)
      IF (ierr .NE. 0) THEN
         WRITE(*,*) ierr, fil
         STOP
      ENDIF
      
      DO 13, l = 1, nw-1
         so2xs(l) = yg(l)
   13 CONTINUE

*_______________________________________________________________________

      RETURN
      END

*=============================================================================*

