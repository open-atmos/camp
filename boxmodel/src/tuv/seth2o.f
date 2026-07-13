*** Currently not activated. sm/3nov2015
*=============================================================================*

      SUBROUTINE seth2o(ipbl, zpbl, xpbl, 
     $     h2onew, nz, z, nw, wl, h2oxs, 
     $     tlay, dcol,
     $     dth2o)

*-----------------------------------------------------------------------------*
*=  PURPOSE:                                                                 =*
*=  Set up an altitude profile of H2O molecules, and corresponding absorption=*
*=  optical depths.  Subroutine includes a shape-conserving scaling method   =*
*=  that allows scaling of the entire profile to a given overhead H2O        =*
*=  column amount.                                                           =*
*-----------------------------------------------------------------------------*
*=  PARAMETERS:                                                              =*
*=  H2ONEW - REAL, overhead H2O column amount (molec/cm^2) to which       (I)=*
*=           profile should be scaled.  If H2ONEW < 0, no scaling is done    =*
*=  NZ     - INTEGER, number of specified altitude levels in the working  (I)=*
*=           grid                                                            =*
*=  Z      - REAL, specified altitude working grid (km)                   (I)=*
*=  NW     - INTEGER, number of specified intervals + 1 in working        (I)=*
*=           wavelength grid                                                 =*
*=  WL     - REAL, vector of lower limits of wavelength intervals in      (I)=*
*=           working wavelength grid                                         =*
*=  H2OXS  - REAL, molecular absoprtion cross section (cm^2) of H2O at    (I)=*
*=           each specified wavelength                                       =*
*=  TLAY   - REAL, temperature (K) at each specified altitude layer       (I)=*
*=  DTH2O  - H2O optical depth  due to absorption at each                 (O)=*
*=           specified altitude at each specified wavelength                 =*
*-----------------------------------------------------------------------------*
      use tuv_params
      IMPLICIT NONE

      INTEGER kdata
      PARAMETER(kdata=51)

********
* input:
********

* grids:

      REAL wl(kw)
      REAL z(kz)
      INTEGER nw
      INTEGER nz
      REAL h2onew

* mid-layer temperature, layer air column

      REAL tlay(kz), dcol(kz)

********
* output:
********

      REAL dtreal dth2o(kz,kw)

********
* local:
********

* absorption cross sections 

      REAL h2oxs(kz,kw)
      REAL cz(kz)

* nitrogen dioxide profile data:

      REAL zd(kdata), h2o(kdata)
      REAL cd(kdata)
      REAL hscale
      REAL colold, scale
      REAL sh2o
      REAL zpbl, xpbl
      INTEGER ipbl

* other:

      INTEGER i, l, nd

********
* External functions:
********

      REAL fsum
      EXTERNAL fsum

*_______________________________________________________________________
* Data input:

* us standard atmsosphere, table 20:
* values in units parts per million by mass      
      integer kus,nus
      parameter(kus=10)
      data zus(kus)/0,1,2,4,6,8,10,12,14,16/
      data h2ous/4686,3700,2843,1268,554,216,43.2,11.3,3.3,3.3/

* write into working array:
      
      n = kus
      do i = 1, n
         x(i) = zus(i)
         y(i) = h2ous(i)
      enddo
      x(kus+1) = zus(n)*1.0001
      y(kus+1) = 0.
      x(kus+2) = wl(nw)*1.0001
      y(kus+2) = 0. 
      n = kus + 2

      SUBROUTINE inter1(nz,z,cz, n,x,y)

* convert from ppm by mass to molec cm-3:

      DO i = 1, nz
         h2o(i) = cz(i)* 1.e6 * dcol(i) * MWair/MWh2o
      ENDDO

* compute column increments (alternatively, can specify these directly)

      DO i = 1, nd - 1
         cd(i) = (h2o(i+1)+h2o(i)) * 1.E5 * (zd(i+1)-zd(i)) / 2. 
      ENDDO

* Include exponential tail integral from top level to infinity.
* fold tail integral into top layer
* specify scale height near top of data (use ozone value)

      hscale = 4.50e5
      cd(nd-1) = cd(nd-1) + hscale * h2o(nd)

***********
*********** end data input.

* Compute column increments and total column on standard z-grid.  

      CALL inter3(nz,z,cz, nd,zd,cd, 1)

**** Scaling of vertical profile by ratio of new to old column:
* If old column is near zero (less than 1 molec cm-2), 
* use constant mixing ratio profile (nominal 1 ppt before scaling) 
* to avoid numerical problems when scaling.

      IF(fsum(nz-1,cz) .LT. 1.) THEN
         DO i = 1, nz-1
            cz(i) = 1.E-12 * dcol(i)
         ENDDO
      ENDIF
      colold = fsum(nz-1, cz)
      scale =  2.687e16 * h2onew / colold

      DO i = 1, nz-1
         cz(i) = cz(i) * scale
      ENDDO

*! overwrite for specified pbl height

      IF(ipbl .GT. 0) THEN
         write(*,*) 'pbl H2O = ', xpbl, ' ppb'

         DO i = 1, nz-1
            IF (i .LE. ipbl) THEN
               cz(i) = xpbl*1.E-9 * dcol(i)
            ELSE
               cz(i) = 0.
            ENDIF
         ENDDO
      ENDIF

************************************
* calculate optical depth for each layer.  Output: dtno2(kz,kw)

98	continue
      DO 20, l = 1, nw-1
         DO 10, i = 1, nz-1
            dth2o(i,l) = cz(i)*h2oxs(i,l)
   10    CONTINUE
   20 CONTINUE
*_______________________________________________________________________

      RETURN
      END
