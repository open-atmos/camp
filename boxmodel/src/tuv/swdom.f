      SUBROUTINE swdom(nw,wl,wc, jdom,dlabel,sdom)

* by S.Madronich, Oct 2015.
* Purpose: load and w-grid the spectral absorption coefficient for waters (lakes, oceans)
* UNITS = 1/meter
* Can use either a generic equation (exponential decrease with wavelength)
* or measured DOM spectral absorption data from various groups and locations 
* e.g. Giles Lake from Craig Williamson, Priv comm. 2015.
      use tuv_params
      IMPLICIT NONE
      INTEGER nw, iw, i, n, n1, n2

      INTEGER jdom
      REAL sdom(kdom,kw)
      CHARACTER*50 dlabel(kdom)

      INTEGER kdata
      PARAMETER(kdata = 1000)
      REAL dum1, dum2
      INTEGER idum
      REAL x1(kdata), x2(kdata)
      REAL y1(kdata), y2(kdata)

      REAL wl(kw), wc(kw), yg1(kw), yg2(kw)
      INTEGER ierr

      INTEGER j
      REAL ydum(kdata,9)
      REAL xdum(kdata)

      CHARACTER*20 aname(kdom)
      REAL x(kdata), y(kdom,kdata) 

      INTEGER idate(100)

* switch for generic absorption equation, rather than individual measured spectra      

      LOGICAL lgener

      jdom = 0

* generic absorption coefficients for lake waters, from
*  Bricaud A., Morel A. and Prieur L., Absorption by dissolved 
* organic matter of teh sea (yellow substance) in the UV and 
* visible domains, Limnol. Oceanogr. 26, 43-53, 1981.
* The pre-exponential at 375 nmn is taken as 1.0/meter, estimated from the ranges 
* given in their Table 3: low 0.06, high 4.24, geometric mean = 0.5
* units are 1/meter

      lgener = .TRUE.
      if(lgener) then 
         
         jdom = jdom + 1
         dlabel(jdom) = 'Generic DOM absorption'

         DO iw = 1, nw-1
            sdom(jdom, iw) = 0.5 * exp(-0.015*(wc(iw) - 375.))
         ENDDO

      RETURN
      ENDIF

***** the remaining code below (enabled with lgener = .FALSE.) is to read individual 
* absorption data sets measured by various research groups.  These data are mostly 
* unpublished and therefore are not part of the standard TUV releases.  If you wish 
* to use these data, please contact TUV author Madronich or the groups who collected the data
* Using 2.e-2 as the detection limit.  Set value to half of that.

* Lake Giles, PA. From Craig Williamson
      
      jdom = jdom + 1
      dlabel(jdom) = 'Giles PA, 1994'

      jdom = jdom + 1
      dlabel(jdom) = 'Giles PA, 2015'

      OPEN(unit=kin,file='DATAE1/Lakes_abs/Giles_dom.abs',status='old')
      DO i = 1, 4
         READ(kin,*)
      ENDDO
      n = 601
      DO i = 1, n
         READ(kin,*) x1(i), dum1, dum2
         x2(i) = x1(i)
         IF(dum1 .LT. 2.e-2) THEN
            y1(i) = 1.e-2
         ELSE
            y1(i) = dum1
         ENDIF
         IF(dum2 .LT. 2.e-2) THEN
            y2(i) = 1.e-2
         ELSE
            y2(i) = dum2
         ENDIF
      ENDDO
      CLOSE(kin)

      n1 = n
      CALL terint(nw,wl,yg1, n1,x1,y1, 1,0)

      n2 = n
      CALL terint(nw,wl,yg2, n2,x2,y2, 1,0)

      DO iw = 1, nw-1
         sdom(jdom-1,iw) = yg1(iw)
         sdom(jdom  ,iw) = yg2(iw)
      ENDDO

**********************
* Lake Tahoe, CA, from Kevin Rose
      
      jdom = jdom + 1
      dlabel(jdom) = 'Tahoe, CA (KR)'

      OPEN(unit=kin,file='DATAE1/Lakes_abs/Tahoe.csv',status='old')
      DO i = 1, 1
         READ(kin,*)
      ENDDO
      n = 601
      DO i = 1, n
         READ(kin,'(I3,1x,F13.8)') idum, dum1
         x1(i) = float(idum)
         IF(dum1 .LT. 2.e-2) THEN
            y1(i) = 1.e-2
         ELSE
            y1(i) = dum1
         ENDIF
      ENDDO
      CLOSE(kin)

      n1 = n
      CALL terint(nw,wl,yg1, n1,x1,y1, 1,0)
      DO iw = 1, nw-1
         sdom(jdom,iw) = yg1(iw)
      ENDDO

**********************
* Lake Taupo, NZ (N. Island) from Kevin Rose
      
      jdom = jdom + 1
      dlabel(jdom) = 'Taupo, NZ (KR)'

      OPEN(unit=kin,file='DATAE1/Lakes_abs/Taupo.csv',status='old')
      DO i = 1, 1
         READ(kin,*)
      ENDDO
      n = 601
      DO i = 1, n
         READ(kin,'(I3,1x,F13.8)') idum, dum1
         x1(i) = float(idum)
         IF(dum1 .LT. 2.e-2) THEN
            y1(i) = 1.e-2
         ELSE
            y1(i) = dum1
         ENDIF
      ENDDO
      CLOSE(kin)

      n1 = n
      CALL terint(nw,wl,yg1, n1,x1,y1, 1,0)
      DO iw = 1, nw-1
         sdom(jdom,iw) = yg1(iw)
      ENDDO

**********************
* Laguna Negra, Chile from Kevin Rose
      
      jdom = jdom + 1
      dlabel(jdom) = 'Laguna Negra, Chile'

      OPEN(unit=kin,file='DATAE1/Lakes_abs/Negra.abs',status='old')
      DO i = 1, 1
         READ(kin,*)
      ENDDO
      n = 601
      DO i = 1, n
         READ(kin,*) idum, dum1
         x1(i) = float(idum)
         IF(dum1 .LT. 2.e-2) THEN
            y1(i) = 1.e-2
         ELSE
            y1(i) = dum1
         ENDIF
      ENDDO
      CLOSE(kin)

      n1 = n
      CALL terint(nw,wl,yg1, n1,x1,y1, 1,0)
      DO iw = 1, nw-1
         sdom(jdom,iw) = yg1(iw)
      ENDDO

**********************
* Acton Lake, Ohio, from Kevin Rose
      
      jdom = jdom + 1
      dlabel(jdom) = 'Acton, OH'

      OPEN(unit=kin,file='DATAE1/Lakes_abs/Acton.csv',status='old')
      DO i = 1, 1
         READ(kin,*)
      ENDDO
      n = 601
      DO i = 1, n
         READ(kin,'(I3,1x,F13.8)') idum, dum1
         x1(i) = float(idum)
         IF(dum1 .LT. 2.e-2) THEN
            y1(i) = 1.e-2
         ELSE
            y1(i) = dum1
         ENDIF
      ENDDO
      CLOSE(kin)

      n1 = n
      CALL terint(nw,wl,yg1, n1,x1,y1, 1,0)
      DO iw = 1, nw-1
         sdom(jdom,iw) = yg1(iw)
      ENDDO

*****************************
* Crystal Lake, Wisconsin, lat=46.002,lon=-89.613, in Vilas County
* caution: there are 22 lakes named Crystal in Wisconsin

      jdom = jdom + 1
      dlabel(jdom) = 'Crystal (Vilas Co.) WI'

      OPEN(unit=kin,file='DATAE1/Lakes_abs/Crystal.abs',status='old')
      DO i = 1, 1
         READ(kin,*)
      ENDDO
      n = 601
      DO i = 1, n
         READ(kin,*) idum, dum1
         x1(i) = float(idum)
         IF(dum1 .LT. 2.e-2) THEN
            y1(i) = 1.e-2
         ELSE
            y1(i) = dum1
         ENDIF
      ENDDO
      CLOSE(kin)

      n1 = n
      CALL terint(nw,wl,yg1, n1,x1,y1, 1,0)
      DO iw = 1, nw-1
         sdom(jdom,iw) = yg1(iw)
      ENDDO

**********************
* Mystic Lake, MT, from Kevin Rose
      
      jdom = jdom + 1
      dlabel(jdom) = 'Mystic, MT'

      OPEN(unit=kin,file='DATAE1/Lakes_abs/Mystic.csv',status='old')
      DO i = 1, 1
         READ(kin,*)
      ENDDO
      n = 601
      DO i = 1, n
         READ(kin,'(I3,1x,F13.8)') idum, dum1
         x1(i) = float(idum)
         IF(dum1 .LT. 2.e-2) THEN
            y1(i) = 1.e-2
         ELSE
            y1(i) = dum1
         ENDIF
      ENDDO
      CLOSE(kin)

      n1 = n
      CALL terint(nw,wl,yg1, n1,x1,y1, 1,0)
      DO iw = 1, nw-1
         sdom(jdom,iw) = yg1(iw)
      ENDDO

*****************************
* Lake Tahoe, specific days of 2014, from Erin Overholt
* label different days by Julian Date.

      OPEN(unit=kin,file='DATAE1/Lakes_abs/Tahoe_eo.abs',status='old')
      DO i = 1, 4
         READ(kin,*)
      ENDDO
      n = 601
      DO i = 1, n
         READ(kin,*) idum, (ydum(i,j), j = 1, 9)
         xdum(i) = float(idum)
         ydum(i,j) = max(ydum(i,j),1.e-2)
      ENDDO
      CLOSE(kin)
**
      jdom = jdom + 1
      dlabel(jdom) = 'Tahoe_014'
      n1 = n
      DO i = 1, n1
         x1(i) = xdum(i)
         y1(i) = ydum(i,1)
      ENDDO
      CALL terint(nw,wl,yg1, n1,x1,y1, 1,0)
      DO iw = 1, nw-1
         sdom(jdom,iw) = yg1(iw)
      ENDDO
**
      jdom = jdom + 1
      dlabel(jdom) = 'Tahoe_043'
      n1 = n
      DO i = 1, n1
         x1(i) = xdum(i)
         y1(i) = ydum(i,2)
      ENDDO
      CALL terint(nw,wl,yg1, n1,x1,y1, 1,0)
      DO iw = 1, nw-1
         sdom(jdom,iw) = yg1(iw)
      ENDDO
**
      jdom = jdom + 1
      dlabel(jdom) = 'Tahoe_100'
      n1 = n
      DO i = 1, n1
         x1(i) = xdum(i)
         y1(i) = ydum(i,3)
      ENDDO
      CALL terint(nw,wl,yg1, n1,x1,y1, 1,0)
      DO iw = 1, nw-1
         sdom(jdom,iw) = yg1(iw)
      ENDDO
**
      jdom = jdom + 1
      dlabel(jdom) = 'Tahoe_141'
      n1 = n
      DO i = 1, n1
         x1(i) = xdum(i)
         y1(i) = ydum(i,4)
      ENDDO
      CALL terint(nw,wl,yg1, n1,x1,y1, 1,0)
      DO iw = 1, nw-1
         sdom(jdom,iw) = yg1(iw)
      ENDDO
**
      jdom = jdom + 1
      dlabel(jdom) = 'Tahoe_161'
      n1 = n
      DO i = 1, n1
         x1(i) = xdum(i)
         y1(i) = ydum(i,5)
      ENDDO
      CALL terint(nw,wl,yg1, n1,x1,y1, 1,0)
      DO iw = 1, nw-1
         sdom(jdom,iw) = yg1(iw)
      ENDDO
**
      jdom = jdom + 1
      dlabel(jdom) = 'Tahoe_231'
      n1 = n
      DO i = 1, n1
         x1(i) = xdum(i)
         y1(i) = ydum(i,6)
      ENDDO
      CALL terint(nw,wl,yg1, n1,x1,y1, 1,0)
      DO iw = 1, nw-1
         sdom(jdom,iw) = yg1(iw)
      ENDDO
**
      jdom = jdom + 1
      dlabel(jdom) = 'Tahoe_260'
      n1 = n
      DO i = 1, n1
         x1(i) = xdum(i)
         y1(i) = ydum(i,7)
      ENDDO
      CALL terint(nw,wl,yg1, n1,x1,y1, 1,0)
      DO iw = 1, nw-1
         sdom(jdom,iw) = yg1(iw)
      ENDDO
**
      jdom = jdom + 1
      dlabel(jdom) = 'Tahoe_289'
      n1 = n
      DO i = 1, n1
         x1(i) = xdum(i)
         y1(i) = ydum(i,8)
      ENDDO
      CALL terint(nw,wl,yg1, n1,x1,y1, 1,0)
      DO iw = 1, nw-1
         sdom(jdom,iw) = yg1(iw)
      ENDDO
**
      jdom = jdom + 1
      dlabel(jdom) = 'Tahoe_352'
      n1 = n
      DO i = 1, n1
         x1(i) = xdum(i)
         y1(i) = ydum(i,9)
      ENDDO
      
      DO iw = 1, nw-1
         sdom(jdom,iw) = yg1(iw)
      ENDDO

*****************************
* Lake Michigan, Manitowoc WI, data from Richard Zepp
* 59 different data sets


      OPEN(unit=kin,file=
     $     'DATAE1/Lakes_abs/GreatLakes_Names.abs',
     $     status='old')
      READ(kin,*)
      DO j = 1, 59
         READ(kin,*) aname(j)
      ENDDO
      CLOSE(kin)

      OPEN(unit=kin,file=
     $     'DATAE1/Lakes_abs/GreatLakesUVDOCData2011_2015.abs'
     $     ,status='old')
      n = 261
      DO i = 1, n
         READ(kin,*) x(i), (y(j,i), j = 1, 59)
      ENDDO
      CLOSE(kin)

      DO j = 1, 59
         jdom = jdom + 1
         dlabel(jdom) = ' '
         dlabel(jdom) = aname(j)
         do i = 1, n
            x1(i) = x(i)
            y1(i) = y(j,i)
         enddo
         CALL terint(nw,wl,yg1, n,x1,y1, 1,0)
         DO iw = 1, nw-1
            sdom(jdom,iw) = yg1(iw)
         ENDDO

      ENDDO

**********************

* Lakes Annie (FL), Acton (OH), and Giles (PA), from Erin Overholt
* file Seasonal_

      OPEN(unit=kin,file=
     $     'DATAE1/Lakes_abs/Seasonal_GAA.abs',
     $     status='old')
      READ(kin,*)
      READ(kin,*) (idate(j), j = 1, 45) 
      DO j = 1, 45
         WRITE(aname(j),'(I6)') idate(j)
c         write(*,*) j, idate(j), aname(j)
      ENDDO
      DO i = 1, 601
         READ(kin,*) x(i), (y(j,i), j = 1, 45)
      ENDDO
      CLOSE(kin)

      DO j = 1, 45
         jdom = jdom + 1
         dlabel(jdom) = ' '
         IF (j .LE. 11)                 dlabel(jdom) = 'Annie'//aname(j)
         IF (j .GE. 12 .AND. j .LE. 37) dlabel(jdom) = 'Acton'//aname(j)
         IF (j .GE. 38)                 dlabel(jdom) = 'Giles'//aname(j)
         DO i = 1, n
            x1(i) = x(i)
            y1(i) = y(j,i)
         ENDDO
         CALL terint(nw,wl,yg1, n,x1,y1, 1,0)
         DO iw = 1, nw-1
            sdom(jdom,iw) = yg1(iw)
         ENDDO
      ENDDO

**********************

* Lake Tahoe at different locations, from  Erin Overholt
* 
      OPEN(unit=kin,file=
     $     'DATAE1/Lakes_abs/Tahoe_locations.csv',
     $     status='old')
      DO j = 1, 17
         READ(kin,'(a20)') aname(j)
      ENDDO
      CLOSE(kin)

      OPEN(unit=kin,file=
     $     'DATAE1/Lakes_abs/Tahoe_2008.abs',
     $     status='old')
      READ(kin,*)
      READ(kin,*)
      READ(kin,*) (idate(j), j = 1, 17) 
      n = 601
      DO i = 1, n
         READ(kin,*) x(i), (y(j,i), j = 1, 17)
      ENDDO
      CLOSE(kin)

      DO j = 1, 17
         jdom = jdom + 1
         dlabel(jdom) = ' '
         dlabel(jdom) = aname(j)
         DO i = 1, n
            x1(i) = x(i)
            y1(i) = y(j,i)
         ENDDO
         CALL terint(nw,wl,yg1, n,x1,y1, 1,0)
         DO iw = 1, nw-1
            sdom(jdom,iw) = yg1(iw)
         ENDDO
      ENDDO

**********************
**********************
      RETURN
      END
      
