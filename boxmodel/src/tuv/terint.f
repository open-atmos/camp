      SUBROUTINE terint(ng,xg,yg,  n,x,y, c1,c2)
      IMPLICIT NONE

* SUBROUTINE to TERminate and INTerpolate a 1D input data array.
* INPUTS:
* working grid:

      INTEGER ng, ig
      REAL xg(ng)

* data array to be gridded:

      INTEGER n, i
      REAL x(n), y(n)

* multipliers for 4 y-values to be added:

      INTEGER c1, c2

* OUTPUT:
* gridded y-values, with proper termination at both ends

      REAL yg(ng)

* INTERNAL:
* input terminator points to be added:

      REAL xt1, yt1
      REAL xt2, yt2
      REAL xt3, yt3
      REAL xt4, yt4

* delta for adding points at beginning or end of data grids

      REAL deltax
      PARAMETER (deltax = 1.E-5)

* error flag:

      INTEGER ierr

* extended array:

      INTEGER n1, k1
      REAL x1(n+4), y1(n+4)

* check for valid input values of c1, c2:

      ierr = 0
      if(c1 .ne. 0 .and. c1 .ne. 1) ierr = 1
      if(c2 .ne. 0 .and. c2 .ne. 1) ierr = 1
      IF (ierr .NE. 0) THEN
         WRITE(*,*) ierr, 'entering terint.f'
         STOP
      ENDIF

* assign x-coordinate to 4 new points

      xt1 = -1.e36
      xt2 = x(1)*(1.-deltax)
      xt3 = x(n)*(1.+deltax)
      xt4 = 1.e36

* terminator multipliers  c1,c2 = e.g., = 1.,0.
* there are really only two practical options:  constant or zero

      yt1 = float(c1) * y(1)
      yt2 = float(c1) * y(1)

      yt3 = float(c2) * y(n)
      yt4 = float(c2) * y(n)

* transcribe input x,y array to avoid modifying it

      n1 = n
      do i = 1, n
         x1(i) = x(i)
         y1(i) = y(i)
      enddo

* extended data array, with up to 4 terminator points:
* note that n1 gets incremented by 1 for each call to ADDPNT      

      k1 = n1 + 4
      call addpnt(x1,y1,k1,n1, xt1,yt1)
      call addpnt(x1,y1,k1,n1, xt2,yt2)
      call addpnt(x1,y1,k1,n1, xt3,yt3)
      call addpnt(x1,y1,k1,n1, xt4,yt4)

* point to grid interpolation:

      CALL inter2(ng,xg,yg, n1,x1,y1,ierr)
      IF (ierr .NE. 0) THEN
         WRITE(*,*) ierr, 'exiting terint.f'
         STOP
      ENDIF

      RETURN
      END
