module ntergeo
   implicit none
contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Generalized interpolation program for a lookup table with a dimension NDIM   !
! Calculate weight factors for the k=2**n points surroundings the target point !
! Y=W1*Y1(x(1,1),...,x(1,n))+                                                  !
!    .... W2*Y2(x(i,2),...,x(2,n)) +                                           !
!         ....                                                                 !
!              .... Wi*Yi(x(i,1),...,x(i,n)) +                                 !
!                     ....                                                     !
!                            +...Wk*Yk(x(k,1),...,x(k,n))                      !
! Weights Wi are calculated based on the inverse distance to the target point  !
! A p-distance calculation is used, here p=2 (parameter pnum can be changed)   !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                              !
! Author : Bertrand Bessagnet - bertrand.bessagnet@citepa.org                  !
!                               bertrand.bessagnet@lmd.polytechnique.fr        !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! Put your lookup table on ONE dimension, if you want to use this program
!
   subroutine interpolation_general(&
   &ndim,      &  ! Scalar    : Integer   :I  : Dimension of the problem N > 1
   &maxdim,    &  ! Scalar    : Integer   :I  : Product of the number of elements of the input grid data on each dimension
   &kdim,      &  ! Array 1D  : Integer   :I  : Vector where are stored the number of data in each dimension in each direction(0:ndim)
   &vect,      &  ! Array 2D  : Real      :I  : Array where are stored all the interva edges for each dimension
   &vtable,    &  ! Array 1D  : Real      :I  : Coordinate values for the point you want to interpolate
   &table,     &  ! Array 1D  : Real      :I  : Coordinate values of the input grid data
   &avedelta,  &  ! Array 1D  : Real      :I  : Inverse average intervals of the input grid data on each dimension
   &maxdelta,  &  ! Array 1D  : Real      :I  : Maximum intervals of the input grid on each dimension
   &resu,      &  ! Scalar    : Real      :O  : Result of the interpolation
   &inei,      &  ! Scalar    : Integer   :I/O: Number of neighbours (reusable as Input for next point)
   &neighbours,&  ! Array 2D  : Integer   :I/O: Coordinates of neighbours in all directions ndim (reusable as Input for next point)
   &weights,   &  ! Array 1D  : Real      :O  : Weight factors
   &found      &  ! Scalar    : Logical   :O  : True if found, False if not found
   &)
!
      implicit none
!
      integer,parameter                                :: iprec=8          ! Precision for real, please do not change
! Begin possible user selection
      logical,parameter                                :: norm=.true.     ! Normalize or not by the average delta on each direction
      logical,parameter                                :: verbose=.false.  ! Level of message writing (true for debug)
      logical,parameter                                :: neighb=.false.    ! "True" only find up to ndim+1 closest neihbours to be faster
      ! "False" find up to 2^NDIM
      integer,parameter                                :: iconf=2          ! Number of cell to account for before and after
      ! the closest point, iconf=2 can be tested not more
      real(kind=iprec),parameter                       :: pnum=1.1d+00     ! p-distance parameter

! End user selection
      real(kind=iprec),parameter                       :: dzero=0.0d+00    ! Zero
      real(kind=iprec),parameter                       :: one=1.0d+00      ! One
!real(kind=iprec),parameter                      :: crit=1.0d-03     ! Criteria convergence for the distance
      integer,parameter                                :: ishowmax=100     ! Array dimension for basis change : dec to base N
      real(kind=iprec),parameter                       :: invpnum=one/pnum ! criteria convergence for the distance

      integer                                          :: nitermax,icp,iconfn,p,iter,npinit
      integer                                          :: np_up,np_dw,ilook,inp,maxdim,ndim,i,j,np,k,nn,pn,inei,prod
      integer                                          :: ineimax
      real(kind=iprec),dimension(:),allocatable        :: savedist
      integer,dimension(ndim)                          :: idelta,incr
      integer,dimension(2**ndim,ndim)                  :: neighbours
      integer,dimension(0:ndim)                        :: kdim
      integer,dimension(ndim)                          :: imatrix
      integer,dimension(ishowmax)                      :: digit
      integer,dimension(ndim)                          :: imatrix_up
      integer,dimension(ndim)                          :: imatrix_dw
      integer,dimension(:),allocatable                 :: npstock_order,iz,npstock
!logical,dimension(:),allocatable                 :: masking
      real(kind=iprec),dimension(ndim)                 :: dist
      real(kind=iprec),dimension(ndim)                 :: vtable
      real(kind=iprec),dimension(ndim)                 :: avedelta,maxdelta
      real(kind=iprec),dimension(ndim,maxdim)          :: vect
      real(kind=iprec),dimension(maxdim)               :: table
      real(kind=iprec),dimension(2**ndim)              :: weights
      real(kind=iprec),dimension(:),allocatable        :: diststore,dist_order
      real(kind=iprec)                                 :: deltadiv,dist_test,totdist,resu,distance
      logical                                          :: found

!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Some initializations
      nn=2**ndim
      pn=2**(ndim-1)
      iconfn=2*iconf+1
      ilook=iconfn**ndim
      found=.true.
      nitermax=2*sum(kdim(1:ndim))

      if(ndim.gt.2) then
         ineimax=ndim+1
      else
         ineimax=nn
      endif

      allocate(savedist(-1:nitermax))

      if(sum(neighbours(1,:)).eq.0) then ! Test here if we start from last closest neighbour
         npinit=1
      else
         call matrix2nbox(verbose,neighbours(1,:),ndim,kdim,npinit)
         if(npinit.eq.0) then
            if(verbose) print *,"Abort, initial point out of bounds..."
            inei=1
            found=.false.
            goto 1000
         endif
      endif
!
      if(ilook.lt.nn) then
         print *,"STOP: it seems you are going to test less than 2**N points"
         stop
      else
         if(verbose) print *,"OK: you are looking for the closest",ilook,"points"
      endif
! Calculate an average delta in each direction
!
! Find the closest neighbour
      savedist(-1:0)=-one
      iterloop: do iter=1,nitermax
         if(iter.eq.1) then
            if(sum(neighbours(1,:)).eq.0) then
               idelta(:)=nint((vtable(:)-vect(:,npinit))*avedelta(:))+neighbours(1,:)
            else
               idelta(:)=neighbours(1,:)
            endif
         else
            idelta(:)=idelta(:)+incr(:)
         endif
         np=1
         do j=1,ndim
            prod=1
            do k=1,j
               prod=prod*kdim(k-1)
            enddo
            if(idelta(j).gt.kdim(j)) then
               if(verbose) print *,"WARNING you are close to the top bounds....",j,idelta(j)
               if(iter.gt.1) then
                  idelta(j)=idelta(j)-1
               else
                  idelta(j)=kdim(j)
               endif
            endif
            if(idelta(j).lt.1) then
               if(verbose) print *,"WARNING you are close to the low bounds....",j,idelta(j)
               if(iter.gt.1) then
                  idelta(j)=idelta(j)+1
               else
                  idelta(j)=1
               endif
            endif
            prod=prod*(idelta(j)-1)
            np=np+prod
         enddo
         distance=dzero
         do i=1,ndim
            if(norm) then
               dist(i)=(vect(i,np)-vtable(i))*avedelta(i)
            else
               dist(i)=(vect(i,np)-vtable(i))
            endif
            incr(i)=0
            if(dist(i).lt.dzero) incr(i)=1
            if(dist(i).gt.dzero) incr(i)=-1
            if(pnum.ne.one) then
               distance=distance+(dabs(dist(i)))**pnum
            else
               distance=distance+dabs(dist(i))
            endif
         enddo
         if(distance.eq.dzero) exit iterloop
         if(pnum.ne.one) distance=distance**(invpnum)
! if((iter.gt.1.and.dabs(savedist(iter-1)-distance)/distance.lt.crit).or.(iter.gt.2.and.savedist(iter-2).eq.distance)) then
         if(iter.gt.2.and.savedist(iter-2).eq.distance) then
            if(verbose) print *,"Minimum distance",distance,"at",iter,"iterations"
            exit iterloop
         endif
         savedist(iter)=distance
      enddo iterloop
      do i=1,ndim
         if(idelta(i).lt.1.or.idelta(i).gt.kdim(i)) then
            if(verbose) print *,"POINT probably out of bounds....",i,idelta(i)
            np=0
         endif
      enddo
      iter=iter-1
      if(verbose.and.iter.eq.nitermax) print *,"WARNING: Convergence issue iter=itermax..."
      if(verbose) print *,"Closest point is NP=",np,idelta
!
      allocate(npstock(ilook))
      allocate(diststore(ilook))
      allocate(npstock_order(ilook))
      allocate(dist_order(ilook))
!allocate(masking(ilook))
      allocate(iz(1))
      weights=dzero
      neighbours=0
!Test if you are lucky :-)
      if(distance.eq.dzero) then
         inei=1
         resu = table(np)
         weights(inei)=one
         neighbours(inei,:)=idelta(:)
         goto 1000
      endif
!
! Select the block to be tested to find the closest 2**N points, for that we
! select a larger block up to <iconf> in each direction, then ilook=(2*iconf+1)**N
! points (iconfn=2*iconf+1)
!
      npstock=0
      diststore=-one
      do p=1,ilook
         inp=p-1
         call dec2base(inp,iconfn,ndim,digit)
         imatrix=idelta+digit(1:ndim)-iconf
         call matrix2nbox(verbose,imatrix,ndim,kdim,npstock(p))
         if(npstock(p).gt.0) then
            diststore(p)=dzero
            do i=1,ndim
               if(norm) then
                  if(pnum.ne.one) then
                     diststore(p)=diststore(p)+(dabs(vect(i,npstock(p))-vtable(i))*avedelta(i))**pnum
                  else
                     diststore(p)=diststore(p)+dabs(vect(i,npstock(p))-vtable(i))*avedelta(i)
                  endif
               else
                  if(pnum.ne.one) then
                     diststore(p)=diststore(p)+(dabs(vect(i,npstock(p))-vtable(i)))**pnum
                  else
                     diststore(p)=diststore(p)+dabs(vect(i,npstock(p))-vtable(i))
                  endif
               endif
            enddo
            if(pnum.ne.one) diststore(p)=diststore(p)**(invpnum)
         else
            diststore(p)=-one
         endif
      enddo
!
!Sort the matrix of ilook elements from lowest to highest distance
!masking=.true.
!where(diststore.lt.dzero) masking=.false.
!icp=0
!dist_order=-one
!do p=1,ilook
! iz=0
! dist_order(p)=minval(diststore,masking)
! iz=minloc(diststore,masking)
! if(iz(1).ne.0) then
!  icp=icp+1
!  npstock_order(p)=npstock(iz(1))
!  masking(iz(1))=.false.
! else
!  npstock_order(p)=0
! endif
!enddo
      icp=0
      dist_order=-one
      lookploop : do p=1,ilook
         iz(1)=0
! dist_order(p)=minval(diststore,mask = diststore >= dzero)
         iz=minloc(diststore, mask = diststore >= dzero)
         if(iz(1).ne.0) then
            dist_order(p)=diststore(iz(1))
            icp=icp+1
            npstock_order(p)=npstock(iz(1))
            diststore(iz(1)) = -one
            if(neighb.and.icp.eq.ineimax) exit lookploop ! Only if neighb=.true. of course
         else
            npstock_order(p) = 0
         endif
      enddo lookploop
      if(verbose) print *,"Number of neighbours inside the domain :",icp,"points"
      if(verbose) print *,"Interpolation done over                :",nn,"points"
!
! Calculate the characteristic length of the grid close to the target
      call nbox2matrix(verbose,npstock_order(1),ndim,kdim(1:ndim),imatrix)
      dist_test=dzero
      do i=1,ndim
         imatrix_up=imatrix
         imatrix_dw=imatrix
         deltadiv=2.0d+00
         if(imatrix(i).eq.kdim(i)) then
            imatrix_up(i)=imatrix(i)
            deltadiv=one
         else
            imatrix_up(i)=imatrix(i)+1
         endif
         if(imatrix(i).eq.1) then
            imatrix_dw(i)=imatrix(i)
            deltadiv=one
         else
            imatrix_dw(i)=imatrix(i)-1
         endif
         call matrix2nbox(verbose,imatrix_up,ndim,kdim,np_up)
         call matrix2nbox(verbose,imatrix_dw,ndim,kdim,np_dw)
         if(norm) then
            if(pnum.ne.one) then
               dist_test=dist_test+(((dabs(vect(i,np_up)-vect(i,np_dw)))/deltadiv)*avedelta(i))**pnum
            else
               dist_test=dist_test+((dabs(vect(i,np_up)-vect(i,np_dw)))/deltadiv)*avedelta(i)
            endif
         else
            if(pnum.ne.one) then
               dist_test=dist_test+((dabs(vect(i,np_up)-vect(i,np_dw)))/deltadiv)**pnum
            else
               dist_test=dist_test+(dabs(vect(i,np_up)-vect(i,np_dw)))/deltadiv
            endif
         endif
      enddo
      dist_test=1.0d+00*dist_test**(invpnum) ! characteristic length of the grid cell close to the target
!
!Compare the distance of the lowest distance to the characteristic length of
!the grid, if too far we exit
!
      totdist=dzero
      if(neighb) nn=ineimax
      do p=1,nn
         if(dist_order(p).gt.dzero.and.npstock_order(p).ne.0) then
            totdist=totdist+one/dist_order(p)
         endif
      enddo
      if(totdist.eq.dzero.or.dist_order(1).gt.dist_test) then
         if(verbose) write(*,999)"Point too far from the matrix...Abort...",dist_order(1),">",dist_test
         inei=1
         resu=dzero
         found=.false.
         go to 1000
      else
         if(verbose) write(*,999)" Perfect: you are inside the domain",dist_order(1),"<",dist_test
         if(verbose) write(*,*)"A slight extrapolation is permitted"
      endif
!
! Main loop to define the weight and calculate the result of interpolation over
! the 2**N or N+1 points included in the reduced block of data of ilook points taking the
! closest points thank to the ranking processes performed before
! diststore->dist_order
      resu=dzero
      inei=0
      weights=dzero
      neighbours=0
      do p=1,nn
         if(npstock_order(p).ne.0) then
            if(dist_order(p).gt.dzero) then ! General case
               inei=inei+1
               weights(inei)=(one/dist_order(p))/totdist
               resu = resu + weights(inei)*table(npstock_order(p))
               call nbox2matrix(verbose,npstock_order(p),ndim,kdim(1:ndim),imatrix)
               neighbours(inei,:)=imatrix(:)
            elseif(dist_order(p).eq.dzero) then ! Case of a point equal to a matrix point
               inei=inei+1
               resu = table(npstock_order(p))
               weights(inei)=one
               call nbox2matrix(verbose,npstock_order(p),ndim,kdim(1:ndim),imatrix)
               neighbours(inei,:)=imatrix(:)
               inei=1
               go to 1000
            else
               if(verbose) print *,"Skip point, probably out of bounds..."
            endif
         endif
      enddo
999   format(a,e14.3,a3,e14.3)
!
1000  continue
      if(allocated(npstock))deallocate(npstock)
      if(allocated(diststore))deallocate(diststore)
      if(allocated(npstock_order))deallocate(npstock_order)
      if(allocated(dist_order))deallocate(dist_order)
!if(allocated(masking))deallocate(masking)
      if(allocated(iz))deallocate(iz)
      if(allocated(savedist))deallocate(savedist)
      return
!
   end subroutine interpolation_general

   subroutine nbox2matrix(verbose,np,ndim,kdim,imatrix)
! Find coordinates in each direction [1,...,N] from a 1D coordinate
! system
! np      : Input  : Coordinate in 1D
! ndim    : Input  : Dimension of the matrix (N)
! kdim    : Input  : Number of edges on each direction
! imatrix : Output : Coordinates in ndim-dimension (I1,...,IN)

      implicit none
      integer                 :: ndim,np
      integer                 :: i,j
      integer                 :: r,z,nval,prod
      integer,dimension(ndim) :: kdim,imatrix
      logical                 :: verbose

      z=np
      do i=1,ndim
         prod=1
         do j=ndim-i,1,-1
            prod=prod*kdim(j)
         enddo
         r=mod(z,prod)
         nval=nint(float(z-r)/float(prod))
         if(r.ne.0) then
            imatrix(ndim+1-i)=nval+1
            z=z-nval*prod
         else
            imatrix(ndim+1-i)=nval
            z=z-(nval-1)*prod
         endif
      enddo

   end subroutine nbox2matrix

   subroutine matrix2nbox(verbose,imatrix,ndim,kdim,np)
! Find a 1D coordinate system from the coordinates in each direction [1,...,N]
! imatrix : Input  : Coordinates in ndim-dimension (I1,...,IN)
! ndim    : Input  : Dimension of the matrix (N)
! kdim    : Input  : Number of edges on each direction (Caution start from 0!)
! np      : Output : Coordinates in 1D


      implicit none
      integer                    :: ndim,np
      integer                    :: j,itest,k,iexit
      integer                    :: prod
      integer,dimension(0:ndim)  :: kdim
      integer,dimension(ndim)    :: imatrix
      logical                    :: verbose

      np=1
      iexit=0
      ploop : do j=1,ndim
         prod=1
         do k=1,j
            prod=prod*kdim(k-1)
         enddo
         itest=imatrix(j)
         if(itest.gt.kdim(j)) then
            if(verbose) print *,"WARNING you are close to the bounds....",j,itest
            np=0
            iexit=1
            exit ploop
         endif
         if(itest.lt.1) then
            if(verbose) print *,"WARNING you are close to the bounds....",j,itest
            np=0
            iexit=1
            exit ploop
         endif
         prod=prod*(itest-1)
         np=np+prod
      enddo ploop
      if(iexit.eq.1) np=0

   end subroutine matrix2nbox

   subroutine dec2base(numb,base,ishow,digit)
! Transform Integer from Decimal to Base "base"
! numb    : Input  : Integer to be transformed
! base    : Input  : Base of the transformation
! ishow   : Input  : not used so far
! digit   : Output : array of digits (table of ishowmax elements)

      implicit none
      integer, parameter          :: ishowmax=100
      integer                     :: i,icount,dec,base,numb,ishow
      integer,dimension(ishowmax) :: digit

      digit=0
      icount = 0
      dec=numb
      do i = 1,ishowmax
         if (mod(dec,base)==0) then
            digit(i) = 0
         else
            digit(i) = mod(dec,base)
         end if
         dec = dec/base
         icount = icount + 1
         if (dec == 0) then
            exit
         end if
      end do
   end subroutine dec2base

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! This program find the maximum and average intervals for a given mesh, grid or look-up table
! Useful to call before entering the general interpolation subroutine
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   subroutine findmaxave(&

      ndim,      &  ! Input : Integer                   : Dimension of the look-up table : 1, 2, 3, 4, ....
      maxdim,    &  ! Input : Integer                   : kdim(1)*kdim(2)*...*kdim(ndim)
      kdim,      &  ! Input : Array 1D, Integer         : Array where are stored the number of data in each direction
      vect,      &  ! Input : Array 2D, Real            : Array where are stored all the interval edges on each dimension
      avedelta,  &  ! Output: Array 1D, Real            : Inverse average interval on each dimension
      maxdelta   &  ! Output: Array 1D, Real            : Maximum interval on each dimension
      )
      !
      implicit none

      ! Begin possible user selection
      integer,parameter                                :: iprec=8        ! Precision for real, please do not change
      logical,parameter                                :: verbose=.false.! Precision for real, please do not change
      ! End user selection
      real(kind=8),parameter                           :: dzero=0.0d+00  ! Zero

      integer                                          :: i,j,np,ndim,maxdim
      integer,dimension(0:ndim)                        :: kdim
      integer,dimension(ndim)                          :: imatrix,imatrixn,icount
      real(kind=iprec),dimension(ndim)                 :: avedelta,maxdelta
      real(kind=iprec),dimension(ndim,maxdim)          :: vect
      real(kind=iprec)                                 :: ddelta

      !
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !
      avedelta=dzero
      icount=0
      do j=1,maxdim
         call nbox2matrix(verbose,j,ndim,kdim(1:ndim),imatrix)
         imatrixn=imatrix
         do i=1,ndim
            imatrixn(i)=imatrix(i)+1
            if(imatrixn(i).le.kdim(i)) then
               call matrix2nbox(verbose,imatrixn,ndim,kdim,np)
               ddelta=dabs(vect(i,np)-vect(i,j))
               avedelta(i)=avedelta(i)+ddelta
               if(j.eq.1) then
                  maxdelta(i)=ddelta
               else
                  if(ddelta.gt.maxdelta(i)) maxdelta(i)=ddelta
               endif
               icount(i)=icount(i)+1
            endif
            imatrixn(i)=imatrixn(i)-1
         enddo
      enddo
      avedelta=dfloat(icount)/avedelta ! Compute inverse as * is faster thar / in the interpolation


      return
      !
   end subroutine findmaxave


end module ntergeo
