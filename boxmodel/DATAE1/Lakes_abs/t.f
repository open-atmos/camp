* routine to read the lake absorption coefficient data from Kevin Rose
* 18 Nov 2015, S. Madronich
* clean up of data:
* eflag1 = True if single digit zero was replaced by floating point zero.
* eflag2 = True if negative value was replaced by floating point zero.

      IMPLICIT NONE
      INTEGER kl
      PARAMETER (kl=130)
      CHARACTER*50 LakeName(kl)
      
      INTEGER kw
      PARAMETER (kw = 601)

      REAL wc(kw)
      REAL abs(kl,kw)

      INTEGER i, j
      CHARACTER*50 aline
      LOGICAL eflag1, eflag2

      character*50 bline(125)
      real blat(125), blon(125)
      real alat(kl), alon(kl)

      open(unit=10,file='NAMES_LIST',status='old')
      do i = 1, kl
         read(10,100) LakeName(i)
 100     format(24x,a50)
      enddo
      close(10)

      open(unit=11,file='names.csv',status='old')
      open(unit=12,file='lat.csv',status='old')
      open(unit=13,file='lon.csv',status='old')
      do j = 1, 125
         read(11,'(a50') bline(j)
         read(12,*) blat(j)
         read(13,*) blon(j)
      enddo
      close(11)
      close(12)
      close(13)

      do i = 1, 2
         write(*,*) i, lakename(i)
         do j = 1, 125
            write(*,*) bline(j)
            if (lakename(i) .eq. bline(j)) then
               alat(i) = blat(j)
               alon(i) = blon(j)
               go to 19
            endif
         enddo
         write(*,*) 'not found:', lakename(i)
         stop '1'
 19      continue
         write(*,*) lakename(i), alat(i), alon(i)
      enddo

      stop

      do i = 1, 2

         write(*,*) lakename(i)
         write(22,*) lakename(i)
         open(unit=11,file=LakeName(i),status='old')
         read(11,*)
         do j = 1, kw

            eflag1 = .false.
            eflag2 = .false.
         
            read(11,'(a50)') aline

            if(index(aline,' ') .gt. 10) then
               read(aline,111)wc(j), abs(i,j)
 111           format(f3.0,1x, f12.9)
            else
               read(aline,112)wc(j)
 112           format(f3.0)
               abs(i,j) = 0.
               eflag1 = .true.
            endif

            if(abs(i,j) .lt. 0. )then
               abs(i,j) = 0.
               eflag2 = .true.
            endif

**output, including error flags:
* eflag1 = True if single digit zero was replaced by floating point zero.
* eflag2 = True if negative value was replaced by floating point zero.

            write(22,222) eflag1, eflag2, wc(j), abs(i,j)
 222        format(2L1,1x,0pf5.1,1x,1pe11.4)
         enddo
         close(11)

      enddo


      end
