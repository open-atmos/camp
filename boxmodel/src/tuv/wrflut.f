      SUBROUTINE wrflut(nw, wl, nz, tlev, aircon)


* Generates look-up tables for WRF-Chem, of pre-interpolated 
* molecular action spectra, xs*qy, as fnct of j, w, T, n.
* Uses swchem and its routines, bottom layer iz = 1, to overwrite 
* at desired T(TEMP) and n(aircon).  
* STOPS at completion.
      use tuv_params
      IMPLICIT NONE
      INTEGER nw, iw, nz, iz, itemp, idens, nj, ij
      CHARACTER*50 jlabel(kj)
      INTEGER tpflag(kj)
      REAL wl(kw), tlev(kz), aircon(kz)
      REAL sj(kj, kz, kw)
      
      OPEN(unit=88,file='../sq_wrf.txt', status='new')

      WRITE(88,888) (wl(iw), iw = 1, nw)
      DO itemp = 1, 4
         tlev(1) = 200. + 30.*FLOAT(itemp-1)
         DO idens = 1, 4
            aircon(1) = 2.45e19 / FLOAT(idens)

            nj = 0
            CALL swchem(nw,wl,nz,tlev,aircon, nj,sj,jlabel,tpflag)

            WRITE(88,881)tlev(1), aircon(1)  
            DO ij = 1, nj
               WRITE(88,882) tpflag(ij), jlabel(ij)
               WRITE(88,888) (sj(ij,1,iw), iw = 1, nw-1)
            ENDDO

         ENDDO
      ENDDO
 881  FORMAT('T,n',1x,0pf10.1,1x, 1pe11.4)
 882  FORMAT(i1,2x,a50)
 888  FORMAT(6(1pe11.4,1x))

      STOP
      RETURN
      END
      
