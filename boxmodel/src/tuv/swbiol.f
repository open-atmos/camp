* This file contains the following subroutines, related to specifying 
* biological spectral weighting functions:
*     swbiol

*=============================================================================*

      SUBROUTINE swbiol(nw,wl,wc,j,s,label)

*-----------------------------------------------------------------------------*
*=  PURPOSE:                                                                 =*
*=  Create or read various weighting functions, e.g. biological action       =*
*=  spectra, UV index, etc.                                                  =*
*-----------------------------------------------------------------------------*
*=  PARAMETERS:                                                              =*
*=  NW     - INTEGER, number of specified intervals + 1 in working        (I)=*
*=           wavelength grid                                                 =*
*=  WL     - REAL, vector of lower limits of wavelength intervals in      (I)=*
*=           working wavelength grid                                         =*
*=  WC     - REAL, vector of central wavelength of wavelength intervals    I)=*
*=           in working wavelength grid                                      =*
*=  J      - INTEGER, counter for number of weighting functions defined  (IO)=*
*=  S      - REAL, value of each defined weighting function at each       (O)=*
*=           defined wavelength                                              =*
*=  LABEL  - CHARACTER*50, string identifier for each weighting function  (O)=*
*=           defined                                                         =*
*-----------------------------------------------------------------------------*
      use tuv_params
      IMPLICIT NONE

      INTEGER kdata
      PARAMETER(kdata=1000)

* input:
      REAL wl(kw), wc(kw)
      INTEGER nw

* input/output:
      INTEGER j

* output: (weighting functions and labels)
      REAL s(ks,kw)
      CHARACTER*50 label(ks)

* internal:
      REAL x1(kdata)
      REAL y1(kdata)
      REAL yg(kw)

      REAL fery, futr
      EXTERNAL fery, futr
      INTEGER i, iw, n

      INTEGER idum
      REAL dum1, dum2
      REAL em, a, b, c
      REAL sum

      REAL a0, a1, a2, a3

*_______________________________________________________________________

********* Photosynthetic Active Radiation (400 < PAR < 700 nm)
* conversion to micro moles m-2 s-1:
*  s = s * (1e6/6.022142E23)(w/1e9)/(6.626068E-34*2.99792458E8)
 
      j = j + 1
      label(j) = 'PAR, 400-700 nm, umol m-2 s-1'
      DO iw = 1, nw-1
         IF (wc(iw) .GT. 400. .AND. wc(iw) .LT. 700.) THEN
            s(j,iw) = 8.36e-3 * wc(iw)
         ELSE
            s(j,iw) = 0.
         ENDIF
      ENDDO

********** unity raf constant slope:  

      j = j + 1
      label(j) = 'Exponential decay, 14 nm/10'
      DO iw = 1, nw-1
         s(j,iw) = 10.**(-(wc(iw) -300.)/14.)
      ENDDO

************ DNA damage action spectrum
* from: Setlow, R. B., The wavelengths in sunlight effective in 
*       producing skin cancer: a theoretical analysis, Proceedings 
*       of the National Academy of Science, 71, 3363 -3366, 1974.
* normalize to unity at 254 nm
* Data read from original hand-drawn plot by Setlow
* received from R. Setlow in May 1995
* data is per quantum (confirmed with R. Setlow in May 1995).  
* Therefore must put on energy basis if irradiance is is energy
* (rather than quanta) units.

      j = j + 1
      label(j) = 'DNA damage, in vitro (Setlow, 1974)'
      OPEN(UNIT=kin,FILE='DATAS1/dna.setlow.new',STATUS='old')
      do i = 1, 11
         read(kin,*)
      enddo
      n = 55
      DO i = 1, n
         READ(kin,*) x1(i), y1(i)

*normalize at 300:
c         y1(i) = (y1(i)/2.4E-02)  *  x1(i)/300. 

* normalize at 254 nm
         y1(i) = (y1(i)/0.85) * (x1(i)/254.)

      ENDDO
      CLOSE (kin)

* terminate data points at both ends, interpolate, and assign       

      CALL terint(nw,wl,yg, n,x1,y1, 1,0)

      do iw = 1, nw-1
         s(j,iw) = yg(iw)
      enddo

* then scale by Ebola decadal inactivation, convert to per hour
c      DO iw = 1, nw-1
c         s(j,iw) = yg(iw) * 3600./17.
* cut off at 320 nm
c         IF(WC(IW) .GT. 320.) S(J,IW) = 0.
c     ENDDO

********* skin cancer in mice,  Utrecht/Phildelphia study
*from de Gruijl, F. R., H. J. C. M. Sterenborg, P. D. Forbes, 
*     R. E. Davies, C. Cole, G. Kelfkens, H. van Weelden, H. Slaper,
*     and J. C. van der Leun, Wavelength dependence of skin cancer 
*     induction by ultraviolet irradiation of albino hairless mice, 
*     Cancer Res., 53, 53-60, 1993.
* Calculate with function futr(w), normalize at 300 nm.

      j = j + 1
      label(j) = 'SCUP-mice (de Gruijl et al., 1993)'
      DO iw = 1, nw-1
         s(j,iw) =  futr(wc(iw)) / futr(300.)
      ENDDO
         
*********** Utrecht/Philadelphia mice spectrum corrected for humans skin.
* From de Gruijl, F.R. and J. C. van der Leun, Estimate of the wavelength 
* dependency of ultraviolet carcinogenesis and its relevance to the
* risk assessment of a stratospheric ozone depletion, Health Phys., 4,
* 317-323, 1994.

      j = j + 1
      label(j) = 'SCUP-human (de Gruijl and van der Leun, 1994)'
      OPEN(UNIT=kin,FILE='DATAS1/SCUP-h',STATUS='old')
      n = 28
      DO i = 1, n
         READ(kin,*) x1(i), y1(i)
      ENDDO
      CLOSE (kin)

      CALL terint(nw,wl,yg, n,x1,y1, 1,0)
      DO iw = 1, nw-1
         s(j,iw) = yg(iw)
      ENDDO
      
***************** standard human erythema action spectrum
*from:
* McKinlay, A. F., and B. L. Diffey, A reference action spectrum for 
* ultraviolet induced erythema in human skin, in Human Exposure to 
* Ultraviolet Radiation: Risks and Regulations, W. R. Passchler 
* and B. F. M. Bosnajokovic, (eds.), Elsevier, Amsterdam, 1987.
** Webb, A.R., H. Slaper, P. Koepke, and A. W. Schmalwieser, 
** Know your standard: Clarifying the CIE erythema action spectrum,
** Photochem. Photobiol. 87, 483-486, 2011.
** Naming after CIE is discouraged because:
** 1)  Possible confusion with previously erroneous CIE publications
** 2)  CIE charges money to get a copy of their reports, including the
**     erroneous ones

      j = j + 1
      label(j) = 'Standard human erythema (Webb et al., 2011)'
      DO iw = 1, nw-1
         s(j,iw) = fery(wc(iw))
      ENDDO

***************** UV index (Canadian - WMO/WHO)
* from:
* Report of the WMO Meeting of experts on UV-B measurements, data quality 
* and standardization of UV indices, World Meteorological Organization 
* (WMO), report No. 95, Geneva, 1994.
* based on the Standard erythema weighting, multiplied by 40.

      j = j + 1
      label(j) = 'UV index (WMO, 1994; Webb et al., 2011)'
      DO iw = 1, nw-1
         s(j,iw) = 40. * fery(wc(iw))
      ENDDO

************* Human erythema - Anders et al.
* from:
* Anders, A., H.-J. Altheide, M. Knalmann, and H. Tronnier,
* Action spectrum for erythema in humands investigated with dye lasers, 
* Photochem. and Photobiol., 61, 200-203, 1995.
* for skin types II and III, Units are J m-2.

      j = j + 1
      label(j) = 'Erythema, humans (Anders et al., 1995)'
      OPEN(UNIT=kin,FILE='DATAS1/ery.anders',STATUS='old')
      do i = 1, 5
         read(kin,*)
      enddo
      n = 28
      DO i = 1, n
         READ(kin,*) x1(i), y1(i)
         y1(i) = 1./y1(i)
      ENDDO
      CLOSE (kin)

      CALL terint(nw,wl,yg, n,x1,y1, 1,0)
      DO iw = 1, nw-1
         s(j,iw) = yg(iw)
      ENDDO

********* 1991-92 ACGIH threshold limit values
* from
* ACGIH, 1991-1992 Threshold Limit Values, American Conference 
*  of Governmental and Industrial Hygienists, 1992.

      j = j + 1
      label(j) = 'Occupational TLV (ACGIH, 1992)'
      OPEN(UNIT=kin,FILE='DATAS1/acgih.1992',STATUS='old')
      n = 56
      DO i = 1, n
         READ(kin,*) x1(i), y1(i)
         y1(i) = y1(i)
      ENDDO
      CLOSE (kin)

      CALL terint(nw,wl,yg, n,x1,y1, 1,0)
      DO iw = 1, nw-1
         s(j,iw) = yg(iw)
      ENDDO

********* phytoplankton, Boucher et al. (1994) 
* from Boucher, N., Prezelin, B.B., Evens, T., Jovine, R., Kroon, B., Moline, M.A.,
* and Schofield, O., Icecolors '93: Biological weighting function for the ultraviolet
*  inhibition  of carbon fixation in a natural antarctic phytoplankton community, 
* Antarctic Journal, Review 1994, pp. 272-275, 1994.
* In original paper, value of b and m (em below are given as positive.  Correct values
* are negative. Also, limit to positive values.

      j = j + 1
      label(j) = 'Phytoplankton (Boucher et al., 1994)'
      a = 112.5
      b = -6.223E-01
      c = 7.670E-04
      em = -3.17E-06
      DO iw = 1, nw-1
         IF (wc(iw) .GT. 290. .AND. wc(iw) .LT. 400.) THEN
            s(j,iw) = em + EXP(a+b*wc(iw)+c*wc(iw)*wc(iw))
         ELSE
            s(j,iw) = 0.
         ENDIF
         s(j,iw) = max(s(j,iw),0.)
      ENDDO

********* phytoplankton, Cullen et al.
* Cullen, J.J., Neale, P.J., and Lesser, M.P., Biological weighting function for the  
*  inhibition of phytoplankton photosynthesis by ultraviolet radiation, Science, 25,
*  646-649, 1992.
* phaeo

      j = j + 1
      label(j) = 'Phytoplankton, phaeo (Cullen et al., 1992)'
      OPEN(UNIT=kin,FILE='DATAS1/phaeo.bio',STATUS='old')
      n = 106
      DO i = 1, n
         READ(kin,*) idum, dum1, dum2, y1(i)
         x1(i) = (dum1+dum2)/2.
      ENDDO
      CLOSE(kin)

      CALL terint(nw,wl,yg, n,x1,y1, 1,0)
      DO iw = 1, nw-1
         s(j,iw) = yg(iw)
      ENDDO

* proro

      j = j + 1
      label(j) = 'Phytoplankton, proro (Cullen et al., 1992)'
      OPEN(UNIT=kin,FILE='DATAS1/proro.bio',STATUS='old')
      n = 100
      DO i = 1, n
         READ(kin,*) idum, dum1, dum2, y1(i)
         x1(i) = (dum1+dum2)/2.
      ENDDO
      CLOSE (kin)

      CALL terint(nw,wl,yg, n,x1,y1, 1,0)
      DO iw = 1, nw-1
         s(j,iw) = yg(iw)
      ENDDO

**** Damage to lens of pig eyes, from 
* Oriowo, M. et al. (2001). Action spectrum for in vitro
* UV-induced cataract using whole lenses. Invest. Ophthalmol. & Vis. Sci. 42,
* 2596-2602.  For pig eyes. Last two columns computed by L.O.Bjorn.

      j = j + 1
      label(j) = 'Cataract, pig (Oriowo et al., 2001)'
      OPEN(UNIT=kin,FILE='DATAS1/cataract_oriowo',STATUS='old')
      DO i = 1, 7
         READ(kin,*)
      ENDDO
      n = 18
      DO i = 1, n
         READ(kin,*) x1(i), dum1, dum1, y1(i)
      ENDDO
      CLOSE(kin)

* extrapolation to 400 nm (has very little effect on raf):
c      do i = 1, 30
c         n = n + 1
c         x1(n) = x1(n-1) + 1.
c         y1(n) = 10**(5.7666 - 0.0254*x1(n))
c      enddo

      CALL terint(nw,wl,yg, n,x1,y1, 1,0)
      DO iw = 1, nw-1
         s(j,iw) = yg(iw)
      ENDDO

****** Plant damage - Caldwell 1971
*  Caldwell, M. M., Solar ultraviolet radiation and the growth and 
* development of higher plants, Photophysiology 6:131-177, 1971.

      j = j + 1
      label(j) = 'Plant damage (Caldwell, 1971)'

* Fit to Caldwell (1971) data by 
* Green, A. E. S., T. Sawada, and E. P. Shettle, The middle 
* ultraviolet reaching the ground, Photochem. Photobiol., 19, 
* 251-259, 1974.
***(19 Jan 2015, SM: corrected typo, was 2.628 instead of 2.618)

c      DO iw = 1, nw-1
c         s(j,iw) = 2.618*(1. - (wc(iw)/313.3)**2)*
c     $        exp(-(wc(iw)-300.)/31.08)
c         IF( s(j,iw) .LT. 0. .OR. wc(iw) .GT. 313.) THEN
c            s(j,iw) = 0.
c         ENDIF
c      ENDDO

* Alternative fit to Caldwell (1971) by 
* Micheletti, M. I. and R. D. Piacentini, Irradiancia espetral solar UV-B y su
* relacion con la efectivdad de dano biologico a las plantas, Anales AFA 
* (Fisica Argentina), vol.13, 242-248, 2002. 


      a0 = 570.25
      a1 = -4.70144
      a2 = 0.01274
      a3 = -1.13118E-5
      DO iw = 1, nw-1
         s(j,iw) = a0 + a1*wc(iw) + a2*wc(iw)**2  + a3*wc(iw)**3
         IF( s(j,iw) .LT. 0. .OR. wc(iw) .GT. 313.) THEN
            s(j,iw) = 0.
         ENDIF
      ENDDO

****** Plant damage - Flint & Caldwell 2003
*  Flint, S. D. and M. M. Caldwell, A biological spectral weigthing
*  function for ozone depletion research with higher plants, Physiologia
*  Plantorum, v.117, 137-144, 2003.
*  Data available to 366 nm

      j = j + 1
      label(j) = 'Plant damage,Flint&Caldwell,2003,orig.'

      DO iw = 1, nw-1
         s(j,iw) = EXP( 4.688272*EXP(
     $        -EXP(0.1703411*(wc(iw)-307.867)/1.15))+
     $        ((390-wc(iw))/121.7557-4.183832) )

* put on per joule (rather than per quantum) basis:

         s(j,iw) = s(j,iw) * wc(iw)/300.

         IF( s(j,iw) .LT. 0. .OR. wc(iw) .GT. 366.) THEN
            s(j,iw) = 0.
         ENDIF
         
      ENDDO

* Version below allows extrapolation to 390 nm.  Neither truncation at 366 nm nor
* extrapolation to 390 can be justified, but consistency may be important.
* User Beware.

      j = j + 1
      label(j) = 'Plant damage,Flint&Caldwell,2003,ext390'

      DO iw = 1, nw-1
         s(j,iw) = EXP( 4.688272*EXP(
     $        -EXP(0.1703411*(wc(iw)-307.867)/1.15))+
     $        ((390-wc(iw))/121.7557-4.183832) )

* put on per joule (rather than per quantum) basis:

         s(j,iw) = s(j,iw) * wc(iw)/300.

         IF( s(j,iw) .LT. 0. .OR. wc(iw) .GT. 390.) THEN
            s(j,iw) = 0.
         ENDIF
         
      ENDDO

****** Vitamin D - CIE 2006
* Action spectrum for the production fo previtamin-D3 in human skin, 
* CIE Techincal Report TC 6-54, Commission Internatinale del'Eclairage, 2006.
* Wavelength range of data is 252-330 nm, but Values below 260 nm and beyond 
* 315 nm were interpolated by CIE using a spline fit.
* TUV also assigns the 252nm value to shorter wavelengths, and zero 
* beyond 330nm.

      j = j + 1
      label(j) = 'Previtamin-D3 (CIE 2006)'

      OPEN(UNIT=kin,FILE='DATAS1/vitamin_D.txt',STATUS='old')
      DO i = 1, 7
         READ(kin,*)
      ENDDO
      n = 79
      DO i = 1, n
         READ(kin,*) x1(i), y1(i)
      ENDDO
      CLOSE(kin)

      CALL terint(nw,wl,yg, n,x1,y1, 1,0)
      DO iw = 1, nw-1
         s(j,iw) = yg(iw)
      ENDDO

****** Non-melanoma skin cancer, CIE 2006.
* Action spectrum for the induction of non-melanoma skin cancer. From:
* Photocarcinogenesis Action Spectrum (Non-Melanoma Skin Cancers), 
* CIE S 019/E:2006, Commission Internationale de l'Eclairage, 2006.
* 1 nm spacing from 250 to 400 nm. Normalized at maximum, 299 nm.
* Set constanta at 3.94E-04 between 340 and 400 nm.
* Assume zero beyond 400 nm.
* Assume constant below 250 nm.

      j = j + 1
      label(j) = 'NMSC (CIE 2006)'

      OPEN(UNIT=kin,FILE='DATAS1/nmsc_cie.txt',STATUS='old')
      DO i = 1, 7
         READ(kin,*)
      ENDDO
      n = 151
      DO i = 1, n
         READ(kin,*) x1(i), y1(i)
      ENDDO
      CLOSE(kin)

      CALL terint(nw,wl,yg, n,x1,y1, 1,0)
      DO iw = 1, nw-1
         s(j,iw) = yg(iw)
      ENDDO

! virus inactivation for testing purposes only, disable in web calculatro
       
      GOTO 999

****** Virus inactivation
* Action spectrum for the inactivation of various viruses.  The action spectrum
* shape is estimated to be the same for all viruses.  It is normalized at 254 nm.
* The energy to induce 90% damage is also reported for 254 nm, for different viruses
* From: Lytle, C. D., and J.-L. Sagripanti, Predicted Inactivation of Viruses of 
* Relevance to Biodefense by Solar Radiation, Journal of Virology, Nov. 2005, 
* p. 14244-14252, doi:10.1128/JVI.79.22.14244-14252.2005.  Data from their figure 1.

* Assume zero beyond 320 nm.
* Assume constant below 254 nm.

      j = j + 1
      label(j) = 'Ebola virus inactivation'

      OPEN(UNIT=kin,FILE='DATAS1/virus.bio',STATUS='old')
      DO i = 1, 3
         READ(kin,*)
      ENDDO
      n = 12
      DO i = 1, n
         READ(kin,*) x1(i), y1(i)
      ENDDO
      CLOSE(kin)

*UV Inactivation Energies at 254 nm, D10 (10% survival) = 2.303 D37 (e-fold)
* Ebola:  17 J/m2
* Variola:  25 J/m2
* Hanta virus: 28 J/m2
* Irradiance is J m-2 s-1 nm-1
* integrated over vavelength, divided by D10: = Frequency (s-1)
* multiply by 3600 to get per hour

      CALL terint(nw,wl,yg, n,x1,y1, 1,0)
      DO iw = 1, nw-1
         s(j,iw) = 3600.*yg(iw)/17.
      ENDDO

****************************************************************
****************************************************************
*_______________________________________________________________________

 999  CONTINUE
      IF (j .GT. ks) STOP '1001'
*_______________________________________________________________________

      RETURN
      END
