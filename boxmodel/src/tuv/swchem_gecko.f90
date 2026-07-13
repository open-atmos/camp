
!> \description Load various "weighting functions", i.e. products of cross
!! section and quantum yield at each altitude and each wavelength.
!! The altitude dependence is necessary to ensure the consideration of
!! pressure and temperature dependence of the cross sections or quantum
!! yields.
!!
!! NOTE: The routine here set identical xs*qy at each altitude levels,
!! no T and M parameters being considered in the current input gecko
!! files.
!!
!! \authors subroutine borrowed from the TUV version modified for GECKO use
!! Provided by Bernard Aumont and Richard Valorso (LISA)
SUBROUTINE swgecko(nw,wl,nz,tlev,airden,jgi,sqg,jglabel2)
  use tuv_params

  IMPLICIT NONE


  INTEGER,INTENT(IN)  :: nw          ! # of specified intervals + 1 in wl grid
  REAL,INTENT(IN)     :: wl(kw)      ! lower limits of intervals in wavelength grid
  INTEGER,INTENT(IN)  :: nz          ! # of altitude levels in working alt. grid
  REAL,INTENT(IN)     :: tlev(kz)    ! temperature (K) at each altitude level
  REAL,INTENT(IN)     :: airden(kz)  ! air density (molec/cc) at each altitude level
  INTEGER,INTENT(INOUT) :: jgi         ! # of gecko J (weighting functions) defined
  CHARACTER(LEN=50)             :: jglabel1(kj) ! photolysis rx (string only for ref) !CMV: unused here, might be useful later
  CHARACTER(LEN=50), intent(out)    :: jglabel2(kj)   ! label # for identification (in boxmodel)
  REAL,INTENT(OUT)    :: sqg(kj,kz,kw)  ! xs*qy (cm^2) for each photolysis reaction,
  ! at each wl and at each altitude level

! local:
  REAL                :: wc(kw)
  INTEGER             :: iw
  CHARACTER(LEN=50)   :: infile

! complete wavelength grid
  DO iw = 1, nw - 1
    wc(iw) = (wl(iw) + wl(iw+1))/2.
  ENDDO

!=======================================================================
! label 00002:  O3 -> O1D
! Cross section:  SAPRC99  -  Quantum yield:  SAPRC99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'O3 -> O1D'
  jglabel2(jgi) = "2"
  infile='l00002_O3O1D.PHD'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 00003:  O3 -> O3P
! Cross section:  SAPRC99  -  Quantum yield:  SAPRC99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'O3 -> O3P'
  jglabel2(jgi) = "3"
  infile='l00003_O3O3P.PHD'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 00004:  NO2 + hv -> NO + O(3P)
! Cross section:  SAPRC99  -  Quantum yield:  SAPRC99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'NO2 -> NO + O(3P)'
  jglabel2(jgi) = "4"
  infile='l00004_NO2.PHD'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 00005:  NO3 + hv -> NO + O2
! Cross section:  SAPRC99  -  Quantum yield:  SAPRC99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'NO3 -> NO + O2'
  jglabel2(jgi) = "5"
  infile='l00005_NO3NO.PHD'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 00006:  NO3 + hv -> NO2 + O(3P)
! Cross section:  SAPRC99  -  Quantum yield:  SAPRC99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'NO3 -> NO2 + O(3P)'
  jglabel2(jgi) = "6"
  infile='l00006_NO3NO2.PHD'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 00011:  H2O2 -> 2 OH
! Cross section:  SAPRC99  -  Quantum yield:  SAPRC99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'H2O2 -> 2 OH'
  jglabel2(jgi) = "11"
  infile='l00011_H2O2.PHD'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 00012:  HONO -> NO + OH
! Cross section:  Stutz 00
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'HONO -> NO + OH'
  jglabel2(jgi) = "12"
  infile='l00012_HONO.txt'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 00013:  HNO3 -> NO2 + OH
! Cross section:  SAPRC99  -  Quantum yield:  SAPRC99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'HNO3 -> NO2 + OH'
  jglabel2(jgi) = "13"
  infile='l00013_HNO3.PHD'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 00014:  HNO4 -> 0.61 NO2 + 0.61 HO2 + 0.39 NO3 + 0.39 OH
! Cross section:  SAPRC99  -  Quantum yield:  SAPRC99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'HNO4 -> 0.61 NO2+ 0.61 HO2+ 0.39 NO3+ 0.39 OH'
  jglabel2(jgi) = "14"
  infile='l00014_HO2NO2.PHD'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 00017:  HCHO -> 2 HO2 + CO
! Cross section:  SAPRC99  -  Quantum yield:  SAPRC99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'HCHO -> 2 HO2 + CO'
  jglabel2(jgi) = "17"
  infile='l00017_HCHO_M.PHD'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 00110:  HCHO -> H2 + CO
! Cross section:  SAPRC99  -  Quantum yield:  SAPRC99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'HCHO -> H2 + CO'
  jglabel2(jgi) = "110"
  infile='l00110_HCHO_R.PHD'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 00100:  CH3ONO2 -> CH3O. + NO2
! Cross section:  IUPAC99  -  Quantum yield:  IUPAC99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3ONO2 -> CH3O + NO2'
  jglabel2(jgi) = "100"
  infile='l00100_methylnitrate.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 00200:  C2H5ONO2 -> C2H5O. + NO2
! Cross section:  IUPAC99  -  Quantum yield:  IUPAC99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'C2H5ONO2 -> C2H5O. + NO2'
  jglabel2(jgi) = "200"
  infile='l00200_ethylnitrate.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 00300:  n_C3H7ONO2 -> n_C3H7O. + NO2
! Cross section:  IUPAC99  -  Quantum yield:  IUPAC99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'n_C3H7ONO2 -> n_C3H7O. + NO2'
  jglabel2(jgi) = "300"
  infile='l00300_n_propylnitrate.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 00400:  i_C3H7ONO2 -> i_C3H7O. + NO2
! Cross section:  IUPAC99  -  Quantum yield:  IUPAC99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'i_C3H7ONO2 -> i_C3H7O. + NO2'
  jglabel2(jgi) = "400"
  infile='l00400_ipropylnitrate.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 00500:  1_C4H9ONO2 -> 1_C4H9O. + NO2
! Cross section:  IUPAC99  -  Quantum yield:  IUPAC99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = '1_C4H9ONO2 -> 1_C4H9O. + NO2'
  jglabel2(jgi) = "500"
  infile='l00500_1butylnitrate.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 00600:  2_C4H9ONO2 -> 2_C4H9O. + NO2
! Cross section:  IUPAC99  -  Quantum yield:  IUPAC99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = '2_C4H9ONO2 -> 2_C4H9O. + NO2'
  jglabel2(jgi) = "600"
  infile='l00600_2butylnitrate.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 00700:  tert_C4H9ONO2 -> tert_C4H9O. + NO2
! Cross section:  Roberts 89  -  Quantum yield:  Roberts 89
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'tert_C4H9ONO2 -> tert_C4H9O. + NO2'
  jglabel2(jgi) = "700"
  infile='l00700_tertbutyl.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 00800:  1_C5H11ONO2 -> 1_C5H11H9O. + NO2
! Cross section:  Zhu 97  -  Quantum yield:  Zhu 97
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = '1_C5H11ONO2 + hv -> 1_C5H11H9O. + NO2'
  jglabel2(jgi) = "800"
  infile='l00800_npentylnitrate.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 00900:  2_C5H11ONO2 -> 2_C5H11H9O. + NO2
! Cross section:  Roberts 89  -  Quantum yield:  Roberts 89
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = '2_C5H11ONO2 + hv -> 2_C5H11H9O. + NO2'
  jglabel2(jgi) = "900"
  infile='l00900_2_pentylN.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 01000:  3_C5H11ONO2 -> 3_C5H11H9O. + NO2
! Cross section:  Roberts 89  -  Quantum yield:  Roberts 89
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = '3_C5H11ONO2 + hv -> 3_C5H11H9O. + NO2'
  jglabel2(jgi) = "1000"
  infile='l01000_3_pentylN.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 01100:  2methyl1propylnitrate -> RO. + NO2
! Cross section:  Clemitshaw 97  -  Quantum yield:  Clemitshaw 97
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = '2methyl1propylnitrate + hv -> RO. + NO2'
  jglabel2(jgi) = "1100"
  infile='l01100_2methyl1propylnitrate.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 01200:  CH3OONO2 -> CH3OO. + NO2
! Cross section: IUPAC99  -  Quantum yield: no data (0.5 each channel)
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3OONO2 + hv -> CH3OO. + NO2'
  jglabel2(jgi) = "1200"
  infile='l01200_CH3O2NO2.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 01300:  CH3OONO2 -> CH3O. + NO3.
! Cross section: IUPAC99  -  Quantum yield: no data (0.5 each channel)
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3OONO2 + hv -> CH3O. + NO3.'
  jglabel2(jgi) = "1300"
  infile='l01300_CH3O2NO2.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 01400:  PAN -> CH3C(O)OO. + NO2
! Cross section: IUPAC99  -  Quantum yield: IUPAC99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'PAN + hv -> CH3C(O)OO. + NO2'
  jglabel2(jgi) = "1400"
  infile='l01400_PAN.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 01700:  CH3CHO + hv -> CH3. + CHO.
! Cross section: IUPAC99  -  Quantum yield: IUPAC99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3CHO + hv -> CH3. + CHO.'
  jglabel2(jgi) = "1700"
  infile='l01700_acetal_R.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 01800:  C2H5CHO + hv -> C2H5. + CHO.
! Cross section: IUPAC99  -  Quantum yield: IUPAC99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'C2H5CHO + hv -> C2H5. + CHO.'
  jglabel2(jgi) = "1800"
  infile='l01800_propanal.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 01900:  n-C3H7CHO + hv -> n-C3H7. +  CHO.
! Cross section: Martinez 92  -  Quantum yield: Moortgat 99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'n-C3H7CHO + hv -> n-C3H7. +  CHO.'
  jglabel2(jgi) = "1900"
  infile='l01900_n_butyraldehyde_rad'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 02000:  n-C3H7CHO + hv -> C2H4 + CH3CHO Norrish II
! Cross section: Martinez 92  -  Quantum yield: Moortgat 99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'n-C3H7CHO + hv -> C2H4 + CH3CHO Norrish II'
  jglabel2(jgi) = "2000"
  infile='l02000_n_butyraldehyde_mol'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 02100:  i-C3H7CHO + hv -> C3H7. + CHO.
! Cross section: Desai 86  -  Quantum yield: Desai 86
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'i-C3H7CHO + hv -> C3H7. + CHO.'
  jglabel2(jgi) = "2100"
  infile='l02100_i_butyraldehyde_R.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 02300:  nC4H9CHO + hv -> C4H9. +  CHO.
! Cross section: ZHU 99   -  Quantum yield: ZHU 99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'nC4H9CHO + hv -> C4H9. +  CHO.'
  jglabel2(jgi) = "2300"
  infile='l02300_n_pentanal_rad'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 02400:  nC4H9CHO + hv -> HCO.
! Cross section: ZHU 99   -  Quantum yield: Moortgat 99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'nC4H9CHO + hv -> HCO.'
  jglabel2(jgi) = "2400"
  infile='l02400_n_pentanal_Norrish'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 02500:  iC4H9CHO + hv -> C4H9. + CHO.
! Cross section: ZHU 99   -  Quantum yield: ZHU 99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'iC4H9CHO + hv -> C4H9. + CHO.'
  jglabel2(jgi) = "2500"
  infile='l02500_i_pentanal_rad'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 02600:  iC4H9CHO + hv -> CH3CHO +  CH2=CHCH3
! Cross section: ZHU 99 -  Quantum yield: Moortgat 99 for n-pentanal (0.63*0.29)
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'iC4H9CHO + hv -> CH3CHO +  CH2=CHCH3'
  jglabel2(jgi) = "2600"
  infile='l02600_i_pentanal_Norrish'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 02700:  t-C4H9CHO + hv -> HCO. +  t-C4H9.
! Cross section: ZHU 99   -  Quantum yield: ZHU 99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 't-C4H9CHO + hv -> HCO. +  t-C4H9.'
  jglabel2(jgi) = "2700"
  infile='l02700_t_pentanal_rad'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 02800:  C5H11CHO + hv -> HCO. + n-C5H11.
! Cross section: Plagens 98   -  Quantum yield: Moortgat 99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'C5H11CHO + hv -> HCO. + n-C5H11.'
  jglabel2(jgi) = "2800"
  infile='l02800_hexanal_rad'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 02900:  C5H11CHO + hv -> CH3CHO + CH2=CHCH2CH3
! Cross section: Plagens 98   -  Quantum yield: Moortgat 99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'C5H11CHO + hv -> CH3CHO + CH2=CHCH2CH3'
  jglabel2(jgi) = "2900"
  infile='l02900_hexanal_norrish'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 03000:  Acetone + hv -> CH3CO + CH3
! Cross section: IUPAC99   -  Quantum yield: IUPAC99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'Acetone + hv -> CH3CO + CH3'
  jglabel2(jgi) = "3000"
  infile='l03000_acetone.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 03100:  CH3COC2H5 + hv -> C2H5. + CH3CO.
! Cross section: IUPAC99 - Quantum yield: Raber et Moortgat 95 (0.34*0.85)
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3COC2H5 + hv -> C2H5. + CH3CO.'
  jglabel2(jgi) = "3100"
  infile='l03100_2butanone_rad1'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 03200:  CH3COC2H5 + hv -> CH3. + C2H5CO.
! Cross section: IUPAC99 - Quantum yield: Raber et Moortgat 95 (0.34*0.15)
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3COC2H5 + hv -> CH3. + C2H5CO.'
  jglabel2(jgi) = "3200"
  infile='l03200_2butanone_rad2'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 03300:  CH3COC3H7 + hv -> C3H7. + CH3CO.
! Cross section: Martinez 92 -  Quantum yield: Raber 95 (for 2-butanone)
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3COC3H7 + hv -> C3H7. + CH3CO.'
  jglabel2(jgi) = "3300"
  infile='l03300_2pentanone_rad'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 03400:  C2H5COC2H5 + hv -> C2H5CO. + C2H5.
! Cross section: Martinez 92 -  Quantum yield: Raber 95 (for 2-butanone)
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'C2H5COC2H5 + hv -> C2H5CO. + C2H5.'
  jglabel2(jgi) = "3400"
  infile='l03400_3pentanone.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 03500:  (CH3)2CHCOCH(CH3)2 + hv -> (CH3)2CHCO(.) + C(.)H(CH3)2
! Cross section: Yujing 2000 -  Quantum yield: Raber 95 (for 2-butanone)
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = '(CH3)2CHCOCH(CH3)2 + hv -> (CH3)2CHCO(.) + C(.)H(CH3)2'
  jglabel2(jgi) = "3500"
  infile='l03500_24dimethyl3pentanone.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 03600:  (CH3)2CHCOCH(CH3)2 + hv -> (CH3)2CHCO(.) + C(.)H(CH3)2
! Cross section: Yujing 2000 - Quantum yield: Raber 95 (for 2-butanone) 0.34*0.3
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3COCH2CH(CH3)2 + hv -> CH3CO. + CH2(.)CH(CH3)2'
  jglabel2(jgi) = "3600"
  infile='l03600_4methyl2pentanone_rad'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 03700:  CH3COCH2CH(CH3)2 + hv -> CH3COCH3 + CH2=CHCH3
! Cross section: Yujing 2000 - Quantum yield: Raber 95 (for 2-butanone) 0.34*0.7
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3COCH2CH(CH3)2 + hv -> CH3COCH3 + CH2=CHCH3'
  jglabel2(jgi) = "3700"
  infile='l03700_4methyl2pentanone_Nor'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 03800:  CH3COCH2CH2CH(CH3)2 + hv -> CH3CO. + .CH2CH2CH(CH3)2
! Cross section: Yujing 2000 - Quantum yield: Raber 95 (for 2-butanone) 0.34*0.3
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3COCH2CH2CH(CH3)2 + hv -> CH3CO. + .CH2CH2CH(CH3)2'
  jglabel2(jgi) = "3800"
  infile='l03800_5methyl2hexanone_rad'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 03900:  CH3COCH2CH2CH(CH3)2 + hv -> CH3COCH3 + CH2=C(CH3)2
! Cross section: Yujing 2000 - Quantum yield: Raber 95 (for 2-butanone) 0.34*0.7
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3COCH2CH2CH(CH3)2 + hv -> CH3COCH3 + CH2=C(CH3)2'
  jglabel2(jgi) = "3900"
  infile='l03900_5methyl2hexanone_Nor'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 04000:  CH2=CHCHO + hv -> CH2CH=CHCO. + HO2.
! Cross section: Gardner 87 - Quantum yield: Magneron 99 *0.4
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH2=CHCHO + hv -> CH2CH=CHCO. + HO2.'
  jglabel2(jgi) = "4000"
  infile='l04000_acrolein0_4.txt'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 04100:  CH2=CHCHO + hv -> CH2=CH. + CHO.
! Cross section: Gardner 87 - Quantum yield: Magneron 99  *0.3
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH2=CHCHO + hv -> CH2=CH. + CHO.'
  jglabel2(jgi) = "4100"
  infile='l04100_acrolein0_3.txt'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 04200:  CH2=CHCHO + hv -> CH3C(.)(.)H + CO
! Cross section: Gardner 87 - Quantum yield: Magneron 99  *0.3
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH2=CHCHO + hv -> CH3C(.)(.)H + CO'
  jglabel2(jgi) = "4200"
  infile='l04200_acrolein0_3.txt'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 04300:  CH2=C(CH3)CHO + hv -> CH2=C(.)CH3 +CHO.
! Cross section: Raber 95 -  Quantum yield: 0.02 * 0.34
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH2=C(CH3)CHO + hv -> CH2=C(.)CH3 +CHO.'
  jglabel2(jgi) = "4300"
  infile='l04300_MACR_034'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 04400:  CH2=C(CH3)CHO + hv -> CH3C(.)(.)CH3 + CO
! Cross section: Raber 95 -  Quantum yield: 0.02 * 0.33
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH2=C(CH3)CHO + hv -> CH3C(.)(.)CH3 + CO'
  jglabel2(jgi) = "4400"
  infile='l04400_MACR_033'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 04500:  CH2=C(CH3)CHO + hv -> CH2=C(CH3)C(.)O
! Cross section: Raber 95 -  Quantum yield: 0.02 * 0.33
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH2=C(CH3)CHO + hv -> CH2=C(CH3)C(.)O'
  jglabel2(jgi) = "4500"
  infile='l04500_MACR_033'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 04600:  CH3CH=CHCHO + hv -> CH3CH=CH(.) + CHO.
! Cross section: Magneron 1999 -  Quantum yield: Magneron 1999
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3CH=CHCHO + hv -> CH3CH=CH(.) + CHO.'
  jglabel2(jgi) = "4600"
  infile='l04600_crotonaldehyde'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 04700:  CH3CH=CHCHO + hv -> CH2CH=CHC(.)O + H.
! Cross section: Magneron 1999 -  Quantum yield: Magneron 1999
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3CH=CHCHO + hv -> CH2CH=CHC(.)O + H.'
  jglabel2(jgi) = "4700"
  infile='l04700_crotonaldehyde'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 04800:  CH3CH=CHCHO + hv -> CH3CH2CH(.)(.)H + CO
! Cross section: Magneron 1999 -  Quantum yield: Magneron 1999
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3CH=CHCHO + hv -> CH3CH2CH(.)(.)H + CO'
  jglabel2(jgi) = "4800"
  infile='l04800_crotonaldehyde'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 04900:  CH2=CHCOCH3 + hv -> CH2=CHCH3 + CO
! Cross section: Raber 1995 -  Quantum yield: Raber 1995
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH2=CHCOCH3 + hv -> CH2=CHCH3 + CO'
  jglabel2(jgi) = "4900"
  infile='l04900_MVK094'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! label 05000:  CH2=CHCOCH3 + hv -> CH2=CH. + CH3CO.
! Cross section: Raber 1995 -  Quantum yield: Raber 1995
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH2=CHCOCH3 + hv -> CH2=CH. + CH3CO.'
  jglabel2(jgi) = "5000"
  infile='l05000_MVK006'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 05100:  CHOCHO + hv -> H2 + 2 CO
! Cross section: IUPAC 99 -  Quantum yield: Magneron 99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CHOCHO + hv -> H2 + 2CO'
  jglabel2(jgi) = "5100"
  infile='l05100_glyoxal_phi1.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 05200:  CHOCHO  + hv -> 2 CHO.
! Cross section: IUPAC 99 -  Quantum yield: Magneron 99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CHOCHO  + hv -> 2 CHO.'
  jglabel2(jgi) = "5200"
  infile='l05200_glyoxal_phi2.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 05300:  CHOCHO + hv -> HCHO + CO
! Cross section: IUPAC 99 -  Quantum yield: Magneron 99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CHOCHO + hv -> HCHO + CO'
  jglabel2(jgi) = "5300"
  infile='l05300_glyoxal_phi3.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 05400:  CH3COCHO + hv -> CHO. + CH3CO.
! Cross section: Calvert 2000  -  Quantum yield: Raber 95
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3COCHO + hv -> CHO. + CH3CO.'
  jglabel2(jgi) = "5400"
  infile='l05400_Mglyoxal090.txt'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 05500:  CH3COCHO + hv -> CO + CH3CHO
! Cross section: Calvert 2000  -  Quantum yield: Raber 95
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3COCHO + hv -> CO + CH3CHO'
  jglabel2(jgi) = "5500"
  infile='l05500_Mglyoxal005.txt'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 05600:  CH3COCHO + hv -> 2 CO + CH4
! Cross section: Calvert 2000  -  Quantum yield: Raber 95
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3COCHO + hv -> 2 CO + CH4'
  jglabel2(jgi) = "5600"
  infile='l05600_Mglyoxal005.txt'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 05700:  CH3COCOCH3 + hv -> 2 CH3CO.
! Cross section: Plum 83   -  Quantum yield: SAPRC 99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3COCOCH3 + hv -> 2 CH3CO.'
  jglabel2(jgi) = "5700"
  infile='l05700_BACL_ADJ.PHD'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 05800:  CHOCH=CHCH=CHCHO + hv ->  CHO. + .CH=CHCH=CHCHO
! Cross section: Klotz 95  -  Quantum yield:
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CHOCH=CHCH=CHCHO + hv ->  CHO. + .CH=CHCH=CHCHO'
  jglabel2(jgi) = "5800"
  infile='l05800_EEhexadienedial'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 05900:  CHOCH=CHCH=CHCHO + hv ->  CHOCH=CHCH=CHCO. + HO2.
! Cross section: Klotz 95  -  Quantum yield:
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CHOCH=CHCH=CHCHO + hv ->  CHOCH=CHCH=CHCO. + HO2.'
  jglabel2(jgi) = "5900"
  infile='l05900_EEhexadienedial'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 06100:  HOC(CH3)=CHCH=CHCHO -> CHOC(CH3)=CHCH=CH. + .CHO
! Cross section: Klotz 95  -  Quantum yield:
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'HOC(CH3)=CHCH=CHCHO -> CHOC(CH3)=CHCH=CH. + .CHO'
  jglabel2(jgi) = "6100"
  infile='l06100_EE2methylhexadienedial.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 06200:  HOC(CH3)=CHCH=CHCHO -> CHOC(CH3)=CHCH=CH. + .CHO
! Cross section: Klotz 95  -  Quantum yield:
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CHOC(CH3)=CHCH=CHCHO -> CHO. + .C(CH3)=CHCH=CHCHO'
  jglabel2(jgi) = "6200"
  infile='l06200_EE2methylhexadienedial.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 06300:  CHOC(CH3)=CHCH=CHCHO -> CHOC(CH3)=CHCH=CHCO. + HO2.
! Cross section: Klotz 95  -  Quantum yield:
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CHOC(CH3)=CHCH=CHCHO -> CHOC(CH3)=CHCH=CHCO. + HO2.'
  jglabel2(jgi) = "6300"
  infile='l06300_EE2methylhexadienedial.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 06400:  CHOC(CH3)=CHCH=CHCHO -> CHOC(CH3)=CHCH=CHCO. + HO2.
! Cross section: Klotz 95  -  Quantum yield:
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CHOC(CH3)=CHCH=CHCHO -> CHOC(CH3)=CHCH=CHCO. + HO2.'
  jglabel2(jgi) = "6400"
  infile='l06400_EE2methylhexadienedial.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 06500:  CH3CH(ONO2)CH2(ONO2) + hv -> CH3CH(O.)CH2(ONO2) + NO2
! Cross section: Barnes 93 -  Quantum yield:
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3CH(ONO2)CH2(ONO2) + hv -> CH3CH(O.)CH2(ONO2) + NO2'
  jglabel2(jgi) = "6500"
  infile='l06500_D1'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 06600:  CH3CH(ONO2)CH2(ONO2) + hv -> products
! Cross section: Barnes 93 -  Quantum yield:
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3CH(ONO2)CH2(ONO2) + hv -> products'
  jglabel2(jgi) = "6600"
  infile='l06600_D1'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 06700:  CH3CH2CH(ONO2)CH2(ONO2) -> CH3CH2CH(O.)CH2(ONO2) + NO2
! Cross section: Barnes 93 -  Quantum yield:
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3CH2CH(ONO2)CH2(ONO2) -> CH3CH2CH(O.)CH2(ONO2) + NO2'
  jglabel2(jgi) = "6700"
  infile='l06700_D2.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 06800:  CH3CH2CH(ONO2)CH2(ONO2) -> CH3CH2CH(ONO2)CH2O. + NO2
! Cross section: Barnes 93 -  Quantum yield:
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3CH2CH(ONO2)CH2(ONO2) -> CH3CH2CH(ONO2)CH2O. + NO2'
  jglabel2(jgi) = "6800"
  infile='l06800_D2.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 06900:  CH3CH(ONO2)CH(ONO2)CH3 -> CH3CH(O.)CH(ONO2)CH3 + NO2
! Cross section: Barnes 93 -  Quantum yield:
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3CH(ONO2)CH(ONO2)CH3 -> CH3CH(O.)CH(ONO2)CH3 + NO2'
  jglabel2(jgi) = "6900"
  infile='l06900_D3.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 07000:  CH2(ONO2)CH(ONO2)CH=CH2 -> CH2(ONO2)CH(O.)CH=CH2 + NO2
! Cross section: Barnes 93 -  Quantum yield:
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH2(ONO2)CH(ONO2)CH=CH2 -> CH2(ONO2)CH(O.)CH=CH2 + NO2'
  jglabel2(jgi) = "7000"
  infile='l07000_D4.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 07100:  CH2(ONO2)CH(ONO2)CH=CH2 -> CH2(O.)CH(ONO2)CH=CH2 + NO2
! Cross section: Barnes 93 -  Quantum yield:
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH2(ONO2)CH(ONO2)CH=CH2 -> CH2(O.)CH(ONO2)CH=CH2 + NO2'
  jglabel2(jgi) = "7100"
  infile='l07100_D4.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 07200:  CH2(ONO2)CH=CHCH2(ONO2) -> CH2(ONO2)CH=CHCH2O. + NO2
! Cross section: Barnes 93 -  Quantum yield:
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH2(ONO2)CH=CHCH2(ONO2) -> CH2(ONO2)CH=CHCH2O. + NO2'
  jglabel2(jgi) = "7200"
  infile='l07200_D5.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 07300:  CH3OOH + hv -> CH3O.+ OH.
! Cross section: IUPAC99 -  Quantum yield: IUPAC99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3OOH + hv -> CH3O.+ OH.'
  jglabel2(jgi) = "7300"
  infile='l07300_CH3OOH.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 07400:  OHCH2CHO + hv -> OHCH2. + .CHO
! Cross section: Tyndall 99 -  Quantum yield: Tyndall 99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'OHCH2CHO + hv -> OHCH2. + .CHO'
  jglabel2(jgi) = "7400"
  infile='l07400_glycolaldehyde'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 07500:  OHCH2CH2ONO2 + hv -> OHCH2CH2O. + NO2
! Cross section: Roberts 89 -  Quantum yield: Roberts 89
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'OHCH2CH2ONO2 + hv -> OHCH2CH2O. + NO2'
  jglabel2(jgi) = "7500"
  infile='l07500_NOE.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 07600:  OHCH2COCH3 + hv -> OHCH2. + .COCH3
! Cross section: Orlando 99 -  Quantum yield:
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'OHCH2COCH3 + hv -> OHCH2. + .COCH3'
  jglabel2(jgi) = "7600"
  infile='l07600_hydroxyacetone.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 07700:  OHCH2OOH + hv -> OHCH2O. + .OH
! Cross section: Bauerle 99 -  Quantum yield:
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'OHCH2OOH + hv -> OHCH2O. + .OH'
  jglabel2(jgi) = "7700"
  infile='l07700_HMHP.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 07800:  CHOCOOH + hv -> CO2 + HCHO
! Cross section: Back 1985 -  Quantum yield: Back 1985
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CHOCOOH + hv -> CO2 + HCHO'
  jglabel2(jgi) = "7800"
  infile='l07800_Glyoxylic_1.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 07900:  CHOCOOH + hv -> 2 CO + H2O
! Cross section: Back 1985 -  Quantum yield: Back 1985
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CHOCOOH + hv -> 2 CO + H2O'
  jglabel2(jgi) = "7900"
  infile='l07900_Glyoxylic_2.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 08000:  COOH-COOH + hv -> CO2 + HCOOH
! Cross section: Yamamoto 1985 -  Quantum yield: Yamamoto 1985
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'COOH-COOH + hv -> CO2 + HCOOH'
  jglabel2(jgi) = "8000"
  infile='l08000_Oxalic_1.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 08100:  COOH-COOH + hv -> CO2 + CO + H2O
! Cross section: Yamamoto 1985 -  Quantum yield: Yamamoto 1985
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'COOH-COOH + hv -> CO2 + CO + H2O'
  jglabel2(jgi) = "8100"
  infile='l08100_Oxalic_2.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 08200:  CH3COCOOH + hv -> CO2 + CH3CHO
! Cross section: Yamamoto 1985 -  Quantum yield: Yamamoto 1985
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3COCOOH + hv -> CO2 + CH3CHO'
  jglabel2(jgi) = "8200"
  infile='l08200_Pyruvic.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

! ---- SURROGATES ---

!=======================================================================
! Label 10100: primary nitrate, use: 1_C5H11ONO2 + hv -> 1_C5H11H9O. + NO2
! Cross section:  Zhu 97  -  Quantum yield:  Zhu 97
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = '1_C5H11ONO2 + hv -> 1_C5H11H9O. + NO2'
  jglabel2(jgi) = "10100"
  infile='l10100_npentylnitrate.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 10200: secondary nitrate, use: 2_C4H9ONO2 -> 2_C4H9O. + NO2
! Cross section:  IUPAC99  -  Quantum yield:  IUPAC99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = '2_C4H9ONO2 -> 2_C4H9O. + NO2'
  jglabel2(jgi) = "10200"
  infile='l10200_2butylnitrate.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 10300: tertiary nitrate, use: tert_C4H9ONO2 -> tert_C4H9O. + NO2
! Cross section:  Roberts 89  -  Quantum yield:  Roberts 89
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'tert_C4H9ONO2 -> tert_C4H9O. + NO2'
  jglabel2(jgi) = "10300"
  infile='l10300_tertbutyl.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 10400: pernitrate, use CH3OONO2 + hv -> CH3O. + NO3.
! Cross section: IUPAC99  -  Quantum yield: no data (0.5 each channel)
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3OONO2 + hv -> CH3O. + NO3.'
  jglabel2(jgi) = "10400"
  infile='l10400_CH3O2NO2.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 10500: PAN like, use PAN + hv -> CH3C(O)OO. + NO2
! Cross section: IUPAC99  -  Quantum yield: IUPAC99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'PAN + hv -> CH3C(O)OO. + NO2'
  jglabel2(jgi) = "10500"
  infile='l10500_PAN.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 20100: RCHO (without gamma-H) photolysis and Calpha= tertiary,
!        n-RCHO (without gamma-H) + hv -> CHO. + R.
!   use: t-C4H9CHO + hv -> HCO. +  t-C4H9.
!   Cross section: t-pentanal (Zhu 99) - Quantum yield:  t-pentanal (Zhu 99)
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'n-aldehyde (no gamma-H + Calpha tert) -> CHO + R.'
  jglabel2(jgi) = "20100"
  infile='l20100_t_pentanal_rad'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 20200: RCHO (without gamma-H) photolysis and Calpha = secondary:
!        RCHO (without gamma-H) + hv -> CHO. + R.
!   use: i-C3H7CHO + hv -> C3H7. + CHO.
!   Cross section: i-butyraldehyde (Desai86) - Quantum yield: i-butyraldehyde (Desai86)
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'n-aldehyde (no gamma-H + Calpha sec) + -> R. + CHO.'
  jglabel2(jgi) = "20200"
  infile='l20200_i_butyraldehyde_R.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 20300: CHO (without gamma-H) and (Calph = primary):
!        RCHO (without gamma-H) + hv -> products
!   use: C2H5CHO + hv -> C2H5. + CHO.
!   Cross section: propanal IUPAC 99 - Quantum yield: propanal IUPAC 99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'linear aldehyde (no gamma-H + Calpha primary) -> CHO + R.'
  jglabel2(jgi) = "20300"
  infile='l20300_propanal.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 20400: RCHO (without gamma-H) and (Calpha = tertiary):
!         RCHO (with gamma-H) + hv -> R. + CHO.
!    use: t-pentanal
!    Cross section: t-pentanal Zhu 99 - Quantum yield: n-pentanal Zhu 99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'aldehyde (gamma-H + Calpha tert) + hv -> CHO.  + R.'
  jglabel2(jgi) = "20400"
  infile='l20400_ald1'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 20500: RCHO (with gamma-H) and (Calpha = tertiary):
!        RCHO (with gamma-H) + hv -> Norrish II
!   use: t-pentanal
!   Cross section: t-pentanal (Zhu 99) - Quantum yield: n-pentanal (Moortgat 99)
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'RCHO (gamma-H + Calpha tert) + hv -> Norrish II'
  jglabel2(jgi) = "20500"
  infile='l20500_ald2'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 20600: RCHO (with gamma-H) and (C alpha not tertiary):
!        RCHO (with gamma-H) + hv -> R. + CHO.
!   use: n-pentanal
!   Cross section: n-pentanal (Zhu 99) - Quantum yield: n-pentanal (Zhu 99)
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'RCHO (gamma-H + Calpha sec or prim) -> CHO. + R.'
  jglabel2(jgi) = "20600"
  infile='l20600_n_pentanal_rad'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 20700: RCHO (with gamma-H) and (Calpha = primary or secondary):
!        RCHO (with gamma-H) + hv -> Norrish II
!   use: n-pentanal
!   Cross section: n-pentanal (Zhu 99) - Quantum yield: n-pentanal (Moortgat 99)
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'RCHO (gamma-H + Calpha sec or prim) -> Norrish II'
  jglabel2(jgi) = "20700"
  infile='l20700_n_pentanal_Norrish'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 21100: RRC=CRCHO structures photolysis:
!        RRC=CRCHO structures + hv -> RRC=C(.)R + CHO.
!   use: acrolein
!   Cross section: acrolein (Gardner 87) - Quantum yield: 0.02 * 0.34
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'RRC=CRCHO structures+ hv -> CHO. + =.'
  jglabel2(jgi) = "21100"
  infile='l21100_acro_struct1.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 21200: RRC=CRCHO structures photolysis:
!        RRC=CRCHO structures + hv -> CO + Criegee
!   use: acrolein
!   Cross section: acrolein (Gardner 87) - Quantum yield: 0.02 * 0.33
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'RRC=CRCHO structures+ hv -> CO + Criegee'
  jglabel2(jgi) = "21200"
  infile='l21200_acro_struct2.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 21300: RRC=CRCHO structures photolysis:
!        RRC=CRCHO structures + hv -> H. + RRC=CRC(.)O
!   use: acrolein
!   Cross section: acrolein (Gardner 87) - Quantum yield: 0.02 * 0.33
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'RRC=CRCHO structures+ hv -> H. + RRC=CRC(.)O'
  jglabel2(jgi) = "21300"
  infile='l21300_acro_struct3.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 21400: RCOCHO photolysis:
!        RCOCHO + hv -> RC(.)O + CHO.
!   use: methyl glyoxal
!   Cross section: Calvert 2000 -  Quantum yield: Raber 95
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'RCOCHO + hv -> RC(.)O + CHO.'
  jglabel2(jgi) = "21400"
  infile='l21400_Mglyoxal1.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 21500: RCOCHO photolysis:
!        RCOCHO + hv -> CO + RCHO
!   use: methyl glyoxal
!   Cross section: Calvert 2000 - Quantum yield: Raber 95
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'RCOCHO + hv -> CO + RCHO'
  jglabel2(jgi) = "21500"
  infile='l21500_Mglyoxal2.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 21600: RCOCHO photolysis:
!        RCOCHO + hv -> 2 CO + RH
!   use: methyl glyoxal
!   Cross section: Calvert 2000 - Quantum yield: Raber 95
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'RCOCHO + hv -> 2 CO + RH'
  jglabel2(jgi) = "21600"
  infile='l21600_Mglyoxal3.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 21800: RRC(OH)CHO photolysis:
!      RRC(OH)CHO + hv -> CHO. + RRC(OH).
! use: glycolaldehyde
! Cross section: Moortgat 99 - Quantum yield: 1
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'RRC(OH)CHO + hv -> CHO. + RRC(OH).'
  jglabel2(jgi) = "21800"
  infile='l21800_glycolaldehyde'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 21900: EE-2.4 hexadienedial photolysis:
!      EE-2.4 hexadienedial + hv -> 4 ways
!   Cross section: Klotz 95 - Quantum yield: 0.30 * 0.25
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'EEhexadienedial + hv -> 4 ways'
  jglabel2(jgi) = "21900"
  infile='l21900_EEhexadienedial1.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 22000: EE-2.4 hexadienedial photolysis:
!     EE-2.4 hexadienedial + hv -> 2 different ways
!   Cross section: Klotz 95 - Quantum yield: 0.30 * 0.5
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'EEhexadienedial + hv -> 2 different ways'
  jglabel2(jgi) = "22000"
  infile='l22000_EEhexadienedial2.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 23000: RC(O)CR=CRCHO photolysis:
!     RC(O)CR=CRCHO + hv ->3H furan-2-one
!   XS: trans-butenedial (Bierbach 94) - QY: trans-butenedial (Bierbach 94)
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'RC(O)CR=CRCHO + hv ->3H furan-2-one'
  jglabel2(jgi) = "23000"
  infile='l23000_butenedial1'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 23100: RC(O)CR=CRCHO photolysis:
!     HC(O)CR=CRCHO + hv -> maleic anhydride + 2 HO2.
!   XS: trans-butenedial (Bierbach 94) - QY: trans-butenedial (Bierbach 94)
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'RC(O)CR=CRCHO + hv ->maleic anhydride + 2 HO2.'
  jglabel2(jgi) = "23100"
  infile='l23100_butenedial2'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 23200: RC(O)CR=CRCHO photolysis:
!     RaC(O)CR=CRCHO + hv -> 5(Ra)-3H-furan-2-one
!   XS: trans-butenedial (Bierbach 94) - QY: trans-butenedial (Bierbach 94)
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'RaC(O)CR=CRCHO + hv -> 5(Ra)-3H-furan-2-one'
  jglabel2(jgi) = "23200"
  infile='l23200_4oxo2pentenal1'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 23300: RC(O)CR=CRCHO photolysis:
!     RC(O)CR=CRCHO + hv -> maleic anhydride + HO2. + R.
!   XS: trans-butenedial (Bierbach 94) - QY: trans-butenedial (Bierbach 94)
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'RC(O)CR=CRCHO + hv ->maleic anhydride + HO2. + R.'
  jglabel2(jgi) = "23300"
  infile='l23300_4oxo2pentenal2'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 23400: PURPOSE: RC(O)CR=CRC(O)R photolysis:
!     RC(O)CR=CRC(O)R + hv -> 4oxo2pentenal + R.
!   XS: 4 oxo 2 pentenal Bierbach 94 - QY: 4 oxo 2 pentenal Bierbach 94
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'RC(O)CR=CRC(O)R + hv -> 4 oxo pentenal + R.'
  jglabel2(jgi) = "23400"
  infile='l23400_3hexene25dione1'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 23500: RC(O)CR=CRC(O)R photolysis:
!     RC(O)CR=CRC(O)R + hv -> maleic anhydride + R. + R.
!   XS: 3 hexene2.5dione (Bierbach 94) - QY: 3 hexene2.5dione Bierbach 94
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'RC(O)CR=CRC(O)R + hv ->maleic anhydride + R. + R.'
  jglabel2(jgi) = "23500"
  infile='l23500_3hexene25dione2'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 30100: ketone without gamma-H  photolysis:
!      n-ketone (no gammaH) -> RC(.)O + R.
!  Cross section: Martinez 92 - Quantum yield: Raber 95 (for 2-butanone)
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'n-ketone (no gammaH) + hv -> RC(.)O + R.'
  jglabel2(jgi) = "30100"
  infile='l30100_n_ketone1'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 30200: ketone (gamma-H + two primary Calpha) -> R. + RCO.
!    n-ketone + hv -> RC(.)O + R.
!  Cross section: Martinez 92 - Quantum yield: Raber 95 (for 2-butanone)
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'n-ketone + hv -> RC(.)O + R.'
  jglabel2(jgi) = "30200"
  infile='l30200_n_ketone2'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 30300:  ketone (gamma-H + two pimary Calpha) + hv -> Norrish II
!    n_ketone + hv -> Norrish II
!  Cross section: Martinez 92 - Quantum yield: Raber 95 (for 2-butanone)
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'n_ketone + hv -> Norrish II'
  jglabel2(jgi) = "30300"
  infile='l30300_n_ketone3'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 30600: ketone (no gamma-H + only 1 primary Calpha)-> R.+RCO.
!    i_ketone (no gamma-H) + hv -> RC(.)O + R.
!  Cross section: mean value between 3pentanone and 24dimethylpentanone
!  Quantum yield: Raber 95 (for 2-butanone)
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'i_ketone (no gamma-H) + hv -> RC(.)O + R.'
  jglabel2(jgi) = "30600"
  infile='l30600_i_ketone1'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 30700: ketone (gammaH and only 1 primary Calpha) -> R. + RC(.)O
!    i_ketone(with gammaH) + hv -> RC(.)O + R.
!  Cross section: mean value between 3pentanone and 24dimethylpentanone
!  Quantum yield:
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'i_ketone(with gammaH) + hv -> RC(.)O + R.'
  jglabel2(jgi) = "30700"
  infile='l30700_i_ketone2'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 30800: ketone (with gammaH)+ hv -> products
!    i_ketone + hv -> Norrish II
!  Cross section: mean value between 3pentanone and 24dimethylpentanone
!  Quantum yield:
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'i_ketone + hv -> Norrish II'
  jglabel2(jgi) = "30800"
  infile='l30800_i_ketone3'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 31100: ketone (without gamma H) + hv -> products
!    t_ketone (no gammaH) + hv -> RC(.)O + R.
!  Cross section: 24dimethylpentanone - QY: Raber 95 (for 2-butanone)
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 't_ketone (no gammaH) + hv -> RC(.)O + R.'
  jglabel2(jgi) = "31100"
  infile='l31100_t_ketone1'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 31200: ketone + hv -> R. + RC(.)O
!    t_ketone + hv -> R. + RC(.)O
!  Cross section: 24dimethylpentanone - Quantum yield:
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 't_ketone + hv -> R. + RC(.)O'
  jglabel2(jgi) = "31200"
  infile='l31200_t_ketone2'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 31300: ketone + hv -> products
!    t_ketone + hv -> Norrish II
!  Cross section: 24dimethylpentanone ! Quantum yield:
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 't_ketone + hv -> Norrish II'
  jglabel2(jgi) = "31300"
  infile='l31300_t_ketone3'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 31600: CH2=CHCOCH3 + hv -> = + CO
!    CH2=CHCOCH3 + hv -> = + CO
!  Cross section: Raber 1995 - Quantum yield: Raber 1995
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH2=CHCOCH3 + hv -> = + CO'
  jglabel2(jgi) = "31600"
  infile='l31600_MVK1'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 31700: CH2=CHCOCH3 + hv ->  =. + RC(.)O
!    CH2=CHCOCH3 + hv -> =. + RC(.)O
!  Cross section: Raber 1995 - Quantum yield: Raber 1995
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH2=CHCOCH3 + hv -> =. + RC(.)O'
  jglabel2(jgi) = "31700"
  infile='l31700_MVK2'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 31800: CH3COCOCH3 + hv -> 2 CH3CO.
!    CH3COCOCH3 + hv -> 2 CH3CO.
!  Cross section: Plum 83 - Quantum yield: SAPRC 99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3COCOCH3 + hv -> 2 CH3CO.'
  jglabel2(jgi) = "31800"
  infile='l31800_BACL_ADJ.PHD'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 31900: OHCH2COCH3 + hv -> products
!    OHCH2COR + hv -> RC(.)O + C(OH)(.)H2
!  Cross section: Orlando 99 - Quantum yield:
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'OHCH2COR + hv -> RC(.)O + C(OH)(.)H2'
  jglabel2(jgi) = "31900"
  infile='l31900_hydroxyacetone_rdt1'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 32100: RC(O)COOH + hv -> RCHO + CO2
!    RC(O)COOH + hv -> RCHO + CO2
!  Cross section: Moortgat 99 - Quantum yield: Moortgat 99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'RC(O)COOH + hv -> RCHO + CO2'
  jglabel2(jgi) = "32100"
  infile='l32100_Pyruvic.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 40000: CH3(ONO) + hv -> CH3(O.) + NO
! Cross section: Taylor 1980 -  Quantum yield: Calvert , oxidation of the oxygenates book
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3(ONO) + hv -> CH3(O.) + NO'
  jglabel2(jgi) = "40000"
  infile='l40000_CH3ONO.txt'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 40001: CH3COCH2(ONO2) + hv -> CH3COCH2(O.) + NO2
! Cross section: Barnes 93 -  Quantum yield: Muller 2001
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CH3COCH2(ONO2) + hv -> CH3COCH2(O.) + NO2'
  jglabel2(jgi) = "40001"
  infile='l40001_alpha_nitrooxyacetone.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 40002: CH3COCH(ONO2)CH3 + hv -> CH3COCH(O.)CH3 + NO2
! Cross section: Barnes 93 -  Quantum yield: Muller 2001
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'CCH3COCH(ONO2)CH3 + hv -> CH3COCH(O.)CH3 + NO2'
  jglabel2(jgi) = "40002"
  infile='l40002_3_nitrooxy_2_butanone.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 40003: C2H5(ONO) + hv -> C2H5(O.) + NO
! XS: Heicklen, 1987 -  QY: Calvert, oxidation of the oxygenates book
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'C2H5(ONO) + hv -> C2H5(O.) + NO'
  jglabel2(jgi) = "40003"
  infile='l40003_C2H5ONO.txt'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 40004: n-C3H7(ONO) + hv -> C3H7(O.) + NO
! XS: Heicklen, 1987 - QY: Calvert, oxidation of the oxygenates book
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'n-C3H7(ONO) + hv -> C3H7(O.) + NO'
  jglabel2(jgi) = "40004"
  infile='l40004_nC3H7ONO.txt'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 40005: n-C4H9(ONO) + hv -> C4H9(O.) + NO
! XS: Heicklen, 1987 - QY: Calvert, oxidation of the oxygenates book
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'n-C4H9(ONO) + hv -> C4H9(O.) + NO'
  jglabel2(jgi) = "40005"
  infile='l40005_nC4H9ONO.txt'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

!=======================================================================
! Label 40100: ROOH + hv -> RO.+ OH.
!  Cross section: IUPAC99 - Quantum yield: IUPAC99
!=======================================================================
  jgi = jgi+1
  jglabel1(jgi) = 'ROOH + hv -> RO.+ OH.'
  jglabel2(jgi) = "40100"
  infile='l40100_CH3OOH.prn'
  CALL get_sq(nz,nw,wl,infile,jglabel1(jgi),jgi,sqg)

  IF (jgi > SIZE(jglabel1)) STOP '1002'
  RETURN

END SUBROUTINE swgecko

!=======================================================================
!=======================================================================
!=======================================================================
SUBROUTINE get_sq(nz,nw,wl,infile,rxlabel,jgi,sqg)
  use tuv_params
  IMPLICIT NONE

  INTEGER,INTENT(IN)       :: nz
  INTEGER,INTENT(IN)       :: nw
  REAL,INTENT(IN)          :: wl(kw)
  CHARACTER*50, INTENT(IN) :: infile
  CHARACTER*50, INTENT(IN) :: rxlabel
  INTEGER, INTENT(IN)      :: jgi
  REAL, INTENT(INOUT)      :: sqg(kj,kz,kw)

  CHARACTER(LEN=13),PARAMETER    :: dirin='DATAJ1/GECKO/'
  CHARACTER(LEN=150)             :: line
  CHARACTER(LEN=:), allocatable  :: filename
  INTEGER :: ierr

  INTEGER,PARAMETER        :: kdata=580  ! max # of wavelength bins in input files
  REAL :: x1(kdata), y1(kdata)
  REAL :: x2(kdata), y2(kdata)
  INTEGER :: idat,i,indat,iw

  REAL :: xs(kw), qy(kw), xsqy

  xs(:)=0. ;  qy(:)=0.

! open the input file
  filename=TRIM(dirin)//infile
  OPEN(kin,FILE=filename, FORM='FORMATTED', STATUS='OLD', IOSTAT=ierr)
  IF (ierr/=0) THEN
    WRITE(6,*) '--error--, in get_sq to open file:',TRIM(filename)
    STOP "in get_sq "
  ENDIF

! read in the data in the input file
  idat=0
  rdloop: DO
    READ(kin,'(a)',IOSTAT=ierr) line
    IF (ierr /=0) THEN
      WRITE(6,*) '--error--, in get_sq while reading:',TRIM(filename)
      WRITE(6,*) 'keyword "END" might be missing after last data'
      STOP "in get_sq "
    ENDIF
    IF (line(1:1)=='!') CYCLE rdloop
    IF (line(1:3)=='END') EXIT rdloop

    idat = idat+1
    IF (idat > kdata) THEN
      WRITE(6,*) '--error--, in get_sq - check table size'
      WRITE(6,*) 'to read data in:',TRIM(filename)
      STOP "in get_sq "
    ENDIF

    READ(line,*,IOSTAT=ierr) x1(idat), y1(idat), y2(idat)
    IF (ierr/=0) THEN
      WRITE(6,*) '--error--, in get_sq while reading line:',TRIM(line)
      STOP "in get_sq "
    ENDIF
    x2(idat) = x1(idat)
  ENDDO rdloop
  CLOSE(kin)
  indat=idat ! save the number of data read in the input file

! add points to "start" and "finish" the cross section dataset
  CALL addpnt(x1,y1,kdata,idat,x1(1)*(1.-deltax), 0.)    ! add just before 1st wavelength
  CALL addpnt(x1,y1,kdata,idat,0., 0.)                   ! add (0, 0) as 1st point
  CALL addpnt(x1,y1,kdata,idat,x1(idat)*(1.+deltax), 0.) ! add just after last wavelength
  CALL addpnt(x1,y1,kdata,idat,1.e+38, 0.)               ! add (inf,0)

! adjust the (x1,y1) grid to the (wl,xs) grid
  CALL inter2(nw,wl,xs,idat,x1,y1,ierr)
  IF (ierr /= 0) THEN
    WRITE(*,*) ierr, rxlabel
    STOP
  ENDIF

! add points to "start" and "finish" the quantum yield dataset
  idat=indat
  CALL addpnt(x2,y2,kdata,idat,x2(1)*(1.-deltax),0.)    ! add just before 1st wavelength
  CALL addpnt(x2,y2,kdata,idat,0.,0.)                   ! add (0, 0) as 1st point
  CALL addpnt(x2,y2,kdata,idat,x2(idat)*(1.+deltax),0.) ! add just after last wavelength
  CALL addpnt(x2,y2,kdata,idat,1.e+38,0.)               ! add (inf,0)

! adjust the (x2,y2) grid to the (wl,qy) grid
  CALL inter2(nw,wl,qy,idat,x2,y2,ierr)
  IF (ierr /= 0) THEN
    WRITE(*,*) ierr, rxlabel
    STOP
  ENDIF

! combine xs*qy dans save to corresponding J index
  DO iw = 1, nw - 1
    DO i = 1, nz
      xsqy = xs(iw) * qy(iw)
      sqg(jgi,i,iw) = xsqy
    ENDDO
  ENDDO

END SUBROUTINE get_sq

!=======================================================================
