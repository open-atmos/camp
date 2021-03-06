! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
! 
! Parameter Module File
! 
! Generated by KPP-2.2.3 symbolic chemistry Kinetics PreProcessor
!       (http://www.cs.vt.edu/~asandu/Software/KPP)
! KPP is distributed under GPL, the general public licence
!       (http://www.gnu.org/copyleft/gpl.html)
! (C) 1995-1997, V. Damian & A. Sandu, CGRER, Univ. Iowa
! (C) 1997-2005, A. Sandu, Michigan Tech, Virginia Tech
!     With important contributions from:
!        M. Damian, Villanova University, USA
!        R. Sander, Max-Planck Institute for Chemistry, Mainz, Germany
! 
! File                 : cb05cl_ae5_Parameters.f90
! Time                 : Thu Feb  8 11:36:55 2018
! Working directory    : /home/Earth/mdawson/Documents/partmc-chem/partmc/test/chemistry/cb05cl_ae5
! Equation file        : cb05cl_ae5.kpp
! Output root filename : cb05cl_ae5
! 
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



MODULE cb05cl_ae5_Parameters

  USE cb05cl_ae5_Precision
  PUBLIC
  SAVE


! NSPEC - Number of chemical species
  INTEGER, PARAMETER :: NSPEC = 75 
! NVAR - Number of Variable species
  INTEGER, PARAMETER :: NVAR = 74 
! NVARACT - Number of Active species
  INTEGER, PARAMETER :: NVARACT = 62 
! NFIX - Number of Fixed species
  INTEGER, PARAMETER :: NFIX = 1 
! NREACT - Number of reactions
  INTEGER, PARAMETER :: NREACT = 188 
! NVARST - Starting of variables in conc. vect.
  INTEGER, PARAMETER :: NVARST = 1 
! NFIXST - Starting of fixed in conc. vect.
  INTEGER, PARAMETER :: NFIXST = 75 
! NONZERO - Number of nonzero entries in Jacobian
  INTEGER, PARAMETER :: NONZERO = 670 
! LU_NONZERO - Number of nonzero entries in LU factoriz. of Jacobian
  INTEGER, PARAMETER :: LU_NONZERO = 748 
! CNVAR - (NVAR+1) Number of elements in compressed row format
  INTEGER, PARAMETER :: CNVAR = 75 
! NLOOKAT - Number of species to look at
  INTEGER, PARAMETER :: NLOOKAT = 75 
! NMONITOR - Number of species to monitor
  INTEGER, PARAMETER :: NMONITOR = 3 
! NMASS - Number of atoms to check mass balance
  INTEGER, PARAMETER :: NMASS = 1 

! Index declaration for variable species in C and VAR
!   VAR(ind_spc) = C(ind_spc)

  INTEGER, PARAMETER :: ind_BNZHRXN = 1 
  INTEGER, PARAMETER :: ind_BNZNRXN = 2 
  INTEGER, PARAMETER :: ind_BENZRO2 = 3 
  INTEGER, PARAMETER :: ind_BENZENE = 4 
  INTEGER, PARAMETER :: ind_ISOPRXN = 5 
  INTEGER, PARAMETER :: ind_SESQRXN = 6 
  INTEGER, PARAMETER :: ind_SESQ = 7 
  INTEGER, PARAMETER :: ind_SULF = 8 
  INTEGER, PARAMETER :: ind_SULRXN = 9 
  INTEGER, PARAMETER :: ind_TOLHRXN = 10 
  INTEGER, PARAMETER :: ind_TOLNRXN = 11 
  INTEGER, PARAMETER :: ind_TOLRO2 = 12 
  INTEGER, PARAMETER :: ind_TRPRXN = 13 
  INTEGER, PARAMETER :: ind_XYLHRXN = 14 
  INTEGER, PARAMETER :: ind_XYLNRXN = 15 
  INTEGER, PARAMETER :: ind_XYLRO2 = 16 
  INTEGER, PARAMETER :: ind_DUMMY = 17 
  INTEGER, PARAMETER :: ind_CL2 = 18 
  INTEGER, PARAMETER :: ind_SO2 = 19 
  INTEGER, PARAMETER :: ind_O1D = 20 
  INTEGER, PARAMETER :: ind_HOCL = 21 
  INTEGER, PARAMETER :: ind_PAN = 22 
  INTEGER, PARAMETER :: ind_TOL = 23 
  INTEGER, PARAMETER :: ind_N2O5 = 24 
  INTEGER, PARAMETER :: ind_XYL = 25 
  INTEGER, PARAMETER :: ind_CH4 = 26 
  INTEGER, PARAMETER :: ind_HONO = 27 
  INTEGER, PARAMETER :: ind_H2O2 = 28 
  INTEGER, PARAMETER :: ind_FACD = 29 
  INTEGER, PARAMETER :: ind_PACD = 30 
  INTEGER, PARAMETER :: ind_PANX = 31 
  INTEGER, PARAMETER :: ind_PNA = 32 
  INTEGER, PARAMETER :: ind_TO2 = 33 
  INTEGER, PARAMETER :: ind_AACD = 34 
  INTEGER, PARAMETER :: ind_ETHA = 35 
  INTEGER, PARAMETER :: ind_MEOH = 36 
  INTEGER, PARAMETER :: ind_ETOH = 37 
  INTEGER, PARAMETER :: ind_HCL = 38 
  INTEGER, PARAMETER :: ind_HCO3 = 39 
  INTEGER, PARAMETER :: ind_ROOH = 40 
  INTEGER, PARAMETER :: ind_MGLY = 41 
  INTEGER, PARAMETER :: ind_CLO = 42 
  INTEGER, PARAMETER :: ind_CRO = 43 
  INTEGER, PARAMETER :: ind_FMCL = 44 
  INTEGER, PARAMETER :: ind_MEPX = 45 
  INTEGER, PARAMETER :: ind_HNO3 = 46 
  INTEGER, PARAMETER :: ind_CO = 47 
  INTEGER, PARAMETER :: ind_OPEN = 48 
  INTEGER, PARAMETER :: ind_ROR = 49 
  INTEGER, PARAMETER :: ind_CRES = 50 
  INTEGER, PARAMETER :: ind_ETH = 51 
  INTEGER, PARAMETER :: ind_TERP = 52 
  INTEGER, PARAMETER :: ind_IOLE = 53 
  INTEGER, PARAMETER :: ind_OLE = 54 
  INTEGER, PARAMETER :: ind_XO2N = 55 
  INTEGER, PARAMETER :: ind_PAR = 56 
  INTEGER, PARAMETER :: ind_ISOP = 57 
  INTEGER, PARAMETER :: ind_ISPD = 58 
  INTEGER, PARAMETER :: ind_ALDX = 59 
  INTEGER, PARAMETER :: ind_ALD2 = 60 
  INTEGER, PARAMETER :: ind_FORM = 61 
  INTEGER, PARAMETER :: ind_NTR = 62 
  INTEGER, PARAMETER :: ind_O3 = 63 
  INTEGER, PARAMETER :: ind_MEO2 = 64 
  INTEGER, PARAMETER :: ind_XO2 = 65 
  INTEGER, PARAMETER :: ind_C2O3 = 66 
  INTEGER, PARAMETER :: ind_CL = 67 
  INTEGER, PARAMETER :: ind_NO = 68 
  INTEGER, PARAMETER :: ind_HO2 = 69 
  INTEGER, PARAMETER :: ind_NO3 = 70 
  INTEGER, PARAMETER :: ind_OH = 71 
  INTEGER, PARAMETER :: ind_CXO3 = 72 
  INTEGER, PARAMETER :: ind_O = 73 
  INTEGER, PARAMETER :: ind_NO2 = 74 

! Index declaration for fixed species in C
!   C(ind_spc)

  INTEGER, PARAMETER :: ind_O2 = 75 

! Index declaration for fixed species in FIX
!    FIX(indf_spc) = C(ind_spc) = C(NVAR+indf_spc)

  INTEGER, PARAMETER :: indf_O2 = 1 

END MODULE cb05cl_ae5_Parameters

