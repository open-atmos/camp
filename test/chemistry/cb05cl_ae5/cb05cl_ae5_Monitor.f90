! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
! 
! Utility Data Module File
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
! File                 : cb05cl_ae5_Monitor.f90
! Time                 : Thu Feb  8 11:36:55 2018
! Working directory    : /home/Earth/mdawson/Documents/partmc-chem/partmc/test/chemistry/cb05cl_ae5
! Equation file        : cb05cl_ae5.kpp
! Output root filename : cb05cl_ae5
! 
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



MODULE cb05cl_ae5_Monitor


  CHARACTER(LEN=15), PARAMETER, DIMENSION(75) :: SPC_NAMES = (/ &
     'BNZHRXN        ','BNZNRXN        ','BENZRO2        ', &
     'BENZENE        ','ISOPRXN        ','SESQRXN        ', &
     'SESQ           ','SULF           ','SULRXN         ', &
     'TOLHRXN        ','TOLNRXN        ','TOLRO2         ', &
     'TRPRXN         ','XYLHRXN        ','XYLNRXN        ', &
     'XYLRO2         ','DUMMY          ','CL2            ', &
     'SO2            ','O1D            ','HOCL           ', &
     'PAN            ','TOL            ','N2O5           ', &
     'XYL            ','CH4            ','HONO           ', &
     'H2O2           ','FACD           ','PACD           ', &
     'PANX           ','PNA            ','TO2            ', &
     'AACD           ','ETHA           ','MEOH           ', &
     'ETOH           ','HCL            ','HCO3           ', &
     'ROOH           ','MGLY           ','CLO            ', &
     'CRO            ','FMCL           ','MEPX           ', &
     'HNO3           ','CO             ','OPEN           ', &
     'ROR            ','CRES           ','ETH            ', &
     'TERP           ','IOLE           ','OLE            ', &
     'XO2N           ','PAR            ','ISOP           ', &
     'ISPD           ','ALDX           ','ALD2           ', &
     'FORM           ','NTR            ','O3             ', &
     'MEO2           ','XO2            ','C2O3           ', &
     'CL             ','NO             ','HO2            ', &
     'NO3            ','OH             ','CXO3           ', &
     'O              ','NO2            ','O2             ' /)

  INTEGER, PARAMETER, DIMENSION(75) :: LOOKAT = (/ &
       1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, &
      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, &
      25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, &
      37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, &
      49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, &
      61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, &
      73, 74, 75 /)

  INTEGER, PARAMETER, DIMENSION(3) :: MONITOR = (/ &
      63, 68, 74 /)

  CHARACTER(LEN=15), DIMENSION(1) :: SMASS
  CHARACTER(LEN=100), PARAMETER, DIMENSION(30) :: EQN_NAMES_0 = (/ &
     '          NO2 --> NO + O                                                                            ', &
     '            O --> O3                                                                                ', &
     '      O3 + NO --> NO2                                                                               ', &
     '      O + NO2 --> NO                                                                                ', &
     '      O + NO2 --> NO3                                                                               ', &
     '       NO + O --> NO2                                                                               ', &
     '     O3 + NO2 --> NO3                                                                               ', &
     '           O3 --> O                                                                                 ', &
     '           O3 --> O1D                                                                               ', &
     '          O1D --> O                                                                                 ', &
     '          O1D --> 2 OH                                                                              ', &
     '      O3 + OH --> HO2                                                                               ', &
     '     O3 + HO2 --> OH                                                                                ', &
     '          NO3 --> O + NO2                                                                           ', &
     '          NO3 --> NO                                                                                ', &
     '     NO + NO3 --> 2 NO2                                                                             ', &
     '    NO3 + NO2 --> NO + NO2                                                                          ', &
     '    NO3 + NO2 --> N2O5                                                                              ', &
     '         N2O5 --> DUMMY + 2 HNO3                                                                    ', &
     '         N2O5 --> 2 HNO3                                                                            ', &
     '         N2O5 --> DUMMY + NO3 + NO2                                                                 ', &
     '         2 NO --> 2 NO2                                                                             ', &
     '     NO + NO2 --> 2 HONO                                                                            ', &
     '      NO + OH --> HONO                                                                              ', &
     '         HONO --> NO + OH                                                                           ', &
     '    HONO + OH --> NO2                                                                               ', &
     '       2 HONO --> NO + NO2                                                                          ', &
     '     OH + NO2 --> HNO3                                                                              ', &
     '    HNO3 + OH --> NO3                                                                               ', &
     '     NO + HO2 --> OH + NO2                                                                          ' /)
  CHARACTER(LEN=100), PARAMETER, DIMENSION(30) :: EQN_NAMES_1 = (/ &
     '    HO2 + NO2 --> PNA                                                                               ', &
     '          PNA --> HO2 + NO2                                                                         ', &
     '     PNA + OH --> NO2                                                                               ', &
     '        2 HO2 --> DUMMY + H2O2                                                                      ', &
     '        2 HO2 --> H2O2                                                                              ', &
     '         H2O2 --> 2 OH                                                                              ', &
     '    H2O2 + OH --> HO2                                                                               ', &
     '          O1D --> HO2 + OH                                                                          ', &
     '           OH --> HO2                                                                               ', &
     '       OH + O --> HO2                                                                               ', &
     '         2 OH --> O                                                                                 ', &
     '         2 OH --> H2O2                                                                              ', &
     '     HO2 + OH --> DUMMY                                                                             ', &
     '      HO2 + O --> OH                                                                                ', &
     '     H2O2 + O --> HO2 + OH                                                                          ', &
     '      NO3 + O --> NO2                                                                               ', &
     '     NO3 + OH --> HO2 + NO2                                                                         ', &
     '    HO2 + NO3 --> HNO3                                                                              ', &
     '     O3 + NO3 --> NO2                                                                               ', &
     '        2 NO3 --> 2 NO2                                                                             ', &
     '          PNA --> 0.61 HO2 + 0.39 NO3 + 0.39 OH + 0.61 NO2                                          ', &
     '         HNO3 --> OH + NO2                                                                          ', &
     '         N2O5 --> NO3 + NO2                                                                         ', &
     '     XO2 + NO --> NO2                                                                               ', &
     '    XO2N + NO --> NTR                                                                               ', &
     '    XO2 + HO2 --> ROOH                                                                              ', &
     '   XO2N + HO2 --> ROOH                                                                              ', &
     '        2 XO2 --> DUMMY                                                                             ', &
     '       2 XO2N --> DUMMY                                                                             ', &
     '   XO2N + XO2 --> DUMMY                                                                             ' /)
  CHARACTER(LEN=100), PARAMETER, DIMENSION(30) :: EQN_NAMES_2 = (/ &
     '     NTR + OH --> HNO3 - -0.66 PAR + 0.33 ALDX + 0.33 ALD2 + 0.33 FORM ... etc.                     ', &
     '          NTR --> - -0.66 PAR + 0.33 ALDX + 0.33 ALD2 + 0.33 FORM + HO2 ... etc.                    ', &
     '    ROOH + OH --> 0.5 ALDX + 0.5 ALD2 + XO2                                                         ', &
     '         ROOH --> 0.5 ALDX + 0.5 ALD2 + HO2 + OH                                                    ', &
     '      CO + OH --> HO2                                                                               ', &
     '     CH4 + OH --> MEO2                                                                              ', &
     '    MEO2 + NO --> FORM + HO2 + NO2                                                                  ', &
     '   MEO2 + HO2 --> MEPX                                                                              ', &
     '       2 MEO2 --> 0.63 MEOH + 1.37 FORM + 0.74 HO2                                                  ', &
     '    MEPX + OH --> 0.7 MEO2 + 0.3 XO2 + 0.3 HO2                                                      ', &
     '         MEPX --> FORM + HO2 + OH                                                                   ', &
     '    MEOH + OH --> FORM + HO2                                                                        ', &
     '    FORM + OH --> CO + HO2                                                                          ', &
     '         FORM --> CO + 2 HO2                                                                        ', &
     '         FORM --> CO                                                                                ', &
     '     FORM + O --> CO + HO2 + OH                                                                     ', &
     '   FORM + NO3 --> HNO3 + CO + HO2                                                                   ', &
     '   FORM + HO2 --> HCO3                                                                              ', &
     '         HCO3 --> FORM + HO2                                                                        ', &
     '    HCO3 + NO --> FACD + HO2 + NO2                                                                  ', &
     '   HCO3 + HO2 --> MEPX                                                                              ', &
     '    FACD + OH --> HO2                                                                               ', &
     '     ALD2 + O --> C2O3 + OH                                                                         ', &
     '    ALD2 + OH --> C2O3                                                                              ', &
     '   ALD2 + NO3 --> HNO3 + C2O3                                                                       ', &
     '         ALD2 --> CO + MEO2 + HO2                                                                   ', &
     '    C2O3 + NO --> MEO2 + NO2                                                                        ', &
     '   C2O3 + NO2 --> PAN                                                                               ', &
     '          PAN --> DUMMY + C2O3 + NO2                                                                ', &
     '          PAN --> C2O3 + NO2                                                                        ' /)
  CHARACTER(LEN=100), PARAMETER, DIMENSION(30) :: EQN_NAMES_3 = (/ &
     '   C2O3 + HO2 --> 0.8 PACD + 0.2 AACD + 0.2 O3                                                      ', &
     '  MEO2 + C2O3 --> 0.1 AACD + FORM + 0.9 MEO2 + 0.9 HO2                                              ', &
     '   XO2 + C2O3 --> 0.1 AACD + 0.9 MEO2                                                               ', &
     '       2 C2O3 --> 2 MEO2                                                                            ', &
     '    PACD + OH --> C2O3                                                                              ', &
     '         PACD --> MEO2 + OH                                                                         ', &
     '    AACD + OH --> MEO2                                                                              ', &
     '     ALDX + O --> OH + CXO3                                                                         ', &
     '    ALDX + OH --> CXO3                                                                              ', &
     '   ALDX + NO3 --> HNO3 + CXO3                                                                       ', &
     '         ALDX --> CO + MEO2 + HO2                                                                   ', &
     '    NO + CXO3 --> ALD2 + XO2 + HO2 + NO2                                                            ', &
     '   CXO3 + NO2 --> PANX                                                                              ', &
     '         PANX --> DUMMY + CXO3 + NO2                                                                ', &
     '         PANX --> CXO3 + NO2                                                                        ', &
     '    PANX + OH --> ALD2 + NO2                                                                        ', &
     '   HO2 + CXO3 --> 0.8 PACD + 0.2 AACD + 0.2 O3                                                      ', &
     '  MEO2 + CXO3 --> 0.1 AACD + 0.9 ALD2 + 0.1 FORM + 0.9 XO2 + HO2                                    ', &
     '   XO2 + CXO3 --> 0.1 AACD + 0.9 ALD2                                                               ', &
     '       2 CXO3 --> 2 ALD2 + 2 XO2 + 2 HO2                                                            ', &
     '  C2O3 + CXO3 --> ALD2 + MEO2 + XO2 + HO2                                                           ', &
     '     PAR + OH --> 0.76 ROR + 0.13 XO2N - -0.11 PAR + 0.05 ALDX + 0.06 ALD2 ... etc.                 ', &
     '          ROR --> 0.02 ROR + 0.04 XO2N - -2.1 PAR + 0.5 ALDX + 0.6 ALD2 ... etc.                    ', &
     '          ROR --> HO2                                                                               ', &
     '    ROR + NO2 --> NTR                                                                               ', &
     '      OLE + O --> 0.2 CO + 0.01 XO2N + 0.2 PAR + 0.3 ALDX + 0.2 ALD2 + 0.2 FORM ... etc.            ', &
     '     OLE + OH --> - -0.7 PAR + 0.62 ALDX + 0.33 ALD2 + 0.8 FORM + 0.8 XO2 ... etc.                  ', &
     '     OLE + O3 --> 0.33 CO - PAR + 0.32 ALDX + 0.18 ALD2 + 0.74 FORM + 0.22 XO2 ... etc.             ', &
     '    OLE + NO3 --> 0.09 XO2N - PAR + 0.56 ALDX + 0.35 ALD2 + FORM + 0.91 XO2 ... etc.                ', &
     '      ETH + O --> CO + FORM + 0.7 XO2 + 1.7 HO2 + 0.3 OH                                            ' /)
  CHARACTER(LEN=100), PARAMETER, DIMENSION(30) :: EQN_NAMES_4 = (/ &
     '     ETH + OH --> 0.22 ALDX + 1.56 FORM + XO2 + HO2                                                 ', &
     '     ETH + O3 --> 0.37 FACD + 0.63 CO + FORM + 0.13 HO2 + 0.13 OH                                   ', &
     '    ETH + NO3 --> 2 FORM + XO2 + NO2                                                                ', &
     '     IOLE + O --> 0.1 CO + 0.1 PAR + 0.66 ALDX + 1.24 ALD2 + 0.1 XO2 + 0.1 HO2 ... etc.             ', &
     '    IOLE + OH --> 0.7 ALDX + 1.3 ALD2 + XO2 + HO2                                                   ', &
     '    IOLE + O3 --> 0.25 CO + 0.35 ALDX + 0.65 ALD2 + 0.25 FORM + 0.5 HO2 ... etc.                    ', &
     '   IOLE + NO3 --> 0.64 ALDX + 1.18 ALD2 + HO2 + NO2                                                 ', &
     '     TOL + OH --> 0.765 TOLRO2 + 0.56 TO2 + 0.36 CRES + 0.08 XO2 + 0.44 HO2 ... etc.                ', &
     '     TO2 + NO --> 0.9 OPEN + 0.1 NTR + 0.9 HO2 + 0.9 NO2                                            ', &
     '          TO2 --> CRES + HO2                                                                        ', &
     '    CRES + OH --> 0.4 CRO + 0.3 OPEN + 0.6 XO2 + 0.6 HO2                                            ', &
     '   CRES + NO3 --> CRO + HNO3                                                                        ', &
     '    CRO + NO2 --> NTR                                                                               ', &
     '    CRO + HO2 --> CRES                                                                              ', &
     '         OPEN --> CO + C2O3 + HO2                                                                   ', &
     '    OPEN + OH --> 2 CO + FORM + XO2 + C2O3 + 2 HO2                                                  ', &
     '    OPEN + O3 --> 0.2 MGLY + 0.69 CO + 0.03 ALDX + 0.7 FORM + 0.03 XO2 ... etc.                     ', &
     '     XYL + OH --> 0.804 XYLRO2 + 0.3 TO2 + 0.8 MGLY + 0.2 CRES + 1.1 PAR ... etc.                   ', &
     '    MGLY + OH --> XO2 + C2O3                                                                        ', &
     '         MGLY --> CO + C2O3 + HO2                                                                   ', &
     '     ISOP + O --> 0.25 PAR + 0.75 ISPD + 0.5 FORM + 0.25 XO2 + 0.25 HO2 ... etc.                    ', &
     '    ISOP + OH --> ISOPRXN + 0.088 XO2N + 0.912 ISPD + 0.629 FORM + 0.991 XO2 ... etc.               ', &
     '    ISOP + O3 --> 0.066 CO + 0.35 PAR + 0.65 ISPD + 0.15 ALDX + 0.6 FORM ... etc.                   ', &
     '   ISOP + NO3 --> 2.4 PAR + 0.2 ISPD + 0.8 ALDX + 0.8 NTR + XO2 + 0.8 HO2 ... etc.                  ', &
     '    ISPD + OH --> 0.168 MGLY + 0.334 CO + 1.565 PAR + 0.12 ALDX + 0.252 ALD2 ... etc.               ', &
     '    ISPD + O3 --> 0.85 MGLY + 0.225 CO + 0.36 PAR + 0.02 ALD2 + 0.15 FORM ... etc.                  ', &
     '   ISPD + NO3 --> 0.15 HNO3 + 0.643 CO + 1.282 PAR + 0.357 ALDX + 0.282 FORM ... etc.               ', &
     '         ISPD --> 0.333 CO + 0.832 PAR + 0.067 ALD2 + 0.9 FORM + 0.7 XO2 ... etc.                   ', &
     '     TERP + O --> TRPRXN + 5.12 PAR + 0.15 ALDX                                                     ', &
     '    TERP + OH --> TRPRXN + 0.25 XO2N + 1.66 PAR + 0.47 ALDX + 0.28 FORM ... etc.                    ' /)
  CHARACTER(LEN=100), PARAMETER, DIMENSION(30) :: EQN_NAMES_5 = (/ &
     '    TERP + O3 --> TRPRXN + 0.001 CO + 0.18 XO2N + 7 PAR + 0.21 ALDX + 0.24 FORM ... etc.            ', &
     '   TERP + NO3 --> TRPRXN + 0.25 XO2N + 0.47 ALDX + 0.53 NTR + 1.03 XO2 ... etc.                     ', &
     '     SO2 + OH --> SULF + SULRXN + HO2                                                               ', &
     '    ETOH + OH --> 0.05 ALDX + 0.9 ALD2 + 0.1 FORM + 0.1 XO2 + HO2                                   ', &
     '    ETHA + OH --> 0.009 XO2N + 0.991 ALD2 + 0.991 XO2 + HO2                                         ', &
     '   ISOP + NO2 --> 2.4 PAR + 0.2 ISPD + 0.8 ALDX + 0.8 NTR + XO2 + 0.2 NO ... etc.                   ', &
     '          CL2 --> 2 CL                                                                              ', &
     '         HOCL --> CL + OH                                                                           ', &
     '      O3 + CL --> CLO                                                                               ', &
     '        2 CLO --> 0.3 CL2 + 1.4 CL                                                                  ', &
     '     CLO + NO --> CL + NO2                                                                          ', &
     '    CLO + HO2 --> HOCL                                                                              ', &
     '    FMCL + OH --> CO + CL                                                                           ', &
     '         FMCL --> CO + CL + HO2                                                                     ', &
     '     CH4 + CL --> HCL + MEO2                                                                        ', &
     '     PAR + CL --> HCL + 0.76 ROR + 0.13 XO2N - -0.11 PAR + 0.05 ALDX + 0.06 ALD2 ... etc.           ', &
     '    ETHA + CL --> HCL + 0.009 XO2N + 0.991 ALD2 + 0.991 XO2 + HO2                                   ', &
     '     ETH + CL --> FMCL + FORM + 2 XO2 + HO2                                                         ', &
     '     OLE + CL --> FMCL - PAR + 0.67 ALDX + 0.33 ALD2 + 2 XO2 + HO2                                  ', &
     '    IOLE + CL --> 0.3 HCL + 0.7 FMCL + 0.3 OLE + 0.3 PAR + 0.55 ALDX + 0.45 ALD2 ... etc.           ', &
     '    ISOP + CL --> 0.15 HCL + 0.85 FMCL + ISPD + XO2 + HO2                                           ', &
     '    FORM + CL --> HCL + CO + HO2                                                                    ', &
     '    ALD2 + CL --> HCL + C2O3                                                                        ', &
     '    ALDX + CL --> HCL + CXO3                                                                        ', &
     '    MEOH + CL --> HCL + FORM + HO2                                                                  ', &
     '    ETOH + CL --> HCL + ALD2 + HO2                                                                  ', &
     '     HCL + OH --> CL                                                                                ', &
     '  TOLRO2 + NO --> TOLNRXN + NO                                                                      ', &
     ' TOLRO2 + HO2 --> TOLHRXN + HO2                                                                     ', &
     '  XYLRO2 + NO --> XYLNRXN + NO                                                                      ' /)
  CHARACTER(LEN=100), PARAMETER, DIMENSION(8) :: EQN_NAMES_6 = (/ &
     ' XYLRO2 + HO2 --> XYLHRXN + HO2                                                                     ', &
     ' BENZENE + OH --> 0.764 BENZRO2 + OH                                                                ', &
     ' BENZRO2 + NO --> BNZNRXN + NO                                                                      ', &
     'BENZRO2 + HO2 --> BNZHRXN + HO2                                                                     ', &
     '    SESQ + O3 --> SESQRXN + O3                                                                      ', &
     '    SESQ + OH --> SESQRXN + OH                                                                      ', &
     '   SESQ + NO3 --> SESQRXN + NO3                                                                     ', &
     '           O2 --> 2 O                                                                               ' /)
  CHARACTER(LEN=100), PARAMETER, DIMENSION(188) :: EQN_NAMES = (/&
    EQN_NAMES_0, EQN_NAMES_1, EQN_NAMES_2, EQN_NAMES_3, EQN_NAMES_4, &
    EQN_NAMES_5, EQN_NAMES_6 /)

! INLINED global variables

! End INLINED global variables


END MODULE cb05cl_ae5_Monitor
