      SUBROUTINE waters(sza, kabs, y,
     $     se_0,de_0,se_y, de_y, se_i, de_i)

* compute air-water interface and propagation into waters
* separately for direct solar and diffuse.
** (many parts of this calculation can be moved to outer loops
** because they don't depend on, e.g., wavelength).  However, 
** should allow for possible wavelength-dependence of index of
** refraction.

      IMPLICIT NONE
      REAL pi
      PARAMETER(pi=3.1415926535898)

* INPUTS:
*   y = depth, meters
*   kabs = monochromatic absorption coefficient (ln, 1/m)
*   sza_a = solar zenith angle in air

      REAL sza, kabs, y

* INTERNAL:
*   sza_a = solar zenith angle of direct beam, in air (limit < 90)
*   sza_w = solar zenith angle of direct beam, in water
*   kang = max.   number of quadrature angles for diffuse light
*   nang = actual number of quadrature angles for diffuse light
*   dza_ai = zenith angle of each angular sector, in air
*   dza_wi = zenith angle of each angular sector, in water

      REAL sza_a, sza_w
      INTEGER kang, nang, i
      PARAMETER(kang=200)
      REAL dza_ai(kang), dza_wi(kang)
      
* radiation just above (air) and just below (water) interface
*     se_a = solar beam irradiance (energy) in air 
*     se_w = solar beam irradiance in water, y = 0
*     df_a = diffuse actinic flux (scalar irradiance) in air
*     de_wi(kang) = diffuse irradiance in water, y = 0

      REAL se_a, se_w
      REAL df_a, de_wi(kang)

*     de_yi = diffuse irradiance at depth = y, for each sector
*     de_ii = integrated irradiance, from 0 to y.

      REAL de_yi, de_ii

* secants of propagation angle (sun and diffuse)

      REAL ssec, dseci

* OUTPUT:
*     se_0 = solar beam irradiance at depth y
*     de_0 = diffuse irradiance at depth y (sum over all angles)
*     se_y = solar beam irradiance at depth y
*     de_y = diffuse irradiance at depth y (sum over all angles)
*     se_i = integral of solar irradiance from surface to depth y
*     de_i = integral of diffuse irradiacne from surface to depth y
*       
      real se_0, de_0      
      REAL se_y, se_i, de_y, de_i

* calculate with normalized inputs, i.e. edir = 1, edn = 1
* note conversion * 2 for edn to actinic flux, quadrature is based on
* equal areas (so equal actinic flux sectors).
C      se_a = edir
C      df_a = 2. * edn

      se_a = 1.
      df_a = 2.


* limit sza at interface to 89.9 degrees

      sza_a = min(sza,89.9)

* air-waters interface
* specify number of angles + 1 for diffuse light:

      nang = 11
      CALL intrfc(sza_a, se_a, sza_w, se_w,
     $     nang, dza_ai, df_a, dza_wi, de_wi)

* compute direct beam in water at depth y
* also compute integral from surface to depth y:

      se_0 = se_w


      ssec = 1./cos(sza_w * pi/180.)
      se_y = se_w * exp(-kabs * y * ssec)
      se_i = (se_w - se_y)/(ssec * kabs)

* compute diffuse radiation in water at depth y
* also compute integral of diffuse radiation from surface to detph y

      de_0 = 0.
      de_y = 0.
      de_i = 0.
      DO i = 1, nang-1
         de_0 = de_0 + de_wi(i) 
         dseci = 1./cos(dza_wi(i) * pi/180.)
         de_yi = de_wi(i) * exp(-kabs * y * dseci)
         de_ii = (de_wi(i) - de_yi) / (dseci * kabs)
         de_y = de_y + de_yi
         de_i = de_i + de_ii
      ENDDO

      RETURN
      END

********************************************************************

      SUBROUTINE intrfc(sza_a, se_a, sza_w, se_w,
     $     nang, dza_ai, df_a, dza_wi, de_wi)

* Inputs:
* direct solar beam (prefix - s)
*     sza_a = solar zenith angle (sza), in atmosphere (a)
*     se_a = solar beam irradiance (se) in atmosphere (a)
* diffuse light (prefix - d)
*     df_a = diffuse actinic flux (df), in atmosphere (a)

* Outputs:
* direct beam
*     sza_w = solar zenith angle in water, direct beam
*     se_w = direct irradiance in water
* diffuse light:
* nang = number of angular directions, cosine-quadrature in atmosphere
*     dza_ai(i) = zenith angle of i-th diffuse beam, in atmosphere
*     df_ai = df_a/(nang-1) = diffuse actinic flux in each angular sector, atmos.
*     dza_wi(i) = zenith angle of i-th diffuse beam in water
*     de_wi(i) = irradiance of each diffuse beam in water

      IMPLICIT none
      REAL pi
      PARAMETER(pi=3.1415926535898)
      REAL sza_a, sza_w
      REAL se_a, se_w
      REAL r

      INTEGER nang, i
      REAL df_a
      REAL dza_ai(nang), df_ai, de_ai
      REAL dza_wi(nang), de_wi(nang)

      real anga, angw

* data for 10-sector quadrature:

      real quadr(10)
      data quadr/87.13402,81.37307,75.52248,69.51268,63.25631,
     $     56.63298,49.45840,41.40962,31.78833,18.19578    /

* direct beam:
* Input:  sza, se_a = fdir(1); Output se_w, r
* (this depends only on sza and can be moved to outer loops)

      CALL snefre(sza_a, sza_w, r)
      se_w = (1. - r) * se_a

* diffuse light:
* divide incident diffuse (assume isotropic) into nang-1 equal actinic fluxes:
* compute corresponging central nang-1 angles, sza_ad

      df_ai = df_a/FLOAT(nang-1)

* replace by quadr data for nang=11

      IF(nang .EQ. 11) THEN
         DO i = 1, nang - 1
            dza_ai(i) = quadr(i)
         ENDDO
      ELSE
         CALL angdif(nang, dza_ai)
      ENDIF

* pass each diffuse beam through inteface:

      DO i = 1, nang - 1
         anga = dza_ai(i) 
         de_ai = df_ai * cos(anga*pi/180.)
         CALL snefre(anga, angw, r)
         de_wi(i) = (1.-r) * de_ai
         dza_wi(i) = angw
      ENDDO

      RETURN
      END

***************************************************************

      SUBROUTINE snefre(angl1, angl2, r)

* computes Snell's refraction and Fresnel reflection

      IMPLICIT none
      REAL pi
      PARAMETER(pi=3.1415926535898)
      REAL angl1, angl2
      REAL rad1, rad2
      REAL x1, x2, x3, x4
      REAL r

      REAL n1, n2

* should make n2 dependent on wavelength

      n1 = 1.0
      n2 = 1.33

* protect from zero:

      angl1 = max(angl1, 0.001)

* Snell's refraction angles:

      rad1 = angl1 * pi/180.
      rad2 = asin((n1/n2)*sin(rad1))
      angl2 = rad2 * 180./pi

* Fresnel's reflection for random polarization

      x1 = (sin(rad1-rad2))**2
      x2 = (sin(rad1+rad2))**2
      x3 = (tan(rad1-rad2))**2
      x4 = (tan(rad1+rad2))**2

      R = x1/(2.*x2)  +  x3/(2.*x4)

      RETURN
      END

****************************************************************

      SUBROUTINE angdif(nang, ang_c)

* Angular quadrature:
* Discretize angles according to equally spaced cosines.  This quadrature is most 
* appropriate for actinic flux.  Quadrature for irradiance could go as
* cos^2.  But actinic flux may be better given  angles change in 
* water anyway, becoming closer to normal.
* Input = nang = number of desired delta-angle sectors + 1
* Output = central angle of each sector
* (of course this can all be replaced by a data table, for chosen nang)


      IMPLICIT NONE
      REAL pi
      PARAMETER(pi=3.1415926535898)
      INTEGER nang, iang
      REAL dcos
      REAL ang_c(nang)
      REAL cos_L, cos_U, cos_C

* nang = number of increments + 1
* dcos = increment in cos space:

      dcos = 1./FLOAT(nang-1)
      DO iang = 1, nang-1
         
* lower
         cos_L = 0. + dcos * FLOAT(iang-1)
         cos_L = MIN(cos_L, 0.99999)

* upper
         cos_U = 0. + dcos * FLOAT(iang)
         cos_U = MIN(cos_U, 0.99999)

* center
         cos_C = (cos_L + cos_U)/2. 
         ang_C(iang) = ACOS(cos_C) * 180./pi

      ENDDO

      RETURN
      END

