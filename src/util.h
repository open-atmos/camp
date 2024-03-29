/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 *
 * Utility functions and commonly used science property calculators
 *
 */
/** \file
 * \brief Utility functions and commonly used science property calculators
 */
#ifndef UTIL_H
#define UTIL_H

// Universal gas constant (J mol-1 K-1)
#define UNIV_GAS_CONST_ 8.314472

/** Calculate the mean speed of a gas-phase species
 * [ \f$\mbox{m}\,\mbox{s}^{-1}\f$ ] :
 * \f[
 *   v = \sqrt{\frac{8RT}{\pi MW}}
 * \f]
 * where R is the ideal gas constant [\f$\mbox{J}\, \mbox{K}^{-1}\,
 * \mbox{mol}^{-1}\f$], T is temperature [K], and MW is the molecular weight of
 * the gas-phase species
 * [\f$\mbox{kg}\, \mbox{mol}^{-1}\f$]
 * @param temperature__K Temperature [K]
 * @param mw__kg_mol Molecular weight of the gas-phase species [\f$\mbox{kg}\,
 * \mbox{mol}^{-1}\f$]
 */
static inline double mean_speed__m_s(double temperature__K,
                                     double mw__kg_mol) {
  return sqrt(8.0 * UNIV_GAS_CONST_ * temperature__K / (M_PI * mw__kg_mol));
}

/** Calculate the mean free path of a gas-phase species [m]
 * \f[
 *   \lambda = 3.0 D_g / v
 * \f]
 * where \f$D_g\f$ is the gas-phase diffusion coefficient
 * [\f$\mbox{m}^2\,\mbox{s}^{-1}\f$] and
 * \f$v\f$ is the mean speed of the gas-phase molecules
 * [ \f$\mbox{m}\,\mbox{s}^{-1}\f$ ].
 *
 * @param diffusion_coeff__m2_s Diffusion coefficient of the gas species
 * [\f$\mbox{m}^2\, \mbox{s}^{-1}\f$]
 * @param temperature__K Temperature [K]
 * @param mw__kg_mol Molecular weight of the gas-phase species [\f$\mbox{kg}\,
 * \mbox{mol}^{-1}\f$]
 */
static inline double mean_free_path__m(double diffusion_coeff__m2_s,
                                       double temperature__K,
                                       double mw__kg_mol) {
  return 3.0 * diffusion_coeff__m2_s /
         mean_speed__m_s(temperature__K, mw__kg_mol);
}

/** Calculate the transition regime correction factor [unitless] \cite Fuchs1971
 * : \f[ f(K_n,\alpha) = \frac{0.75 \alpha ( 1 + K_n )}{K_n(1+K_n) + 0.283\alpha
 * K_n + 0.75 \alpha} \f] where the Knudsen Number \f$K_n = \lambda / r\f$
 * (where \f$\lambda\f$ is the mean free path [m] of the gas-phase species and
 * \f$r\f$ is the effective radius of the particles [m]), and \f$ \alpha \f$ is
 * the mass accomodation coefficient, which is typically assumed to equal 0.1
 * \cite Zaveri2008.
 *
 *  @param mean_free_path__m mean free path of the gas-phase species [m]
 *  @param radius__m Particle effective radius [m]
 *  @param alpha Mass accomodation coefficient [unitless]
 */
static inline double transition_regime_correction_factor(
    double mean_free_path__m, double radius__m, double alpha) {
  double K_n = mean_free_path__m / radius__m;
  return (0.75 * alpha * (1.0 + K_n)) /
         (K_n * K_n + (1.0 + 0.283 * alpha) * K_n + 0.75 * alpha);
}

/** Calculate the derivative of the transition regime correction factor by
 *  particle radius.
 *  \f[
 *    \frac{d f_{fs}}{d r} = \frac{0.75 \alpha \lambda (K_n^2 + 2K_n + 1 +
 * (0.283-0.75)\alpha)}{r^2(K_n^2 + (1+0.283\alpha)K_n + 0.75\alpha)^2} \f]
 *  \todo double check the correction factor derivative equation
 *  where the Knudsen Number \f$K_n = \lambda / r\f$ (where \f$\lambda\f$ is the
 *  mean free path [m] of the gas-phase species and \f$r\f$ is the effective
 * radius of the particles [m]), and \f$ \alpha \f$ is the mass accomodation
 * coefficient, which is typically assumed to equal 0.1 \cite Zaveri2008.
 *
 *  @param mean_free_path__m mean free path of the gas-phase species [m]
 *  @param radius__m Particle effective radius [m]
 *  @param alpha Mass accomodation coefficient [unitless]
 */
static inline double d_transition_regime_correction_factor_d_radius(
    double mean_free_path__m, double radius__m, double alpha) {
  double K_n = mean_free_path__m / radius__m;
  return (0.75 * alpha * mean_free_path__m *
          (K_n * K_n + 2.0 * K_n + 1.0 + (0.283 - 0.75) * alpha)) /
         pow(radius__m *
                 (K_n * K_n + (1.0 + 0.283 * alpha) * K_n + 0.75 * alpha),
             2);
}

/** Calculate the gas-aerosol reaction rate constant for the transition regime
 * [\f$\mbox{m}^3\, \mbox{particle}^{-1}\, \mbox{s}^{-1}\f$]
 *
 *  The rate constant \f$k_c\f$ is calculated according to \cite Zaveri2008 as:
 *  \f[
 *    k_c = 4 \pi r D_g f_{fs}( K_n, \alpha )
 *  \f]
 *  where \f$r\f$ is the radius of the particle(s) [m], \f$D_g\f$ is the
 * diffusion coefficient of the gas-phase species
 * [\f$\mbox{m}^2\,\mbox{s}^{-1}\f$], \f$f_{fs}( K_n, \alpha )\f$ is the Fuchs
 * Sutugin transition regime correction factor [unitless], \f$K_n\f$ is the
 * Knudsen Number [unitess], and \f$\alpha\f$ is the mass accomodation
 * coefficient.
 *
 *  Rates can be calculated as:
 *  \f[
 *    r_c = [G] N_a k_c
 *  \f]
 *  where \f$[G]\f$ is the gas-phase species concentration [ppm], \f$N_a\f$ is
 * the number concentration of particles [\f$\mbox{particle}\,\mbox{m}^{-3}\f$]
 * and the rate \f$r_c\f$ is in [\f$\mbox{ppm}\,\mbox{s}^{-1}\f$].
 *
 *  @param diffusion_coeff__m2_s Diffusion coefficent of the gas species
 *  [\f$\mbox{m}^2\, \mbox{s}^{-1}\f$]
 *  @param mean_free_path__m Mean free path of gas molecules [m]
 *  @param radius__m Particle radius [m]
 *  @param alpha Mass accomodation coefficient [unitless]
 */
static inline double gas_aerosol_transition_rxn_rate_constant(
    double diffusion_coeff__m2_s, double mean_free_path__m,
    double radius__m, double alpha) {
  return 4.0 * M_PI * radius__m * diffusion_coeff__m2_s *
         transition_regime_correction_factor(mean_free_path__m, radius__m,
                                             alpha);
}

/** Calculate the derivative of a transition-regime gas-aerosol reaction
 * rate by particle radius
 * \f[
 *   \frac{d k_c}{d r} = 4 \pi D_g ( f_{fs} + r \frac{d_{fs}}{d r} )
 * \f]
 *  where \f$r\f$ is the radius of the particle(s) [m], \f$D_g\f$ is the
 * diffusion coefficient of the gas-phase species [\f$\mbox{m}^2\,
 * \mbox{s}^{-1}\f$] and \f$f_{fs}( K_n, \alpha )\f$ is the Fuchs Sutugin
 * transition regime correction factor [unitless] (\f$K_n\f$ is the Knudsen
 * Number [unitess] and \f$\alpha\f$ is the mass accomodation coefficient.
 *
 *  @param diffusion_coeff__m2_s Diffusion coefficent of the gas species
 *  [\f$\mbox{m}^2\, \mbox{s}^{-1}\f$]
 *  @param mean_free_path__m Mean free path of gas molecules [m]
 *  @param radius__m Particle radius [m]
 *  @param alpha Mass accomodation coefficient [unitless]
 */
static inline double d_gas_aerosol_transition_rxn_rate_constant_d_radius(
    double diffusion_coeff__m2_s, double mean_free_path__m, double radius__m,
    double alpha) {
  return 4.0 * M_PI * diffusion_coeff__m2_s *
         (transition_regime_correction_factor(mean_free_path__m, radius__m,
                                              alpha) +
          radius__m * d_transition_regime_correction_factor_d_radius(
                          mean_free_path__m, radius__m, alpha));
}

/** Calculate the gas-aerosol reaction rate for the continuum regime \cite Tie2003
 * [\f$\mbox{m}^3\, \mbox{particle}^{-1}\, \mbox{s}^{-1}\f$]
 *
 * The rate constant \f$k_c\f$ is calculated as:
 * \f[
 *   k_c = \frac{4 \pi r^2_e}{\left(\frac{r}{D_g} + \frac{4}{v(T)\gamma}\right)}
 * \f]
 *
 * where \f$r\f$ is the particle radius [m],
 * \f$D_g\f$ is the gas-phase diffusion coefficient of the reactant
 * [\f$\mbox{m}^2\mbox{s}^{-1}\f$], \f$\gamma\f$ is the reaction probability [unitless],
 * and v is the mean free speed of the gas-phase reactant.
 *
 * @param diffusion_coeff__m2_s Diffusion coefficient of the gas species
 *  [\f$\mbox{m}^2\, \mbox{s}^{-1}\f$]
 *  @param mean_speed__m_s Mean speed of the gas molecule [m s-1]
 *  @param radius__m Particle radius [m]
 *  @param alpha Mass accomodation coefficient [unitless]
 */
static inline double gas_aerosol_continuum_rxn_rate_constant(
    double diffusion_coeff__m2_s, double mean_speed__m_s,
    double radius__m, double alpha) {
  return 4.0 * M_PI * radius__m * radius__m /
         ( radius__m / diffusion_coeff__m2_s +
           4.0 / ( mean_speed__m_s * alpha ) );
}

/** Calculate the derivative of the continuum-regime gas-aerosol reaction rate
 * constant by particle radius \cite Tie2003
 * \f[
 *   \frac{dk_c}{dr} = 4 \pi \frac{\frac{r^2}{D_g} +
 *      \frac{8r}{v(T) \gamma}}{\left(\frac{r}{D_g} + \frac{4}{v(T) \gamma}\right)^2}
 * \f]
 *
 * where \f$r\f$ is the particle radius [m],
 * \f$D_g\f$ is the gas-phase diffusion coefficient of the reactant
 * [\f$\mbox{m}^2\mbox{s}^{-1}\f$], \f$\gamma\f$ is the reaction probability [unitless],
 * and v is the mean free speed of the gas-phase reactant.
 *
 * @param diffusion_coeff__m2_s Diffusion coefficient of the gas species
 *  [\f$\mbox{m}^2\, \mbox{s}^{-1}\f$]
 *  @param mean_speed__m_s Mean speed of the gas molecule [m s-1]
 *  @param radius__m Particle radius [m]
 *  @param alpha Mass accomodation coefficient [unitless]
 */
static inline double d_gas_aerosol_continuum_rxn_rate_constant_d_radius(
    double diffusion_coeff__m2_s, double mean_speed__m_s,
    double radius__m, double alpha) {
  double nom = radius__m * radius__m / diffusion_coeff__m2_s +
               8.0 * radius__m / ( mean_speed__m_s * alpha );
  double denom = radius__m / diffusion_coeff__m2_s +
                 4.0 / ( mean_speed__m_s * alpha );
  return 4.0 * M_PI * nom / ( denom * denom );
}
#endif  // UTIL_H
