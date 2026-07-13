/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

/** \file
 * \brief Aerosol representation utility fuctions
 */
#ifndef AERO_SOLVER_DEV_H_
#define AERO_SOLVER_DEV_H_

#include "common_dev.h"
#include "cuda_structs.h"

#define CHEM_SPEC_UNKNOWN_TYPE 0
#define CHEM_SPEC_VARIABLE 1
#define CHEM_SPEC_CONSTANT 2
#define CHEM_SPEC_PSSA 3
#define CHEM_SPEC_ACTIVITY_COEFF 4

/** \brief Get the effective particle radius \f$r_{eff}\f$ (m)
 *
 * The modal mass effective radius is calculated for a log-normal distribution
 * where the geometric mean diameter (\f$\tilde{D}_n\f$) and geometric standard
 * deviation (\f$\sigma_g\f$) are set by the aerosol model prior to
 * solving the chemistry. Thus, all \f$\frac{\partial r_{eff}}{\partial y}\f$
 * are zero. The effective radius is calculated according to the equation given
 * in Table 1 of Zender \cite Zender2002 :
 *
 * \f[
 * \tilde{\sigma_g} \equiv ln( \sigma_g )
 * \f]
 * \f[
 * D_s = D_{eff} = \tilde{D_n} e^{5 \tilde{\sigma}_g^2 / 2}
 * \f]
 * \f[
 * r_{eff} = \frac{D_{eff}}{2}
 * \f]
 *
 * For bins, \f$r_{eff}\f$ is assumed to be the bin radius.
 *
 * \param aero_phase_idx Index of the aerosol phase within the representation
 * \param radius Effective particle radius (m)
 * \param partial_deriv \f$\frac{\partial r_{eff}}{\partial y}\f$ where \f$y\f$
 *                       are species on the state array
 * \param aero_rep_int_data Pointer to the aerosol representation integer data
 * \param aero_rep_float_data Pointer to the aerosol representation
 *                            floating-point data
 * \param aero_rep_env_data Pointer to the aerosol representation
 *                          environment-dependent parameters
 */
__device__ void aero_rep_modal_binned_mass_get_effective_radius__m(
    int aero_phase_idx, double *radius, double *partial_deriv,
    int *aero_rep_int_data, double *aero_rep_float_data) {
  int *int_data = aero_rep_int_data;
  double *float_data = aero_rep_float_data;

  for (int i_section = 0; i_section < int_data[0]; i_section++) {
    int int4i = int_data[4 + i_section];
    int intint4i1 = int_data[int4i + 1];
    for (int i_bin = 0; i_bin < int_data[int4i]; i_bin++) {
      aero_phase_idx -= intint4i1;
      if (aero_phase_idx < 0) {
        *radius =
            float_data[int_data[4 + int_data[0] + i_section] + i_bin * 3 + 1];
        // Effective radii are constant for bins and modes
        if (partial_deriv) {
          for (int i_phase = 0; i_phase < intint4i1; ++i_phase) {
            for (int i_elem = 0;
                 i_elem < int_data[int4i + 2 + 2 * int_data[int4i] * intint4i1 +
                                   i_bin * intint4i1 + i_phase];
                 ++i_elem) {
              *(partial_deriv++) = 0.0;
            }
          }
        }
        i_section = int_data[0];
        break;
      }
    }
  }
}

/// Minimum aerosol-phase mass concentration [kg m-3]
#define MINIMUM_MASS_ 1.0e-25L
/// Minimum mass assumed density [kg m-3]
#define MINIMUM_DENSITY_ 1800.0L
// volum = MINIMUM_MASS_ 1.0e-25L / MINIMUM_DENSITY_ 1800.0L

/** \brief Get the volume of an aerosol phase
 *
 * \param sc Pointer to the GPU model data (state, env, aero_phase)
 * \param aero_phase_idx Index of the aerosol phase to use in the calculation
 * \param state_var Pointer to the aerosol phase on the state variable array
 * \param volume Pointer to hold the aerosol phase volume
 *               (\f$\mbox{\si{\cubic\metre\per\cubic\metre}}\f$ or
 *                \f$\mbox{\si{\cubic\metre\per particle}}\f$)
 * \param jac_elem When not NULL, a pointer to an array whose length is the
 *                 number of Jacobian elements used in calculations of mass and
 *                 volume of this aerosol phase returned by
 *                 \c aero_phase_get_used_jac_elem and whose contents will be
 *                 set to the partial derivatives of total phase volume by
 *                 concentration \f$\frac{dv}{dy_i}\f$ of each component
 *                 species \f$y_i\f$.
 */
__device__ void aero_phase_get_volume__m3_m3(ModelDataGPU *md,
                                             int aero_phase_idx,
                                             double *state_var, double *volume,
                                             double *jac_elem) {
  // Get the requested aerosol phase data
  int *int_data =
      &(md->aero_phase_int_data[md->aero_phase_int_indices[aero_phase_idx]]);
  double *float_data = &(
      md->aero_phase_float_data[md->aero_phase_float_indices[aero_phase_idx]]);

  // Sum the mass and MW
  *volume = MINIMUM_MASS_ / MINIMUM_DENSITY_;
  int i_jac = 0;
  for (int i_spec = 0; i_spec < int_data[0]; i_spec++) {
    int spec_type = int_data[1 + i_spec];
    if (spec_type == CHEM_SPEC_VARIABLE || spec_type == CHEM_SPEC_CONSTANT ||
        spec_type == CHEM_SPEC_PSSA) {
      double density = float_data[int_data[0] + i_spec];
      *volume += state_var[i_spec] / density;
      if (jac_elem) {
        jac_elem[i_jac] = 1.0 / density;
        i_jac++;
      }
    }
  }
}

#undef MINUMUM_MASS_
#undef MINIMUM_DENSITY_

/** \brief Get the particle number concentration \f$n\f$
 * (\f$\mbox{\si{\#\per\cubic\metre}}\f$)
 *
 * The modal mass number concentration is calculated for a log-normal
 * distribution where the geometric mean diameter (\f$\tilde{D}_n\f$) and
 * geometric standard deviation (\f$\tilde{\sigma}_g\f$) are set by the aerosol
 * model prior to solving the chemistry. The number concentration is
 * calculated according to the equation given in Table 1 of Zender
 * \cite Zender2002 :
 * \f[
 *      n = N_0 = \frac{6V_0}{\pi}\tilde{D}_n^{-3}e^{-9
 * ln(\tilde{\sigma}_g)^2/2} \f] \f[ V_0 = \sum_i{\frac{m_i}{\rho_i}} \f] where
 * \f$\rho_i\f$ and \f$m_i\f$ are the density and total mass of species \f$i\f$
 * in the specified mode.
 *
 * The binned number concentration is calculated according to:
 * \f[
 *     n = V_0 / V_p
 * \f]
 * \f[
 *     V_p = \frac{4}{3}\pi r^{3}
 * \f]
 * where \f$r\f$ is the radius of the size bin and \f$V_0\f$ is defined as
 * above.
 *
 * \param sc Pointer to the structure containing the grid_cell data state array
 * \param mc Pointer to the GPU model data
 * \param aero_phase_idx Index of the aerosol phase within the representation
 * \param number_conc Particle number concentration, \f$n\f$
 *                    (\f$\mbox{\si{\#\per\cubic\centi\metre}}\f$)
 * \param partial_deriv \f$\frac{\partial n}{\partial y}\f$ where \f$y\f$ are
 *                      the species on the state array
 * \param aero_rep_int_data Pointer to the aerosol representation integer data
 * \param aero_rep_float_data Pointer to the aerosol representation
 *                            floating-point data
 * \param aero_rep_env_data Pointer to the aerosol representation
 *                          environment-dependent parameters
 */
__device__ void aero_rep_modal_binned_mass_get_number_conc__n_m3(
    ModelDataGPU *md, int aero_phase_idx, double *state, double *number_conc,
    double *partial_deriv, int *aero_rep_int_data, double *aero_rep_float_data,
    double *aero_rep_env_data) {
  int *int_data = aero_rep_int_data;
  double *float_data = aero_rep_float_data;

  for (int i_section = 0; i_section < int_data[0] && aero_phase_idx >= 0;
       i_section++) {
    int int4i = int_data[4 + i_section];
    int intint4i1 = int_data[int4i + 1];
    double loggsd = log(aero_rep_env_data[int_data[0] + i_section]);
    double modal_factor =
        6.0 /
        (M_PI * aero_rep_env_data[i_section] * aero_rep_env_data[i_section] *
         aero_rep_env_data[i_section] * exp(4.5 * loggsd * loggsd));

    for (int i_bin = 0; i_bin < int_data[int4i] && aero_phase_idx >= 0;
         i_bin++) {
      aero_phase_idx -= intint4i1;
      double bindp =
          float_data[int_data[4 + int_data[0] + i_section] - 1 + i_bin * 3] /
          2.0;
      double binned_factor =
          0.23873241463784300365353 / (bindp * bindp * bindp);

      if (aero_phase_idx < 0) {
        *number_conc =
            float_data[int_data[4 + int_data[0] + i_section] + i_bin * 3];
        if (partial_deriv) {
          for (int i_phase = 0; i_phase < intint4i1; ++i_phase) {
            // Get a pointer to the phase on the state array
            // double *state = (double *)(sc->grid_cell_state);
            state += int_data[int4i + 2 + i_bin * intint4i1 + i_phase] - 1;

            // Get the aerosol phase volume [m3 m-3]
            double phase_volume = 0.0;
            aero_phase_get_volume__m3_m3(
                md,
                int_data[int4i + 2 + int_data[int4i] * intint4i1 +
                         i_bin * intint4i1 + i_phase] -
                    1,
                state, &phase_volume, partial_deriv);
#define BINNED 1
#define MODAL 2
            // Convert d_vol/d_conc to d_number/d_conc
            int ilim = int_data[int4i + 2 + 2 * int_data[int4i] * intint4i1 +
                                i_bin * intint4i1 + i_phase];
            switch (int_data[int4i - 1]) {
            case (MODAL):
              for (int i_elem = 0; i_elem < ilim; ++i_elem) {
                *(partial_deriv++) *= modal_factor;
              }
              break;
            case (BINNED):
              for (int i_elem = 0; i_elem < ilim; ++i_elem) {
                *(partial_deriv++) *= binned_factor;
              }
              break;
            }
#undef BINNED
#undef MODAL
          }
        }
        i_section = int_data[0];
        break;
      }
    }
  }
}

/** \brief Get the mass and average MW in an aerosol phase
 *
 * \param md Pointer to the GPU model data
 * \param aero_phase_idx Index of the aerosol phase to use in the calculation
 * \param state_var Pointer the aerosol phase on the state variable array
 * \param mass Pointer to hold total aerosol phase mass
 *             (\f$\mbox{\si{\kilogram\per\cubic\metre}}\f$ or
 *              \f$\mbox{\si{\kilogram\per particle}}\f$)
 * \param MW Pointer to hold average MW of the aerosol phase
 *           (\f$\mbox{\si{\kilogram\per\mol}}\f$)
 * \param jac_elem_mass When not NULL, a pointer to an array whose length is the
 *                 number of Jacobian elements used in calculations of mass and
 *                 volume of this aerosol phase returned by
 *                 \c aero_phase_get_used_jac_elem and whose contents will be
 *                 set to the partial derivatives of mass by concentration
 *                 \f$\frac{dm}{dy_i}\f$ of each component species \f$y_i\f$.
 * \param jac_elem_MW When not NULL, a pointer to an array whose length is the
 *                 number of Jacobian elements used in calculations of mass and
 *                 volume of this aerosol phase returned by
 *                 \c aero_phase_get_used_jac_elem and whose contents will be
 *                 set to the partial derivatives of total molecular weight by
 *                 concentration \f$\frac{dMW}{dy_i}\f$ of each component
 *                 species \f$y_i\f$.
 */
__device__ void aero_phase_get_mass__kg_m3(ModelDataGPU *md, int aero_phase_idx,
                                           double *state_var, double *mass,
                                           double *MW, double *jac_elem_mass,
                                           double *jac_elem_MW) {
  // Get the requested aerosol phase data
  int *int_data =
      &(md->aero_phase_int_data[md->aero_phase_int_indices[aero_phase_idx]]);
  double *float_data = &(
      md->aero_phase_float_data[md->aero_phase_float_indices[aero_phase_idx]]);

  // Sum the mass and MW
  double l_mass = 1.0e-25L;
  double moles = 1.0e-26L;
  int i_jac = 0;
  for (int i_spec = 0; i_spec < int_data[0]; i_spec++) {
    int spec_type = int_data[1 + i_spec];
    if (spec_type == CHEM_SPEC_VARIABLE || spec_type == CHEM_SPEC_CONSTANT ||
        spec_type == CHEM_SPEC_PSSA) {
      l_mass += state_var[i_spec];
      moles += state_var[i_spec] / (double)float_data[i_spec];
      if (jac_elem_mass)
        jac_elem_mass[i_jac] = 1.0;
      if (jac_elem_MW)
        jac_elem_MW[i_jac] = 1.0 / float_data[i_spec];
      i_jac++;
    }
  }
  *MW = (double)l_mass / moles;
  if (jac_elem_MW) {
    for (int j_jac = 0; j_jac < i_jac; j_jac++) {
      jac_elem_MW[j_jac] =
          (moles - jac_elem_MW[j_jac] * l_mass) / (moles * moles);
    }
  }
  *mass = (double)l_mass;
}

/** \brief Get the total mass in an aerosol phase \f$m\f$
 * (\f$\mbox{\si{\kilogram\per\cubic\metre}}\f$)
 *
 * \param sc Pointer to the structure containing the grid_cell data state array
 * \param mc Pointer to the GPU model data
 * \param aero_phase_idx Index of the aerosol phase within the representation
 * \param aero_phase_mass Total mass in the aerosol phase, \f$m\f$
 *                        (\f$\mbox{\si{\kilogram\per\cubic\metre}}\f$)
 * \param partial_deriv \f$\frac{\partial m}{\partial y}\f$ where \f$y\f$ are
 *                      the species on the state array
 * \param aero_rep_int_data Pointer to the aerosol representation integer data
 * \param aero_rep_float_data Pointer to the aerosol representation
 *                            floating-point data
 */
__device__ void aero_rep_modal_binned_mass_get_aero_phase_mass__kg_m3(
    ModelDataGPU *md, int aero_phase_idx, double *state,
    double *aero_phase_mass, double *partial_deriv, int *aero_rep_int_data,
    double *aero_rep_float_data) {
  int *int_data = aero_rep_int_data;
  double *float_data = aero_rep_float_data;

  for (int i_section = 0; i_section < int_data[0] && aero_phase_idx >= 0;
       ++i_section) {
    int int4i = int_data[4 + i_section];
    int intint4i1 = int_data[int4i + 1];
    for (int i_bin = 0; i_bin < int_data[int4i] && aero_phase_idx >= 0;
         ++i_bin) {
      if (aero_phase_idx < 0 || aero_phase_idx >= intint4i1) {
        aero_phase_idx -= intint4i1;
        continue;
      }
      for (int i_phase = 0; i_phase < intint4i1; ++i_phase) {
        if (aero_phase_idx == 0) {
          *aero_phase_mass =
              float_data[int_data[4 + int_data[0] + i_section] - 1 +
                         3 * int_data[int4i] + i_bin * intint4i1 + i_phase];
          if (partial_deriv) {
            // Get a pointer to the phase on the state array
            // double *state = (double *)(sc->grid_cell_state);
            state += int_data[int4i + 2 + i_bin * intint4i1 + i_phase] - 1;

            // Get d_mass / d_conc
            double mass, mw;
            aero_phase_get_mass__kg_m3(
                md,
                int_data[int4i + 2 + int_data[int4i] * intint4i1 +
                         i_bin * intint4i1 + i_phase] -
                    1,
                state, &mass, &mw, partial_deriv, NULL);
            partial_deriv +=
                int_data[int4i + 2 + 2 * int_data[int4i] * intint4i1 +
                         i_bin * intint4i1 + i_phase];
          }

          // Other phases present in the bin or mode do not contribute to
          // the aerosol phase mass
        } else if (partial_deriv) {
          for (int i_elem = 0;
               i_elem < int_data[int4i + 2 + 2 * int_data[int4i] * intint4i1 +
                                 i_bin * intint4i1 + i_phase];
               ++i_elem) {
            *(partial_deriv++) = 0.0;
          }
        }
        aero_phase_idx -= 1;
      }
    }
  }
}

/** \brief Get the average molecular weight in an aerosol phase
 **        \f$m\f$ (\f$\mbox{\si{\kilogram\per\mole}}\f$)
 *
 * \param sc Pointer to the structure containing the grid_cell data state array
 * \param mc Pointer to the GPU model data
 * \param aero_phase_idx Index of the aerosol phase within the representation
 * \param aero_phase_avg_MW Average molecular weight in the aerosol phase
 *                          (\f$\mbox{\si{\kilogram\per\mole}}\f$)
 * \param partial_deriv \f$\frac{\partial m}{\partial y}\f$ where \f$y\f$ are
 *                      the species on the state array
 * \param aero_rep_int_data Pointer to the aerosol representation integer data
 * \param aero_rep_float_data Pointer to the aerosol representation
 *                            floating-point data
 */
__device__ void aero_rep_modal_binned_mass_get_aero_phase_avg_MW__kg_mol(
    ModelDataGPU *md, int aero_phase_idx, double *state,
    double *aero_phase_avg_MW, double *partial_deriv, int *aero_rep_int_data,
    double *aero_rep_float_data) {
  int *int_data = aero_rep_int_data;
  double *float_data = aero_rep_float_data;

  for (int i_section = 0; i_section < int_data[0] && aero_phase_idx >= 0;
       ++i_section) {
    int int4i = int_data[4 + i_section];
    int intint4i1 = int_data[int4i + 1];
    for (int i_bin = 0; i_bin < int_data[int4i] && aero_phase_idx >= 0;
         ++i_bin) {
      if (aero_phase_idx < 0 || aero_phase_idx >= intint4i1) {
        aero_phase_idx -= intint4i1;
        continue;
      }
      for (int i_phase = 0; i_phase < intint4i1; ++i_phase) {
        if (aero_phase_idx == 0) {
          *aero_phase_avg_MW =
              float_data[int_data[4 + int_data[0] + i_section] - 1 +
                         (3 + intint4i1) * int_data[int4i] + i_bin * intint4i1 +
                         i_phase];
          if (partial_deriv) {
            // Get a pointer to the phase on the state array
            // double *state = (double *)(sc->grid_cell_state);
            state += int_data[int4i + 2 + i_bin * intint4i1 + i_phase] - 1;

            // Get d_MW / d_conc
            double mass, mw;
            aero_phase_get_mass__kg_m3(
                md,
                int_data[int4i + 2 + int_data[int4i] * intint4i1 +
                         i_bin * intint4i1 + i_phase] -
                    1,
                state, &mass, &mw, NULL, partial_deriv);
            partial_deriv +=
                int_data[int4i + 2 + 2 * int_data[int4i] * intint4i1 +
                         i_bin * intint4i1 + i_phase];
          }

          // Other phases present in the bin/mode do not contribute to the
          // average MW of the aerosol phase
        } else if (partial_deriv) {
          for (int i_elem = 0;
               i_elem < int_data[int4i + 2 + 2 * int_data[int4i] * intint4i1 +
                                 i_bin * intint4i1 + i_phase];
               ++i_elem) {
            *(partial_deriv++) = 0.0;
          }
        }
        aero_phase_idx -= 1;
      }
    }
  }
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
__device__ static inline double
gpu_transition_regime_correction_factor(double mean_free_path__m,
                                        double radius__m, double alpha) {
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
__device__ static inline double
gpu_d_transition_regime_correction_factor_d_radius(double mean_free_path__m,
                                                   double radius__m,
                                                   double alpha) {
  double K_n = mean_free_path__m / radius__m;
  double topow =
      radius__m * (K_n * K_n + (1.0 + 0.283 * alpha) * K_n + 0.75 * alpha);
  return (0.75 * alpha * mean_free_path__m *
          (K_n * K_n + 2.0 * K_n + 1.0 - 0.467 * alpha)) /
         (topow * topow);
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
__device__ static inline double
gpu_gas_aerosol_transition_rxn_rate_constant(double diffusion_coeff__m2_s,
                                             double mean_free_path__m,
                                             double radius__m, double alpha) {
  return 4.0 * M_PI * radius__m * diffusion_coeff__m2_s *
         gpu_transition_regime_correction_factor(mean_free_path__m, radius__m,
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
__device__ static inline double
gpu_d_gas_aerosol_transition_rxn_rate_constant_d_radius(
    double diffusion_coeff__m2_s, double mean_free_path__m, double radius__m,
    double alpha) {
  return 4.0 * M_PI * diffusion_coeff__m2_s *
         (gpu_transition_regime_correction_factor(mean_free_path__m, radius__m,
                                                  alpha) +
          radius__m * gpu_d_transition_regime_correction_factor_d_radius(
                          mean_free_path__m, radius__m, alpha));
}

/** Calculate the gas-aerosol reaction rate for the continuum regime \cite
 * Tie2003
 * [\f$\mbox{m}^3\, \mbox{particle}^{-1}\, \mbox{s}^{-1}\f$]
 *
 * The rate constant \f$k_c\f$ is calculated as:
 * \f[
 *   k_c = \frac{4 \pi r^2_e}{\left(\frac{r}{D_g} + \frac{4}{v(T)\gamma}\right)}
 * \f]
 *
 * where \f$r\f$ is the particle radius [m],
 * \f$D_g\f$ is the gas-phase diffusion coefficient of the reactant
 * [\f$\mbox{m}^2\mbox{s}^{-1}\f$], \f$\gamma\f$ is the reaction probability
 * [unitless], and v is the mean free speed of the gas-phase reactant.
 *
 * @param diffusion_coeff__m2_s Diffusion coefficient of the gas species
 *  [\f$\mbox{m}^2\, \mbox{s}^{-1}\f$]
 *  @param mean_speed__m_s Mean speed of the gas molecule [m s-1]
 *  @param radius__m Particle radius [m]
 *  @param alpha Mass accomodation coefficient [unitless]
 */
__device__ static inline double
gpu_gas_aerosol_continuum_rxn_rate_constant(double diffusion_coeff__m2_s,
                                            double mean_speed__m_s,
                                            double radius__m, double alpha) {
  return 4.0 * M_PI * radius__m * radius__m /
         (radius__m / diffusion_coeff__m2_s + 4.0 / (mean_speed__m_s * alpha));
}

/** Calculate the derivative of the continuum-regime gas-aerosol reaction rate
 * constant by particle radius \cite Tie2003
 * \f[
 *   \frac{dk_c}{dr} = 4 \pi \frac{\frac{r^2}{D_g} +
 *      \frac{8r}{v(T) \gamma}}{\left(\frac{r}{D_g} + \frac{4}{v(T)
 * \gamma}\right)^2} \f]
 *
 * where \f$r\f$ is the particle radius [m],
 * \f$D_g\f$ is the gas-phase diffusion coefficient of the reactant
 * [\f$\mbox{m}^2\mbox{s}^{-1}\f$], \f$\gamma\f$ is the reaction probability
 * [unitless], and v is the mean free speed of the gas-phase reactant.
 *
 * @param diffusion_coeff__m2_s Diffusion coefficient of the gas species
 *  [\f$\mbox{m}^2\, \mbox{s}^{-1}\f$]
 *  @param mean_speed__m_s Mean speed of the gas molecule [m s-1]
 *  @param radius__m Particle radius [m]
 *  @param alpha Mass accomodation coefficient [unitless]
 */
__device__ static inline double
gpu_d_gas_aerosol_continuum_rxn_rate_constant_d_radius(
    double diffusion_coeff__m2_s, double mean_speed__m_s, double radius__m,
    double alpha) {
  double nom = radius__m * radius__m / diffusion_coeff__m2_s +
               8.0 * radius__m / (mean_speed__m_s * alpha);
  double denom =
      radius__m / diffusion_coeff__m2_s + 4.0 / (mean_speed__m_s * alpha);
  return 4.0 * M_PI * nom / (denom * denom);
}
#endif // AERO_SOLVER_DEV_H_