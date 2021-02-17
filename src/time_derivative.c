/* Copyright (C) 2019 Matthew Dawson
 * Licensed under the GNU General Public License version 2 or (at your
 * option) any later version. See the file COPYING for details.
 *
 * Functions of the time derivative structure
 *
 */
/** \file
 * \brief Functions of the time derivative structure
 */
#include "time_derivative.h"
#include <math.h>
#include <stdio.h>

int time_derivative_initialize(TimeDerivative *time_deriv,
                               unsigned int num_spec) {
  if (num_spec <= 0) return 0;

#ifdef TIME_DERIVATIVE_LONG_DOUBLE

  time_deriv->production_rates =
      (long double *)malloc(num_spec * sizeof(long double));
  if (time_deriv->production_rates == NULL) return 0;

  time_deriv->loss_rates =
      (long double *)malloc(num_spec * sizeof(long double));
  if (time_deriv->loss_rates == NULL) {
    free(time_deriv->production_rates);
    return 0;
  }

#else

  time_deriv->production_rates =
          (double *)malloc(num_spec * sizeof(double));
  if (time_deriv->production_rates == NULL) return 0;

  time_deriv->loss_rates =
          (double *)malloc(num_spec * sizeof(double));
  if (time_deriv->loss_rates == NULL) {
    free(time_deriv->production_rates);
    return 0;
  }

#endif

  //time_deriv->num_spec_per_cell = num_spec;
  //time_deriv->num_spec = num_spec*n_cells;
  //time_deriv->i_cell = 0;

  time_deriv->num_spec = num_spec;

#ifdef PMC_DEBUG
  time_deriv->last_max_loss_precision = 0.0;
#endif

  return 1;
}

void time_derivative_reset(TimeDerivative time_deriv) {
  for (unsigned int i_spec = 0; i_spec < time_deriv.num_spec; ++i_spec) {
    time_deriv.production_rates[i_spec] = 0.0;
    time_deriv.loss_rates[i_spec] = 0.0;
  }
}

#ifdef TIME_DERIVATIVE_OUTPUT_PTRS_1

void time_derivative_output(TimeDerivative time_deriv, double *dest_array,
                            double *deriv_est, unsigned int output_precision) {
  long double *r_p = time_deriv.production_rates;
  long double *r_l = time_deriv.loss_rates;

#ifdef PMC_DEBUG
  time_deriv.last_max_loss_precision = 1.0;
#endif

#ifdef PMC_DEBUG_DERIV
  printf("Time_deriv r_p r_l deriv_est scale_fact\n");
#endif
  for (unsigned int i_spec = 0; i_spec < time_deriv.num_spec; i_spec++) {
#ifdef PMC_DEBUG_DERIV
    //printf("deriv_est %-le\n", deriv_est[0]);
#endif
    double prec_loss = 1.0;
    if (*r_p + *r_l != 0.0) {
      //printf("r_p[0]: %-Le r_l[0]: %-Le r_p[0] + r_l[0]: %-Le\n",r_p[0], r_l[0], r_p[0] + r_l[0]);
      if (deriv_est) {
#ifdef TIME_DERIVATIVE_LONG_DOUBLE
        long double scale_fact;
#else
        double scale_fact;
#endif
        scale_fact =
            1.0 / (*r_p + *r_l) /
            (1.0 / (*r_p + *r_l) + MAX_PRECISION_LOSS / fabsl(*r_p - *r_l));
        *dest_array =
            scale_fact * (*r_p - *r_l) + (1.0 - scale_fact) * (*deriv_est);
#ifdef PMC_DEBUG_DERIV
        printf("%-le %-le %-le %-le\n", *r_p, *r_l, *deriv_est, scale_fact);
#endif
      } else {
        *dest_array = *r_p - *r_l;
      }
#ifdef PMC_DEBUG
      if (*r_p != 0.0 && *r_l != 0.0) {
        prec_loss = *r_p > *r_l ? 1.0 - *r_l / *r_p : 1.0 - *r_p / *r_l;
        if (prec_loss < time_deriv.last_max_loss_precision)
          time_deriv.last_max_loss_precision = prec_loss;
      }
#endif
    } else {
      *dest_array = 0.0;
    }
#ifndef BASIC_TIME_DERIVATIVE_RESET
#else
    *r_p=0.0;
    *r_l=0.0;
#endif
    ++r_p;
    ++r_l;
    ++dest_array;
    if (deriv_est) ++deriv_est;
#ifdef PMC_DEBUG
    if (output_precision == 1) {
      printf("\nspec %d prec_loss %le", i_spec, -log(prec_loss) / log(2.0));
    }
#endif
  }
}

#else

void time_derivative_output(TimeDerivative time_deriv, double *dest_array,
                            double *deriv_est, unsigned int output_precision) {
  long double *r_p = time_deriv.production_rates;
  long double *r_l = time_deriv.loss_rates;

#ifdef PMC_DEBUG
  time_deriv.last_max_loss_precision = 1.0;
#endif

#ifdef PMC_DEBUG_DERIV
  printf("Time_deriv r_p r_l deriv_est scale_fact\n");
#endif
  for (unsigned int i = 0; i < time_deriv.num_spec; i++) {
#ifdef PMC_DEBUG_DERIV
    //printf("deriv_est %-le\n", deriv_est[0]);
#endif
    double prec_loss = 1.0;
    if (r_p[i] + r_l[i] != 0.0) {
      //printf("r_p[0]: %-Le r_l[0]: %-Le r_p[0] + r_l[0]: %-Le\n",r_p[0], r_l[0], r_p[0] + r_l[0]);
      if (deriv_est) {
#ifdef TIME_DERIVATIVE_LONG_DOUBLE
        long double scale_fact;
#else
        double scale_fact;
#endif
        scale_fact =
                1.0 / (r_p[i] + r_l[i]) /
                (1.0 / (r_p[i] + r_l[i]) + MAX_PRECISION_LOSS / fabs(r_p[i]- r_l[i]));
        dest_array[i] =
                scale_fact * (r_p[i] - r_l[i]) + (1.0 - scale_fact) * (deriv_est[i]);
#ifdef PMC_DEBUG_DERIV
        printf("%-le %-le %-le %-le\n", r_p[i], r_l[i], deriv_est[i], scale_fact);
#endif
      } else {
        dest_array[i] = r_p[i] - r_l[i];
      }
#ifdef PMC_DEBUG
      if (r_p[i] != 0.0 && r_l[i] != 0.0) {
        prec_loss = r_p[i] > r_l[i] ? 1.0 - r_l[i] / r_p[i] : 1.0 - r_p[i] / r_l[i];
        if (prec_loss < time_deriv.last_max_loss_precision)
          time_deriv.last_max_loss_precision = prec_loss;
      }
#endif
    } else {
      dest_array[i] = 0.0;
    }
#ifndef BASIC_TIME_DERIVATIVE_RESET
#else
    r_p[i]=0.0;
    r_l[i]=0.0;
#endif
#ifdef PMC_DEBUG
    if (output_precision == 1) {
      printf("\nspec %d prec_loss %le", i_spec, -log(prec_loss) / log(2.0));
    }
#endif
  }
}

#endif

#ifdef TIME_DERIVATIVE_LONG_DOUBLE
void time_derivative_add_value(TimeDerivative time_deriv, unsigned int spec_id,
                               long double rate_contribution) {
#else
void time_derivative_add_value(TimeDerivative time_deriv, unsigned int spec_id,
                               double rate_contribution) {
#endif

  if (rate_contribution > 0.0) {
    time_deriv.production_rates[spec_id] += rate_contribution;
  } else {
    time_deriv.loss_rates[spec_id] += -rate_contribution;
  }
}

#ifdef PMC_DEBUG
double time_derivative_max_loss_precision(TimeDerivative time_deriv) {
  return -log(time_deriv.last_max_loss_precision) / log(2.0);
}
#endif

void time_derivative_free(TimeDerivative time_deriv) {
  free(time_deriv.production_rates);
  free(time_deriv.loss_rates);
}
