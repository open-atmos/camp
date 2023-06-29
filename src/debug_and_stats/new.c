//
// Created by cguzman on 29/06/23.
//

#include "new.h"

void rxn_calc_deriv_new(SolverData *sd, double time_step){
  int len=sd->time_deriv.num_spec;

  for (int i=0; i<len; i++){
    sd->loss_rates_new[i]=sd->time_deriv.loss_rates[i];
  }



  compare_long_doubles(sd->time_deriv.loss_rates, sd->loss_rates_new, len,"sd->time_deriv.loss_rates");
  //printf("new end\n");
}

/*

 double *yout=sd->deriv_data;
for (int i=0; i<len; i++) {
 long double *r_p = sd->time_deriv.production_rates;
 long double *r_l = sd->time_deriv.loss_rates;
 if (r_p[i] + r_l[i] != 0.0) {
   double scale_fact;
   scale_fact = 1.0 / (r_p[i] + r_l[i]) /
                (1.0 / (r_p[i] + r_l[i]) +
                 MAX_PRECISION_LOSS / fabs(r_p[i] - r_l[i]));
   yout[i] =
       scale_fact * (r_p[i] - r_l[i]) + (1.0 - scale_fact) * (sd->jac_deriv_data[i]);
 } else {
   yout[i] = 0.0;
 }
}

 */

/*
 //wrong

 //if(memcmp(sd->time_deriv.loss_rates,sd->loss_rates_new,len*sizeof(long double))!=0)
 //printf("Wrong memcpy");

 */