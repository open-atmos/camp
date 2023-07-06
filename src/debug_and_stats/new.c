//
// Created by cguzman on 29/06/23.
//

#include "new.h"

#ifdef NEW

static int rxn_step;
static int len_rate_contrib_l;
static int *n_rates_spec;
static int *i_rate_contrib;

void rxn_add_id(unsigned int spec_id, int prod_or_loss, double rate){
  if(prod_or_loss == JACOBIAN_LOSS){
    if(rxn_step==0){
      n_rates_spec[spec_id]++;
      len_rate_contrib_l++;
    }
    else if(rxn_step==1){
      rate_contrib_l[i_rate_contrib[spec_id]+n_rates_spec[spec_id]]=rate;
      n_rates_spec[spec_id]++;
    }
  }
}

void rxn_gpu_first_order_loss_calc_deriv_contrib(int *rxn_int_data,
          double *rxn_env_data){
  int *int_data = rxn_int_data;
  if (int_data[2] >= 0) rxn_add_id(int_data[2],JACOBIAN_LOSS,rxn_env_data[0]);
}

void rxn_gpu_CMAQ_H2O2_calc_deriv_contrib(int *rxn_int_data,
          double *rxn_env_data){
  int *int_data = rxn_int_data;
  int i_dep_var = 0;
  for (int i_spec=0; i_spec<int_data[0]; i_spec++, i_dep_var++) {
    if (int_data[(2 + int_data[0]+int_data[1]+i_dep_var)] < 0) continue;
    rxn_add_id(int_data[(2 + int_data[0]+int_data[1]+i_dep_var)],JACOBIAN_LOSS,rxn_env_data[0]);
  }
  for (int i_spec=0; i_spec<int_data[1]; i_spec++, i_dep_var++) {
    if (int_data[(2 + int_data[0]+int_data[1]+i_dep_var)] < 0) continue;
    rxn_add_id(int_data[(2 + int_data[0]+int_data[1]+i_dep_var)],JACOBIAN_PRODUCTION,rxn_env_data[0]);
  }
}

void rxn_gpu_CMAQ_OH_HNO3_calc_deriv_contrib(int *rxn_int_data,
          double *rxn_env_data){
  int *int_data = rxn_int_data;
  int i_dep_var = 0;
  for (int i_spec=0; i_spec<int_data[0]; i_spec++, i_dep_var++) {
    if (int_data[(2 + int_data[0] + int_data[1] + i_dep_var)] < 0) continue;
    rxn_add_id(int_data[(2 + int_data[0] + int_data[1] + i_dep_var)],JACOBIAN_LOSS,rxn_env_data[0]);
  }
  for (int i_spec=0; i_spec<int_data[1]; i_spec++, i_dep_var++) {
    if (int_data[(2 + int_data[0] + int_data[1] + i_dep_var)] < 0) continue;
    rxn_add_id(int_data[(2 + int_data[0] + int_data[1] + i_dep_var)],JACOBIAN_PRODUCTION,rxn_env_data[0]);
  }
}

void rxn_gpu_arrhenius_calc_deriv_contrib(int *rxn_int_data, double *rxn_env_data){
  int *int_data = rxn_int_data;
    int i_dep_var = 0;
    for (int i_spec=0; i_spec<int_data[0]; i_spec++, i_dep_var++) {
      if (int_data[2 + int_data[0] + int_data[1] + i_dep_var] < 0) continue;
      rxn_add_id(int_data[2 + int_data[0] + int_data[1] + i_dep_var],JACOBIAN_LOSS,rxn_env_data[0]);
    }
    for (int i_spec=0; i_spec<int_data[1]; i_spec++, i_dep_var++) {
      if (int_data[2 + int_data[0] + int_data[1] + i_dep_var] < 0) continue;
      rxn_add_id(int_data[2 + int_data[0] + int_data[1] + i_dep_var],JACOBIAN_PRODUCTION,rxn_env_data[0]);
    }
}

void rxn_gpu_troe_calc_deriv_contrib(int *rxn_int_data,
          double *rxn_env_data){
  int *int_data = rxn_int_data;
  int i_dep_var = 0;
  for (int i_spec=0; i_spec<int_data[0]; i_spec++, i_dep_var++) {
    if (int_data[(2 + int_data[0]+int_data[1]+i_dep_var)] < 0) continue;
    rxn_add_id(int_data[(2 + int_data[0]+int_data[1]+i_dep_var)],JACOBIAN_LOSS,rxn_env_data[0]);
  }
  for (int i_spec=0; i_spec<int_data[1]; i_spec++, i_dep_var++) {
    if (int_data[(2 + int_data[0]+int_data[1]+i_dep_var)] < 0) continue;
    rxn_add_id(int_data[(2 + int_data[0]+int_data[1]+i_dep_var)],JACOBIAN_PRODUCTION,rxn_env_data[0]);
  }
}

void rxn_gpu_photolysis_calc_deriv_contrib(int *rxn_int_data,
          double *rxn_env_data){
  int *int_data = rxn_int_data;
  int i_dep_var = 0;
  for (int i_spec=0; i_spec<int_data[0]; i_spec++, i_dep_var++) {
    if (int_data[(3 + int_data[0]+int_data[1]+i_dep_var)] < 0) continue;
    rxn_add_id(int_data[(3 + int_data[0]+int_data[1]+i_dep_var)],JACOBIAN_LOSS,rxn_env_data[0]);
  }
  for (int i_spec=0; i_spec<int_data[1]; i_spec++, i_dep_var++) {
    if (int_data[(3 + int_data[0]+int_data[1]+i_dep_var)] < 0) continue;
    rxn_add_id(int_data[(3 + int_data[0]+int_data[1]+i_dep_var)],JACOBIAN_PRODUCTION,rxn_env_data[0]);
  }
}

void rxns_get_ids(SolverData *sd){
  ModelData *md=&sd->model_data;
  int n_rxn = md->n_rxn;
  md->grid_cell_state = md->total_state;
  md->grid_cell_env = md->total_env;
  md->grid_cell_rxn_env_data = md->rxn_env_data;
  for (int i_rxn = 0; i_rxn < n_rxn; i_rxn++) {
    int *rxn_int_data =
        &(md->rxn_int_data[md->rxn_int_indices[i_rxn]]);
    double *rxn_env_data =
        &(md->grid_cell_rxn_env_data[md->rxn_env_idx[i_rxn]]);
    int rxn_type = *(rxn_int_data++);
    switch (rxn_type) {
      case RXN_ARRHENIUS:
        rxn_gpu_arrhenius_calc_deriv_contrib(rxn_int_data,rxn_env_data
                                             );
        break;
      case RXN_CMAQ_H2O2:
        rxn_gpu_CMAQ_H2O2_calc_deriv_contrib(rxn_int_data, rxn_env_data
                                             );
        break;
      case RXN_CMAQ_OH_HNO3:
        rxn_gpu_CMAQ_OH_HNO3_calc_deriv_contrib(rxn_int_data, rxn_env_data
                                                );
        break;
      case RXN_FIRST_ORDER_LOSS:
        rxn_gpu_first_order_loss_calc_deriv_contrib(rxn_int_data,rxn_env_data
            );
        break;
      case RXN_PHOTOLYSIS:
        rxn_gpu_photolysis_calc_deriv_contrib(rxn_int_data,rxn_env_data
                                              );
        break;
      case RXN_TROE:
        rxn_gpu_troe_calc_deriv_contrib(rxn_int_data, rxn_env_data
                                        );
        break;
      default:
        printf("ERROR: The default case of so-so switch was reached.\n");
    }
  }
}

void transform_n_rates_spec_to_accumulative_index(SolverData *sd){
  ModelData *md = &(sd->model_data);
  int accum=0;
  for(int i=0; i<md->n_per_cell_state_var-1; i++){
    accum+=n_rates_spec[i];
    i_rate_contrib[i+1]=accum;
    n_rates_spec[i]=0;
  }
  n_rates_spec[md->n_per_cell_state_var]=0;
}

void rxn_get_ids(SolverData *sd){
  ModelData *md = &(sd->model_data);
  n_rates_spec = (int *) calloc(md->n_per_cell_state_var, sizeof(int));
  i_rate_contrib = (int *) calloc(md->n_per_cell_state_var, sizeof(int));
  len_rate_contrib_l=0;
  rxn_step=0;
  rxns_get_ids(sd);
  rxn_step=1;
  rate_contrib_l = (long double *)malloc((len_rate_contrib_l + 1) * sizeof(long double));
  transform_n_rates_spec_to_accumulative_index(sd);
  rxns_get_ids(sd);
}

void rxn_calc_deriv_new(SolverData *sd){
  ModelData *md=&sd->model_data;
  int len=(int)sd->time_deriv.num_spec;

  /*
  for(int i=0; i<md->n_per_cell_state_var; i++) {
    for(int j=0; j<n_rates_spec[i]; j++) {


      rate_contrib_l[n_rates_spec[j]];

      md->loss_rates_new[i]=;

    }
  }
  */

   for (int i=0; i<len; i++){
    md->loss_rates_new[i]=sd->time_deriv.loss_rates[i];
  }

  compare_long_doubles(sd->time_deriv.loss_rates, md->loss_rates_new, len,"sd->time_deriv.loss_rates");
  //printf("new end\n");
}

void rxn_free(){
  free(n_rates_spec);
  free(i_rate_contrib);
  free(rate_contrib_l);
}

#endif

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