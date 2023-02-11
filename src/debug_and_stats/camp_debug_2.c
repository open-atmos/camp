/*
 * -----------------------------------------------------------------
 * Programmer(s): Christian G. Ruiz and Mario Acosta
 * -----------------------------------------------------------------
 * Copyright (C) 2022 Barcelona Supercomputing Center
 * SPDX-License-Identifier: MIT
 */

#include "camp_debug_2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../camp_solver.h"

#include <unistd.h>

#ifdef CAMP_USE_MPI
#include <mpi.h>
#endif

#ifdef ENABLE_NETCDF
#include <netcdf.h>

void nc(int status) { //handle netcdf error
  if (status != NC_NOERR) {
    fprintf(stderr, "%s\n", nc_strerror(status));
    exit(-1);
  }
}

void export

void export_cell_netcdf(SolverData *sd){

  /*
  printf("export_cell_netcdf start\n");
  ModelData *md = &(sd->model_data);
  int ncid;
  int nvars=5;
  int dimids[nvars], varids[nvars];
  char file_name[]="cell_";
  char s_icell[20];
  sprintf(s_icell,"%d",sd->icell);
  strcat(file_name,s_icell);
  sd->icell++;
  strcat(file_name,"timestep_");
  char s_tstep[20];
  sprintf(s_tstep,"%d",sd->tstep);
  strcat(file_name,s_tstep);
  if(sd->icell>=sd->n_cells_tstep){
    sd->icell=0;
    sd->tstep++;
  }
  strcat(file_name,".nc");
  char file_path[1024];
  getcwd(file_path, sizeof(file_path));
  strcat(file_path,"/");
  strcat(file_path,file_name);
  printf("Creating netcdf file at %s\n",file_path);
  nc(nc_create_par(file_path, NC_NETCDF4|NC_MPIIO, MPI_COMM_WORLD, MPI_INFO_NULL, &ncid));
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int ncells=md->n_cells;
  int i=0;
  nc(nc_def_dim(ncid,"nstate",md->n_per_cell_state_var*ncells*size,&dimids[i]));
  nc(nc_def_var(ncid, "state", NC_DOUBLE, 1, &dimids[i], &varids[i++]));
  nc(nc_def_dim(ncid,"nrxn_int_data",md->n_rxn_int_param*ncells*size,&dimids[i]));
  nc(nc_def_var(ncid, "rxn_int_data", NC_INT, 1, &dimids[i], &varids[i++]));
  nc(nc_def_dim(ncid,"nrxn_float_data",md->n_rxn_float_param*ncells*size,&dimids[i]));
  nc(nc_def_var(ncid, "rxn_float_data", NC_DOUBLE, 1, &dimids[i], &varids[i++]));
  nc(nc_def_dim(ncid,"nrxn_env_data",md->n_rxn_env_data*ncells*size,&dimids[i]));
  nc(nc_def_var(ncid, "rxn_env_data", NC_DOUBLE, 1, &dimids[i], &varids[i++]));
  nc(nc_def_dim(ncid,"ntotal_env",CAMP_NUM_ENV_PARAM_*ncells*size,&dimids[i]));
  nc(nc_def_var(ncid, "total_env", NC_DOUBLE, 1, &dimids[i], &varids[i++]));
  i=0;
  nc(nc_enddef(ncid));
  printf("export_cell_netcdf nc_enddef\n");
  for (int i = 0; i < nvars; i++)
    nc(nc_var_par_access(ncid, varids[i], 1, 0));
  i=0;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  size_t start=md->n_per_cell_state_var*ncells*rank; size_t count=md->n_per_cell_state_var*ncells;
  nc(nc_put_vara_double(ncid, varids[i++], &start, &count, md->total_state));
  start=md->n_rxn_int_param*ncells*rank; count=md->n_rxn_int_param*ncells;
  nc(nc_put_vara_int(ncid, varids[i++], &start, &count, md->rxn_int_data));
  start=md->n_rxn_float_param*ncells*rank; count=md->n_rxn_float_param*ncells;
  nc(nc_put_vara_double(ncid, varids[i++], &start, &count, md->rxn_float_data));
  start=md->n_rxn_env_data*ncells*rank; count=md->n_rxn_env_data*ncells;
  nc(nc_put_vara_double(ncid, varids[i++], &start, &count, md->rxn_env_data));
  start=CAMP_NUM_ENV_PARAM_*ncells*rank; count=CAMP_NUM_ENV_PARAM_*ncells;
  nc(nc_put_vara_double(ncid, varids[i++], &start, &count, md->total_env));
  i=0;
  nc(nc_close(ncid));
  printf("export_cell_netcdf end\n");
   */
  /*
  for (int i = 0; i < md->n_per_cell_state_var; i++) {
    printf("b rank %d %d %-le\n",rank,i,md->total_state[i]);
  }
   */
}

void import_cell_netcdf(SolverData *sd){
  printf("import_cell_netcdf start\n");
  ModelData *md = &(sd->model_data);
  int ncid;
  int nvars=5;
  int varids[nvars];
  char file_name[]="cell_1_timestep_1.nc";
  //char file_name[]="cell_";
  char file_path[1024];
  getcwd(file_path, sizeof(file_path));
  strcat(file_path,"/");
  strcat(file_path,file_name);
  printf("Opening netcdf file at %s\n",file_path);
  nc(nc_open_par(file_path, NC_NOWRITE|NC_PNETCDF, MPI_COMM_WORLD, MPI_INFO_NULL, &ncid));
  int i=0;
  nc(nc_inq_varid(ncid, "state", &varids[i++]));
  nc(nc_inq_varid(ncid, "rxn_int_data", &varids[i++]));
  nc(nc_inq_varid(ncid, "rxn_float_data", &varids[i++]));
  nc(nc_inq_varid(ncid, "rxn_env_data", &varids[i++]));
  nc(nc_inq_varid(ncid, "total_env", &varids[i++]));
  i=0;
  for (int i = 0; i < nvars; i++)
    nc(nc_var_par_access(ncid, varids[i], 1, 0));
  i=0;
  int ncells=md->n_cells;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  size_t start=md->n_per_cell_state_var*ncells*rank; size_t count=md->n_per_cell_state_var*ncells;
  nc(nc_get_vara_double(ncid, varids[i++], &start, &count, md->total_state));
  start=md->n_rxn_int_param*ncells*rank; count=md->n_rxn_int_param*ncells;
  nc(nc_get_vara_int(ncid, varids[i++], &start, &count, md->rxn_int_data));
  start=md->n_rxn_float_param*ncells*rank; count=md->n_rxn_float_param*ncells;
  nc(nc_get_vara_double(ncid, varids[i++], &start, &count, md->rxn_float_data));
  start=md->n_rxn_env_data*ncells*rank; count=md->n_rxn_env_data*ncells;
  nc(nc_get_vara_double(ncid, varids[i++], &start, &count, md->rxn_env_data));
  start=CAMP_NUM_ENV_PARAM_*ncells*rank; count=CAMP_NUM_ENV_PARAM_*ncells;
  nc(nc_get_vara_double(ncid, varids[i++], &start, &count, md->total_env));
  i=0;
  printf("import_cell_netcdf end\n");
  //md->rxn_int_data[n_rxn_int_param*ncells], md->rxn_float_data[n_rxn_float_param*ncells], md->rxn_env_data[md->n_rxn_env_data*ncells], md->total_env[CAMP_NUM_ENV_PARAM_*ncells]
  /*
  for (int i = 0; i < md->n_per_cell_state_var; i++) {
    printf("c rank %d %d %-le\n",rank,i,md->total_state[i]);
  }
   */
}
#endif
#ifndef CSR_MATRIX
void swapCSC_CSR2(int n_row, int n_col, int* Ap, int* Aj, double* Ax, int* Bp, int* Bi, double* Bx){

  int nnz=Ap[n_row];

  memset(Bp, 0, (n_row+1)*sizeof(int));

  for (int n = 0; n < nnz; n++){
    Bp[Aj[n]]++;
  }

  //cumsum the nnz per column to get Bp[]
  for(int col = 0, cumsum = 0; col < n_col; col++){
    int temp  = Bp[col];
    Bp[col] = cumsum;
    cumsum += temp;
  }
  Bp[n_col] = nnz;

  for(int row = 0; row < n_row; row++){
    for(int jj = Ap[row]; jj < Ap[row+1]; jj++){
      int col  = Aj[jj];
      int dest = Bp[col];

      Bi[dest] = row;
      Bx[dest] = Ax[jj];

      Bp[col]++;
    }
  }

  for(int col = 0, last = 0; col <= n_col; col++){
    int temp  = Bp[col];
    Bp[col] = last;
    last    = temp;
  }

}

#endif

void check_iszerod(long double *x, int len, const char *s){

#ifndef DEBUG_CHECK_ISZEROD

  int n_zeros=0;
  for (int i=0; i<len; i++){
    if(x[i]==0.0){
      printf("ZERO %s %d[%d]",s,i);
      exit(0);
    }
  }

#endif

}

void check_isnanld(long double *x, int len, const char *s){

#ifndef DEBUG_CHECK_ISNANLD

  int n_zeros=0;
  for (int i=0; i<len; i++){
    if(isnan(x[i])){
      printf("NAN %s %d[%d]",s,i);
      exit(0);
    }
  }

#endif

}

void check_isnand(double *x, int len, const char *s){

  int n_zeros=0;
  for (int i=0; i<len; i++){
    if(isnan(x[i])){
      printf("NAN %s %d[%d]",s,i);
      exit(0);
    }
  }

}

/*
void check_isnand(double *x, int len, int var_id){

  int n_zeros=0;
  for (int i=0; i<len; i++){
    if(isnan(x[i]))
      printf("NAN %d[%d]",var_id,i);
  }

}*/

/*
void print_int(int *x, int len, char *s){

  for (int i=0; i<len; i++){
    printf("%s %d[%d]",s,i);
  }

}

void print_double(double *x, int len, char *s){

  for (int i=0; i<len; i++){
    printf("%s %d[%d]",s,i);
  }

}
 */

int compare_doubles(double *x, double *y, int len, const char *s){

  int flag=1;
  double tol=0.01;
  //float tol=0.0001;
  double rel_error;
  int n_fails=0;
  for (int i=0; i<len; i++){
    if(x[i]==0)
      rel_error=0.;
    else
      rel_error=abs((x[i]-y[i])/x[i]);
      //rel_error=(x[i]-y[i]/(x[i]+1.0E-60));
    if(rel_error>tol){
      printf("compare_doubles %s rel_error %le for tol %le at [%d]: %le vs %le\n",
              s,rel_error,tol,i,x[i],y[i]);
      flag=0;
      n_fails++;
      if(n_fails==4)
        return flag;
    }
  }

  return flag;

}

void print_current_directory(){
  char cwd[1024];
  if (getcwd(cwd, sizeof(cwd)) != NULL) {
    printf("Current working dir: %s\n", cwd);
  } else {
    printf("getcwd() error");
  }
}

#ifdef CAMP_USE_GPU
void get_camp_config_variables(SolverData *sd){
  FILE *fp;
  char buff[255];
  char path[] = "config_variables_c_solver.txt";
  //printf("get_camp_config_variables\n");
  fp = fopen("config_variables_c_solver.txt", "r");
  if (fp == NULL){
    printf("Could not open file %s, setting use_cpu ON\n",path);
    sd->use_cpu=1;
    sd->use_gpu_cvode=0;
  }else{
    fscanf(fp, "%s", buff);
    if(strstr(buff,"USE_CPU=ON")!=NULL){
      sd->use_cpu=1;
    }
    else{
      sd->use_cpu=0;
    }
    fscanf(fp, "%s", buff);
    if(strstr(buff,"USE_GPU_CVODE=ON")!=NULL){
      sd->use_gpu_cvode=1;
    }
    else if(strstr(buff,"USE_GPU_CVODE=2")!=NULL){
      sd->use_gpu_cvode=2;
    }
    else{
      sd->use_gpu_cvode=0;
    }
    sd->nDevices = 1;
    fscanf(fp, "%d", &sd->nDevices);
    fclose(fp);
  }
  //printf("get_camp_config_variables\n");
}
#endif

void export_counters_open(SolverData *sd){
  ModelData *md = &(sd->model_data);
#ifdef CAMP_DEBUG_GPU
  //char rel_path[] = "../../../../../exported_counters_";
  //char rel_path[] =
  //        "/gpfs/scratch/bsc32/bsc32815/a2s8/nmmb-monarch/MODEL/SRC_LIBS/camp/"
  //        "test/monarch/exports/camp_input";  // monarch
  //char rel_path[]=
  //  "/gpfs/scratch/bsc32/bsc32815/gpucamp/exported_counters_";
  char rel_path[]=
          "out/exported_counters_";
  char rank_str[64];
  char path[1024];
#ifdef CAMP_USE_MPI
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
  int rank=0;
#endif
  if (rank==999){
    printf("Exporting profiling counters rank %d counterFail %d counterSolve"
           " %d\n", rank, sd->counterFail, sd->counterSolve);
  }
  sprintf(rank_str, "%d", rank);
  strcpy(path, rel_path);
  strcat(path, rank_str);
  strcat(path, ".csv");
  FILE *file;
  file = fopen(path, "w");
  if (file == NULL) {
    printf("Can't create file in function export_counters_open \n");
    exit(1);
  }
  fprintf(file, "mpi_rank %d\n", rank);
#endif
}



