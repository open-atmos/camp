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
#include <unistd.h>
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

void export_netcdf(SolverData *sd){
  printf("export_netcdf start\n");
  ModelData *md = &(sd->model_data);
  if(md->n_cells>1){
    printf("export_netcdf ERROR: Use One-cell mode, multicells do not work for this function md->n_cells %d\n",md->n_cells);
    exit(0);
  }
  int z=0;
  int ncid;
  int nvars=3;
  if(md->n_aero_rep_env_data!=0) nvars++;
  if(md->n_sub_model_env_data!=0) nvars++;
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
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  strcat(file_name,"mpirank_");
  char s_mpirank[20];
  sprintf(s_mpirank,"%d",rank);
  strcat(file_name,s_mpirank);
  strcat(file_name,".nc");
  char file_path[1024];
  getcwd(file_path, sizeof(file_path));
  strcat(file_path,"/");
  strcat(file_path,file_name);
  printf("Creating netcdf file at %s rank %d\n",file_path, rank);
  nc(nc_create(file_path, NC_CLOBBER, &ncid));
  printf("Created netcdf file at %s rank %d\n",file_path, rank);
  int i=0;
  nc(nc_def_dim(ncid,"nstate",md->n_per_cell_state_var,&dimids[i]));
  nc(nc_def_var(ncid, "state", NC_DOUBLE, 1, &dimids[i], &varids[i]));
  i++;
  nc(nc_def_dim(ncid,"nrxn_env_data",md->n_rxn_env_data,&dimids[i]));
  nc(nc_def_var(ncid, "rxn_env_data", NC_DOUBLE, 1, &dimids[i], &varids[i]));
  i++;
  if(md->n_aero_rep_env_data!=0) {
    nc(nc_def_dim(ncid, "naero_rep_env_data",
                  md->n_aero_rep_env_data, &dimids[i]));
    nc(nc_def_var(ncid, "aero_rep_env_data", NC_DOUBLE, 1, &dimids[i],
                  &varids[i]));
    i++;
  }
  if(md->n_sub_model_env_data!=0){
    nc(nc_def_dim(ncid,"nsub_model_env_data",md->n_sub_model_env_data,&dimids[i]));
    nc(nc_def_var(ncid, "sub_model_env_data", NC_DOUBLE, 1, &dimids[i], &varids[i]));
    i++;
  }
  nc(nc_def_dim(ncid,"ntotal_env",CAMP_NUM_ENV_PARAM_,&dimids[i]));
  nc(nc_def_var(ncid, "total_env", NC_DOUBLE, 1, &dimids[i], &varids[i]));
  i=0;
  nc(nc_enddef(ncid));
  nc(nc_put_var_double(ncid, varids[i], md->total_state));
  i++;
  nc(nc_put_var_double(ncid, varids[i], md->rxn_env_data));
  i++;
  if(md->n_aero_rep_env_data!=0) {
    nc(nc_put_var_double(ncid, varids[i], md->aero_rep_env_data));
    i++;
  }
  if(md->n_sub_model_env_data!=0) {
    nc(nc_put_var_double(ncid, varids[i], md->sub_model_env_data));
    i++;
  }
  nc(nc_put_var_double(ncid, varids[i], md->total_env));
  i++;
  i=0;
  nc(nc_close(ncid));
  printf("export_netcdf end\n");
/*
  for (int i = 0; i < md->n_per_cell_state_var; i++) {
    printf("b rank %d %d %-le\n",rank,i,md->total_state[i]);
  }
*/
  if(sd->icell>=sd->n_cells_tstep){
    printf("export_netcdf exit sd->icell %d\n", sd->icell);
    MPI_Barrier(MPI_COMM_WORLD);
    exit(0);
  }
}

void export_cells_netcdf(SolverData *sd) {
  printf("export_cells_netcdf start\n");
  ModelData *md = &(sd->model_data);
  int ncid;
  int nvars = 3;
  if(md->n_aero_rep_env_data!=0) nvars++;
  if(md->n_sub_model_env_data!=0) nvars++;
  int dimids[nvars], varids[nvars];
  char file_name[] = "cells_";
  char s_icell[20];
  int ncells = md->n_cells;
  sprintf(s_icell, "%d", ncells);
  strcat(file_name, s_icell);
  strcat(file_name, "timestep_");
  char s_tstep[20];
  sprintf(s_tstep, "%d", sd->tstep);
  strcat(file_name, s_tstep);
  strcat(file_name, ".nc");
  char file_path[1024];
  getcwd(file_path, sizeof(file_path));
  strcat(file_path, "/");
  strcat(file_path, file_name);
  printf("Creating netcdf file at %s\n", file_path);
  nc(nc_create_par(file_path, NC_NETCDF4 | NC_MPIIO, MPI_COMM_WORLD,
                   MPI_INFO_NULL, &ncid));
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int i = 0;
  nc(nc_def_dim(ncid, "nstate", md->n_per_cell_state_var * ncells * size,
                &dimids[i]));
  nc(nc_def_var(ncid, "state", NC_DOUBLE, 1, &dimids[i], &varids[i]));
  i++;
  nc(nc_def_dim(ncid, "nrxn_env_data", md->n_rxn_env_data * ncells * size,
                &dimids[i]));
  nc(nc_def_var(ncid, "rxn_env_data", NC_DOUBLE, 1, &dimids[i], &varids[i]));
  i++;
  if (md->n_aero_rep_env_data != 0) {
    nc(nc_def_dim(ncid, "naero_rep_env_data",
                  md->n_aero_rep_env_data * ncells * size, &dimids[i]));
    nc(nc_def_var(ncid, "aero_rep_env_data", NC_DOUBLE, 1, &dimids[i],
                  &varids[i]));
    i++;
  }
  if (md->n_sub_model_env_data != 0) {
    nc(nc_def_dim(ncid, "nsub_model_env_data",
                  md->n_sub_model_env_data * ncells * size, &dimids[i]));
    nc(nc_def_var(ncid, "sub_model_env_data", NC_DOUBLE, 1, &dimids[i],
                  &varids[i]));
    i++;
  }
  nc(nc_def_dim(ncid, "ntotal_env", CAMP_NUM_ENV_PARAM_ * ncells * size,
                &dimids[i]));
  nc(nc_def_var(ncid, "total_env", NC_DOUBLE, 1, &dimids[i], &varids[i]));
  i++;
  i = 0;
  nc(nc_enddef(ncid));
  for (int i = 0; i < nvars; i++) nc(nc_var_par_access(ncid, varids[i], 1, 0));
  i = 0;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  size_t start = md->n_per_cell_state_var * ncells * rank;
  size_t count = md->n_per_cell_state_var * ncells;
  nc(nc_put_vara_double(ncid, varids[i], &start, &count, md->total_state));
  i++;
  start = md->n_rxn_env_data * ncells * rank;
  count = md->n_rxn_env_data * ncells;
  nc(nc_put_vara_double(ncid, varids[i], &start, &count, md->rxn_env_data));
  i++;
  if (md->n_aero_rep_env_data != 0) {
    start = md->n_aero_rep_env_data * ncells * rank;
    count = md->n_aero_rep_env_data * ncells;
    nc(nc_put_vara_double(ncid, varids[i], &start, &count,
                          md->aero_rep_env_data));
    i++;
  }
  if (md->n_sub_model_env_data != 0){
    start = md->n_sub_model_env_data * ncells * rank;
    count = md->n_sub_model_env_data * ncells;
    nc(nc_put_vara_double(ncid, varids[i], &start, &count,
                        md->sub_model_env_data));
    i++;
  }
  start=CAMP_NUM_ENV_PARAM_*ncells*rank;
  count=CAMP_NUM_ENV_PARAM_*ncells;
  nc(nc_put_vara_double(ncid, varids[i], &start, &count, md->total_env));
  i++;
  i=0;
  nc(nc_close(ncid));
  printf("export_cells_netcdf end\n");
  /*
  for (int i = 0; i < md->n_per_cell_state_var; i++) {
    printf("b rank %d %d %-le\n",rank,i,md->total_state[i]);
  }
   */
}

void join_netcdfs(SolverData *sd){
  printf("join_netcdfs start\n");
  ModelData *md = &(sd->model_data);
  int ncells=md->n_cells;
  if(ncells<1){
    printf("ERROR: join_netcdfs ncells<2 ncells %d, when it needs multi-cells enabled\n", ncells);
    exit(0);
  }
  char s_tstep[20];
  sprintf(s_tstep, "%d", sd->tstep);
  for (int icell = 0; icell < ncells; icell++) {
    int ncid;
    int nvars = 3;
    if(md->n_aero_rep_env_data!=0) nvars++;
    if(md->n_sub_model_env_data!=0) nvars++;
    int varids[nvars];
    char file_name[] = "cell_";
    char s_icell[20];
    sprintf(s_icell, "%d", icell);
    strcat(file_name, s_icell);
    strcat(file_name, "timestep_");
    strcat(file_name, s_tstep);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    strcat(file_name,"mpirank_");
    char s_mpirank[20];
    sprintf(s_mpirank,"%d",rank);
    strcat(file_name,s_mpirank);
    strcat(file_name, ".nc");
    char file_path[1024];
    getcwd(file_path, sizeof(file_path));
    strcat(file_path, "/");
    strcat(file_path, file_name);
    printf("Opening netcdf file at %s\n", file_path);
    nc(nc_open(file_path, NC_NOWRITE, &ncid));
    int i = 0;
    size_t nstate;
    int dimid;
    nc(nc_inq_dimid(ncid, "nstate", &dimid));
    nc(nc_inq_dimlen(ncid, dimid, &nstate));
    if(nstate!=md->n_per_cell_state_var){
      printf("nstate!=md->n_per_cell_state_var %d %d",nstate,md->n_per_cell_state_var);
      exit(0);
    }
    nc(nc_inq_varid(ncid, "state", &varids[i]));
    i++;
    nc(nc_inq_varid(ncid, "rxn_env_data", &varids[i]));
    i++;
    if(md->n_aero_rep_env_data!=0){
      nc(nc_inq_varid(ncid, "aero_rep_env_data", &varids[i]));
      i++;
    }
    if(md->n_sub_model_env_data!=0){
      nc(nc_inq_varid(ncid, "sub_model_env_data", &varids[i]));
      i++;
    }
    nc(nc_inq_varid(ncid, "total_env", &varids[i]));
    i++;
    i = 0;
    size_t count = md->n_per_cell_state_var;
    nc(nc_get_var_double(ncid, varids[i], &md->total_state[icell*count]));
    i++;
    int n_rxn=md->n_rxn;
    count = md->n_rxn_env_data;
    nc(nc_get_var_double(ncid, varids[i], &md->rxn_env_data[icell*count]));
    i++;
    if(md->n_aero_rep_env_data!=0) {
      count = md->n_aero_rep_env_data;
      nc(nc_get_var_double(ncid, varids[i], &md->aero_rep_env_data[icell * count]));
      i++;
    }
    if(md->n_sub_model_env_data!=0) {
      count = md->n_sub_model_env_data;
      nc(nc_get_var_double(ncid, varids[i], &md->sub_model_env_data[icell * count]));
      i++;
    }
    count = CAMP_NUM_ENV_PARAM_;
    nc(nc_get_var_double(ncid, varids[i], &md->total_env[icell*count]));
    i++;
    i = 0;
  }
  export_cells_netcdf(sd);
  MPI_Barrier(MPI_COMM_WORLD);
  printf("join_netcdfs end, exiting... \n");
  exit(0);
}

void import_multi_cell_netcdf(SolverData *sd){ //Use to import files in one-cell and multicell
  printf("import_multi_cell_netcdf start\n");
  ModelData *md = &(sd->model_data);
  int ncid;
  int nvars=3;
  if(md->n_aero_rep_env_data!=0) nvars++;
  if(md->n_sub_model_env_data!=0) nvars++;
  int varids[nvars];
  char file_name[]="cells_";
  char s_icell[20];
  sprintf(s_icell,"%d",sd->n_cells_tstep);
  strcat(file_name,s_icell);
  int ncells=md->n_cells;
  strcat(file_name,"timestep_");
  char s_tstep[20];
  sprintf(s_tstep,"%d",sd->tstep);
  strcat(file_name,s_tstep);
  sd->tstep++;
  strcat(file_name,".nc");
  char file_path[1024];
  getcwd(file_path, sizeof(file_path));
  strcat(file_path,"/");
  strcat(file_path,file_name);
  printf("Opening netcdf file at %s\n",file_path);
  nc(nc_open_par(file_path, NC_NOWRITE|NC_PNETCDF, MPI_COMM_WORLD, MPI_INFO_NULL, &ncid));
  int i=0;
  nc(nc_inq_varid(ncid, "state", &varids[i]));
  i++;
  nc(nc_inq_varid(ncid, "rxn_env_data", &varids[i]));
  i++;
  if(md->n_aero_rep_env_data!=0){
  nc(nc_inq_varid(ncid, "aero_rep_env_data", &varids[i]));
  i++;
  }
  if(md->n_sub_model_env_data!=0){
  nc(nc_inq_varid(ncid, "sub_model_env_data", &varids[i]));
  i++;
  }
  nc(nc_inq_varid(ncid, "total_env", &varids[i]));
  i=0;
  for (int i = 0; i < nvars; i++)
    nc(nc_var_par_access(ncid, varids[i], 1, 0));
  i=0;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int icell;
  size_t start = md->n_per_cell_state_var * ncells * rank;
  size_t count = md->n_per_cell_state_var * ncells;
  nc(nc_get_vara_double(ncid, varids[i], &start, &count, md->total_state));
  i++;
  start = md->n_rxn_env_data * ncells * rank;
  count = md->n_rxn_env_data * ncells;
  nc(nc_get_vara_double(ncid, varids[i], &start, &count, md->rxn_env_data));
  i++;
  if(md->n_aero_rep_env_data!=0) {
    start = md->n_aero_rep_env_data * ncells * rank;
    count = md->n_aero_rep_env_data * ncells;
    nc(nc_get_vara_double(ncid, varids[i], &start, &count,
                          md->aero_rep_env_data));
    i++;
  }
  if(md->n_sub_model_env_data!=0) {
    start = md->n_sub_model_env_data * ncells * rank;
    count = md->n_sub_model_env_data * ncells;
    nc(nc_get_vara_double(ncid, varids[i], &start, &count,
                          md->sub_model_env_data));
    i++;
  }
  start = CAMP_NUM_ENV_PARAM_ * ncells * rank;
  count = CAMP_NUM_ENV_PARAM_ * ncells;
  nc(nc_get_vara_double(ncid, varids[i], &start, &count, md->total_env));
  i++;
  i = 0;
  printf("import_multi_cell_netcdf end\n");
/*
  for (int i = 0; i < md->n_per_cell_state_var; i++) {
    printf("c rank %d %d %-le\n",rank,i,md->total_state[i]);
  }i++
  */
}

void import_one_cell_netcdf(SolverData *sd){ //Use to import files in one-cell and multicell
  printf("import_one_cell_netcdf start\n");
  ModelData *md = &(sd->model_data);
  int ncid;
  int nvars=3;
  if(md->n_aero_rep_env_data!=0) nvars++;
  if(md->n_sub_model_env_data!=0) nvars++;
  int varids[nvars];
  char file_name[]="cells_";
  char s_icell[20];
  sprintf(s_icell,"%d",sd->n_cells_tstep);
  strcat(file_name,s_icell);
  strcat(file_name,"timestep_");
  char s_tstep[20];
  sprintf(s_tstep,"%d",sd->tstep);
  strcat(file_name,s_tstep);
  strcat(file_name,".nc");
  char file_path[1024];
  getcwd(file_path, sizeof(file_path));
  strcat(file_path,"/");
  strcat(file_path,file_name);
  printf("Opening netcdf file at %s\n",file_path);
  nc(nc_open_par(file_path, NC_NOWRITE|NC_PNETCDF, MPI_COMM_WORLD, MPI_INFO_NULL, &ncid));
  int i=0;
  nc(nc_inq_varid(ncid, "state", &varids[i]));
  i++;
  nc(nc_inq_varid(ncid, "rxn_env_data", &varids[i]));
  i++;
  if(md->n_aero_rep_env_data!=0){
    nc(nc_inq_varid(ncid, "aero_rep_env_data", &varids[i]));
    i++;
  }
  if(md->n_sub_model_env_data!=0){
    nc(nc_inq_varid(ncid, "sub_model_env_data", &varids[i]));
    i++;
  }
  nc(nc_inq_varid(ncid, "total_env", &varids[i]));
  i++;
  i=0;
  for (int i = 0; i < nvars; i++)
    nc(nc_var_par_access(ncid, varids[i], 1, 0));
  i=0;
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int ncells=sd->n_cells_tstep;
  size_t start = md->n_per_cell_state_var  * sd->icell + rank*(md->n_per_cell_state_var*ncells);
  size_t count = md->n_per_cell_state_var ;
  nc(nc_get_vara_double(ncid, varids[i], &start, &count, md->total_state));
  i++;
  start = md->n_rxn_env_data * sd->icell + rank*(md->n_rxn_env_data*ncells);
  count = md->n_rxn_env_data;
  nc(nc_get_vara_double(ncid, varids[i], &start, &count, md->rxn_env_data));
  i++;
  if(md->n_aero_rep_env_data!=0) {
    start = md->n_aero_rep_env_data * sd->icell +
            rank * (md->n_aero_rep_env_data * ncells);
    count = md->n_aero_rep_env_data;
    nc(nc_get_vara_double(ncid, varids[i], &start, &count,
                          md->aero_rep_env_data));
    i++;
  }
  if(md->n_sub_model_env_data!=0) {
    start = md->n_sub_model_env_data * sd->icell +
            rank * (md->n_sub_model_env_data * ncells);
    count = md->n_sub_model_env_data;
    nc(nc_get_vara_double(ncid, varids[i], &start, &count,
                          md->sub_model_env_data));
    i++;
  }
  start = CAMP_NUM_ENV_PARAM_ * sd->icell + rank*(CAMP_NUM_ENV_PARAM_*ncells);
  count = CAMP_NUM_ENV_PARAM_;
  nc(nc_get_vara_double(ncid, varids[i], &start, &count, md->total_env));
  i++;
  i = 0;
  sd->icell++;
  if(sd->icell>=sd->n_cells_tstep){
    sd->icell=0;
    sd->tstep++;
  }
/*
  printf("import_one_cell_netcdf end\n");
  for (int i = 0; i < md->n_per_cell_state_var; i++) {
    printf("c rank %d %d %-le\n",rank,i,md->total_state[i]);
  }
  printf("rank %d env %-le %-le",rank, md->total_env[i],md->total_env[i+1]);
*/
}

void import_netcdf(SolverData *sd){ //Use to import files in one-cell and multicell
  ModelData *md = &(sd->model_data);
  if(md->n_cells > 1) import_multi_cell_netcdf(sd);
  else import_one_cell_netcdf(sd);
}

void cell_netcdf(SolverData *sd){
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(sd->tstep==0) {
#ifdef EXPORT_NETCDF
    export_netcdf(sd);
#else
#ifdef JOIN_NETCDFS
    join_netcdfs(sd);
#else
#ifdef IMPORT_NETCDF
    import_netcdf(sd);
#endif
#endif
#endif
  }
}
#endif

void check_iszerod(long double *x, int len, const char *s){
#ifndef DEBUG_CHECK_ISZEROD
  int n_zeros=0;
  for (int i=0; i<len; i++){
    if(x[i]==0.0){
      printf("ZERO %s x[%d]",s,i);
      exit(0);
    }
  }
#endif
}

void check_isnanld(long double *x, int len, const char *s){
  int n_zeros=0;
  for (int i=0; i<len; i++){
    if(isnan(x[i])){
      printf("NAN %s x[%d]",s,i);
      exit(0);
    }
  }
}

void check_isnand(double *x, int len, const char *s){
  int n_zeros=0;
  for (int i=0; i<len; i++){
    if(isnan(x[i])){
      printf("NAN %s x[%d]",s,i);
      exit(0);
    }
  }
}

int compare_doubles(double *x, double *y, int len, const char *s){
  int flag=1;
  double tol=0.01;
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

#ifdef CAMP_DEBUG_MOCKMONARCH
void get_camp_config_variables(SolverData *sd){
  FILE *fp;
  char buff[255];
  char path[] = "settings/config_variables_c_solver.txt";
  fp = fopen(path, "r");
  if (fp == NULL){
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
      printf("Current working dir: %s\n", cwd);
    } else {
      printf("getcwd() error");
      exit(0);
    }
    printf("Could not open file %s, setting use_cpu ON and use_gpu_cvode OFF\n",path);
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
    else{
      sd->use_gpu_cvode=0;
    }
    fscanf(fp, "%d", &sd->nDevices);
    fscanf (fp, "%d", &sd->nCellsGPUPerc);
    fclose(fp);
  }
}
#endif