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

void init_export_state_netcdf(SolverData *sd){
  ModelData *md = &(sd->model_data);
  if(md->n_cells==1){
    int size,rank;
    int comm=sd->comm;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    int n_state_nc=sd->n_cells_tstep*md->n_per_cell_state_var;
    sd->state_nc = (double *)malloc(sizeof(double) * n_state_nc);
  }else{
    //printf("init_export_state_netcdf\n");
    sd->icell=sd->n_cells_tstep;
  }
}

void export_state_netcdf(SolverData *sd){
  ModelData *md = &(sd->model_data);
  int rank;
  int comm=sd->comm;
  MPI_Comm_rank(comm, &rank);
  int n_state_nc=sd->n_cells_tstep*md->n_per_cell_state_var;
  if(md->n_cells==1 && sd->icell<sd->n_cells_tstep){ //gather
    //if(rank==0)printf("export_state gather\n");
    for (int i = 0; i < md->n_per_cell_state_var; i++) {
      sd->state_nc[i+sd->icell*md->n_per_cell_state_var]=md->total_state[i];
    }
    sd->icell++;
  }
  //printf("%d %d %d\n",n_state_nc,md->n_cells,sd->n_cells_tstep);
  if(sd->icell>=sd->n_cells_tstep){ //export all cells
    if(rank==0)printf("export_state start\n");
    int ncid;
    int nvars=1;
    int dimids[nvars], varids[nvars];
    char file_path[]="out/state.nc";
    nc(nc_create_par(file_path, NC_NETCDF4 | NC_MPIIO,
                     comm,MPI_INFO_NULL, &ncid));
    printf("Created netcdf file at %s rank %d\n",file_path, rank);
    int i=0;
    int size;
    MPI_Comm_size(comm, &size);
    nc(nc_def_dim(ncid,"nstate",n_state_nc*size,&dimids[i]));
    nc(nc_def_var(ncid, "state", NC_DOUBLE, 1, &dimids[i], &varids[i]));
    nc(nc_enddef(ncid));
    size_t start = n_state_nc * rank;
    size_t count = n_state_nc;
    if(md->n_cells==1)
      nc(nc_put_vara_double(ncid, varids[i],&start, &count, sd->state_nc));
    else
      nc(nc_put_vara_double(ncid, varids[i],&start, &count, md->total_state));
    nc(nc_close(ncid));
    if(rank==0)printf("export_state end\n");
    /*
      if(sd->icell>=sd->n_cells_tstep){
        MPI_Barrier(comm);
        if(rank==0) printf("export_netcdf exit sd->icell %d\n", sd->icell);
        MPI_Barrier(comm);
        int err=0;
        MPI_Abort(1,err);
        exit(0);
      }
    */
  }
}


void export_netcdf(SolverData *sd){
  int comm=sd->comm;
  int rank;
  MPI_Comm_rank(comm, &rank);
  if(rank==0)printf("export_netcdf start\n");
  ModelData *md = &(sd->model_data);
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
  strcat(file_name,"mpirank_");
  char s_mpirank[20];
  sprintf(s_mpirank,"%d",rank);
  strcat(file_name,s_mpirank);
  strcat(file_name,".nc");
  char file_path[1024];
  getcwd(file_path, sizeof(file_path));
  strcat(file_path,"/out/");
  strcat(file_path,file_name);
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
/*
  if(sd->icell>=sd->n_cells_tstep){
    MPI_Barrier(comm);
    if(rank==0) printf("export_netcdf exit sd->icell %d\n", sd->icell);
    MPI_Barrier(comm);
    int err=0;
    MPI_Abort(1,err);
    exit(0);
  }
  */
}

void export_cells_netcdf(SolverData *sd) {
  int comm=sd->comm;
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
  nc(nc_create_par(file_path, NC_NETCDF4 | NC_MPIIO, comm,
                   MPI_INFO_NULL, &ncid));
  int size;
  MPI_Comm_size(comm, &size);
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
  MPI_Comm_rank(comm, &rank);
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
  int comm=sd->comm;
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
    MPI_Comm_rank(comm, &rank);
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
  MPI_Barrier(comm);
  printf("join_netcdfs end, exiting... \n");
  exit(0);
}

void import_multi_cell_netcdf(SolverData *sd){
  int comm=sd->comm;
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
  nc(nc_open_par(file_path, NC_NOWRITE|NC_PNETCDF, comm, MPI_INFO_NULL, &ncid));
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
  MPI_Comm_rank(comm, &rank);
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
}

void import_one_cell_netcdf(SolverData *sd){
  int comm=sd->comm;
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
  nc(nc_open_par(file_path, NC_NOWRITE|NC_PNETCDF, comm, MPI_INFO_NULL, &ncid));
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
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
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
}

void import_netcdf(SolverData *sd){
  ModelData *md = &(sd->model_data);
  if(md->n_cells > 1) import_multi_cell_netcdf(sd);
  else import_one_cell_netcdf(sd);
}

void cell_netcdf(SolverData *sd){
  int comm=sd->comm;
  int rank;
  MPI_Comm_rank(comm, &rank);
  if(sd->tstep==0) {
#ifndef EXPORT_NETCDF
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

void init_export_state(SolverData *sd){
  int comm=sd->comm;
  ModelData *md = &(sd->model_data);
  printf("MPI_COMM_WORLD %d COMM %d \n",MPI_COMM_WORLD,comm);
  int rank;
  MPI_Comm_rank(comm, &rank);
  printf("COMM %d \n",comm);
  char file_path[]="out/state.csv";
  if(rank==0){
    FILE *fptr;
    fptr = fopen(file_path,"w");//overwrite file
    fclose(fptr);
  }
}

void export_state(SolverData *sd){
  int comm=sd->comm;
  ModelData *md = &(sd->model_data);
  int rank;
  MPI_Comm_rank(comm, &rank);
  if(rank==0)printf("export_state start\n");
  char file_path[]="out/state.csv";
  int size;
  MPI_Comm_size(comm, &size);
  //int access=0;
  //if(rank==0)access=1;
  for (int k=0; k<md->n_cells; k++) {
    for (int j = 0; j < size; j++) {
      //if (access && rank == j) {
      if (rank == j) {
        FILE *fptr;
        fptr = fopen(file_path, "a");
        // maybe compare time with print cell-by-cell instead of gather all
        // maybe move to print_double
        int len = md->n_per_cell_state_var;
        double *x = md->total_state + k * md->n_per_cell_state_var;
        for (int i = 0; i < len; i++) {
          fprintf(fptr, "%.17le\n",x[i]);
        }
        fclose(fptr);
        //MPI_Send(&access, 1, MPI_INT, access, 123, comm);
      }
      MPI_Barrier(comm);
    }
  }
  if(rank==0)printf("export_state end\n");
}


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

void print_double(double *x, int len, const char *s){
#ifdef USE_PRINT_ARRAYS
  for (int i=0; i<len; i++){
    printf("%s[%d]=%.17le\n",s,i,x[i]);
  }
#endif
}

void print_int(int *x, int len, const char *s){
#ifdef USE_PRINT_ARRAYS
  for (int i=0; i<len; i++){
    printf("%s[%d]=%d\n",s,i,x[i]);
  }
#endif
}

int compare_doubles(double *x, double *y, int len, const char *s){
  int flag=1;
  double tol=0.;
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

int compare_long_doubles(long double *x, long double *y, int len, const char *s){
  int flag=1;
  double tol=0.;
  double rel_error;
  int n_fails=0;
  for (int i=0; i<len; i++){
    if(x[i]==0)
      rel_error=0.;
    else
      rel_error=abs((x[i]-y[i])/x[i]);
    //rel_error=(x[i]-y[i]/(x[i]+1.0E-60));
    if(rel_error>tol){
      printf("compare_long_doubles %s rel_error %le for tol %le at [%d]: %le vs %le\n",
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
  sd->use_cpu=1;
  sd->use_gpu_cvode=0;
  sd->use_new=0;
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
  }else{
    fscanf(fp, "%s", buff);
    if(!strstr(buff,"USE_CPU=ON")!=NULL){
      sd->use_cpu=0;
    }
    fscanf(fp, "%s", buff);
    if(strstr(buff,"USE_GPU_CVODE=ON")!=NULL){
      sd->use_gpu_cvode=1;
    }
    fscanf(fp, "%d", &sd->nDevices);
    fscanf (fp, "%d", &sd->nCellsGPUPerc);
    fscanf(fp, "%s", buff);
    if(strstr(buff,"New")!=NULL){
      sd->use_new=1;
    }
    fclose(fp);
  }
}
#endif