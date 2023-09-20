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

void init_export_state(SolverData *sd){
  ModelData *md = &(sd->model_data);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if(size!=1){
    printf("export_state is only for"
        " 1 process, use 1 process, disable or update export_state\n");
    exit(0);
  }
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  char file_path[]="out/state.csv";
  if(rank==0){
    printf("export_state enabled\n");
    FILE *fptr;
    fptr = fopen(file_path,"w");
    fclose(fptr);
  }
}

void export_state(SolverData *sd){
  ModelData *md = &(sd->model_data);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  for (int k=0; k<md->n_cells; k++) {
    if (rank == 0) {
      FILE *fptr;
      fptr = fopen("out/state.csv", "a");
      int len = md->n_per_cell_state_var;
      double *x = md->total_state + k * len;
      for (int i = 0; i < len; i++) {
        fprintf(fptr, "%.17le\n",x[i]);
      }
      fclose(fptr);
    }
  }
}

void init_export_stats(){
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  char file_path[]="out/stats.csv";
  if(rank==0){
    printf("export_stats enabled\n");
    FILE *fptr;
    fptr = fopen(file_path,"w");
    fprintf(fptr, "counterBCG,counterLS,"
      "countersolveCVODEGPU,countercvStep,"
      "timeLS,timeBiconjGradMemcpy,timeCVode,"
      "dtcudaDeviceCVode,dtPostBCG,timeAux,"
      "timeNewtonIteration,timeJac,"
      "timelinsolsetup,timecalc_Jac,"
      "timeRXNJac,timef,timeguess_helper,"
      "timecvStep\n");
    fclose(fptr);
  }
}

void export_stats(int ntimers,int ncounters, int *counters, double *times){
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    FILE *fptr;
    fptr = fopen("out/stats.csv", "a");
    fprintf(fptr, "%d",counters[0]);
    for (int i = 1; i < ncounters; i++) {
      fprintf(fptr, ",%d",counters[i]);
    }
    for (int i = 0; i < ntimers; i++) {
      fprintf(fptr, ",%.17le",times[i]);
    }
    fprintf(fptr, "\n");
    fclose(fptr);
  }
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
#ifndef USE_PRINT_ARRAYS
  for (int i=0; i<len; i++){
    printf("%s[%d]=%.17le\n",s,i,x[i]);
  }
#endif
}

void print_int(int *x, int len, const char *s){
#ifndef USE_PRINT_ARRAYS
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