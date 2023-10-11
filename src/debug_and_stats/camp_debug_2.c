/*
 * -----------------------------------------------------------------
 * Programmer(s): Christian G. Ruiz
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

void get_export_state_name(char filename[]){
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  char s_mpirank[64];
  strcpy(filename, "out/");
  sprintf(s_mpirank,"%d",rank);
  strcat(filename,s_mpirank);
  strcat(filename,"state.csv");
}

void init_export_state(SolverData *sd){
  char filename[64];
  get_export_state_name(filename);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(rank==0)
    printf("export_state enabled\n");
  FILE *fptr;
  fptr = fopen(filename,"w");
  fclose(fptr);
}

void export_state(SolverData *sd){
  ModelData *md = &(sd->model_data);
  char filename[64];
  get_export_state_name(filename);
  for (int k=0; k<md->n_cells; k++) {
    FILE *fptr;
    fptr = fopen(filename, "a");
    int len = md->n_per_cell_state_var;
    double *x = md->total_state + k * len;
    for (int i = 0; i < len; i++) {
      fprintf(fptr, "%.17le\n",x[i]);
    }
    fclose(fptr);
  }
}

void join_export_state(){
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if(size==1){
    rename("out/0state.csv","out/state.csv");
    return;
  }
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(rank==0){
  printf("join_export_state start\n");
  const char *outputFileName = "out/state.csv";
  FILE *outputFile = fopen(outputFileName, "w");
  for (int i = 0; i<size; i++) {
    char inputFileName[50];
    sprintf(inputFileName, "out/%dstate.csv", i);
    FILE *inputFile = fopen(inputFileName, "r");
    char buffer[1024];
    size_t bytesRead;
    while ((bytesRead = fread(buffer, 1, sizeof(buffer), inputFile)) > 0) {
      fwrite(buffer, 1, bytesRead, outputFile);
    }
    fclose(inputFile);
    remove(inputFileName);
  }
  fclose(outputFile);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

void init_export_stats(){
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  char file_path[]="out/stats.csv";
  if(rank==0){
    printf("export_stats enabled\n");
    FILE *fptr;
    fptr = fopen(file_path,"w");
    fprintf(fptr, "timecvStep,timeCVode\n");
    fclose(fptr);
  }
}

void export_stats(SolverData *sd){
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    FILE *fptr;
    fptr = fopen("out/stats.csv", "a");
    CVodeMem cv_mem = (CVodeMem) sd->cvode_mem;
    fprintf(fptr, "%.17le,",cv_mem->timecvStep);
    fprintf(fptr, "%.17le",sd->timeCVode);
    fprintf(fptr, "\n");
    fclose(fptr);
  }
  join_export_state();
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

void get_camp_config_variables(SolverData *sd){
  sd->use_cpu=1;
  sd->is_export_stats=0;
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
    printf("Could not open file %s, setting default values: is_cpu=ON is_export_stats=OFF\n",path);
  }else{
    fscanf(fp, "%s", buff);
    if(strstr(buff,"USE_CPU=OFF")!=NULL){
      sd->use_cpu=0;
    }
    fscanf(fp, "%d", &sd->nDevices);
    fscanf(fp, "%s", buff);
    if(strstr(buff,"IS_EXPORT_STATS=ON")!=NULL){
      sd->is_export_stats=1;
    }
    fclose(fp);
  }
}