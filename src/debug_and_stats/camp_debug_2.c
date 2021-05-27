/* Copyright (C) 2019 Christian Guzman
 * Licensed under the GNU General Public License version 1 or (at your
 * option) any later version. See the file COPYING for details.
 *
 * Debug and stats functions
 *
 */

#include "camp_debug_2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../camp_solver.h"

#include <unistd.h>

#ifdef PMC_USE_MPI
#include <mpi.h>
#endif

void check_iszerod(long double *x, int len, char *s){

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

void check_isnanld(long double *x, int len, char *s){

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

void check_isnand(double *x, int len, char *s){

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

void print_current_directory(){

  char cwd[1024];
  if (getcwd(cwd, sizeof(cwd)) != NULL) {
    printf("Current working dir: %s\n", cwd);
  } else {
    printf("getcwd() error");
  }

}

void get_camp_config_variables(SolverData *sd){

  FILE *fp;
  char buff[255];

  //print_current_directory();

  fp = fopen("config_variables_c_solver.txt", "r");
  if (fp == NULL){
    printf("Could not open file get_camp_config_variables");
  }
  fscanf(fp, "%s", buff);

  if(strstr(buff,"USE_CPU=ON")!=NULL){
    sd->use_cpu=1;
  }
  else{
    sd->use_cpu=0;
  }

  fclose(fp);
}

void export_counters_open(SolverData *sd)
{

  ModelData *md = &(sd->model_data);

#ifdef PMC_DEBUG_GPU

  //char rel_path[] = "../../../../../exported_counters_";
  //char rel_path[] =
  //        "/gpfs/scratch/bsc32/bsc32815/a2s8/nmmb-monarch/MODEL/SRC_LIBS/partmc/"
  //        "test/monarch/exports/camp_input";  // monarch
  //char rel_path[]=
  //  "/gpfs/scratch/bsc32/bsc32815/gpupartmc/exported_counters_";

  char rel_path[]=
          "out/exported_counters_";

  char rank_str[64];
  char path[1024];

#ifdef PMC_USE_MPI

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

  sd->file = fopen(path, "w");

  if (sd->file == NULL) {
    printf("Can't create file in function export_counters_open \n");
    exit(1);
  }

  fprintf(sd->file, "mpi_rank %d\n", rank);

#endif

}



