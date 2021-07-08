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

#ifdef CSR_MATRIX
void swapCSC_CSR(int n_row, int n_col, int* Ap, int* Aj, double* Ax, int* Bp, int* Bi, double* Bx){

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
  int rel_error;
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

void get_camp_config_variables(SolverData *sd){

  //printf("get_camp_config_variables\n");

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

  //printf("get_camp_config_variables\n");

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



