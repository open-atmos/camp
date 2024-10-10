/*
 * -----------------------------------------------------------------
 * Programmer(s): Christian G. Ruiz
 * -----------------------------------------------------------------
 * Copyright (C) 2022 Barcelona Supercomputing Center
 * SPDX-License-Identifier: MIT
 */

#include <string.h>
#include "camp_solver.h"

#ifdef PROFILE_SOLVING
#ifdef CAMP_USE_MPI
#include <mpi.h>
#endif
#endif

/**
 * @brief Generates the name of the export state file based on the MPI rank.
 *
 * @param filename The array to store the generated filename.
 */
void get_export_state_name(char filename[]) {
#ifdef PROFILE_SOLVING
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  char s_mpirank[64];
  strcpy(filename, "out/");
  sprintf(s_mpirank, "%d", rank);
  strcat(filename, s_mpirank);
  strcat(filename, "state.csv");
#endif
}

/**
 * Initializes the export state.
 *
 * This function is responsible for initializing the export state. It checks if
 * the PROFILE_SOLVING macro is defined and if so, it creates a file with the
 * given filename and closes it immediately. The export state is enabled only
 * for the process with rank 0.
 *
 */
void init_export_state() {
#ifdef PROFILE_SOLVING
  char filename[64];
  get_export_state_name(filename);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) printf("export_state enabled\n");
  FILE *fptr;
  fptr = fopen(filename, "w");
  fclose(fptr);
#endif
}

/**
 * Export the state of the solver.
 *
 * This function exports the state of the solver to a file. It is only executed
 * when the `PROFILE_SOLVING` macro is defined. The state is exported for each
 * cell in the model data. The exported state variables are stored in the
 * `total_state` array of the model data.
 *
 * @param sd A pointer to the SolverData struct.
 */
void export_state(SolverData *sd) {
#ifdef PROFILE_SOLVING
  ModelData *md = &(sd->model_data);
  char filename[64];
  get_export_state_name(filename);
  for (int k = 0; k < md->n_cells; k++) {
    FILE *fptr;
    fptr = fopen(filename, "a");
    int len = md->n_per_cell_state_var;
    double *x = md->total_state + k * len;
    for (int i = 0; i < len; i++) {
      fprintf(fptr, "%.17le\n", x[i]);
    }
    fclose(fptr);
  }
#endif
}

/**
 * @brief Joins and exports the state of the program.
 *
 * This function is responsible for joining and exporting the state of the
 * program. It is only executed when the macro PROFILE_SOLVING is defined.
 *
 * If the MPI_Comm_size is equal to 1, the function renames the file
 * "out/0state.csv" to "out/state.csv" and returns.
 *
 * If the MPI_Comm_size is greater than 1, the function is executed only by the
 * process with rank 0. It opens the file "out/state.csv" in write mode and
 * iterates over all the processes, opening their respective state files
 * ("out/<rank>state.csv") in read mode. It then reads the contents of each
 * input file and writes them to the output file. After that, it closes the
 * input file and removes it. Finally, it closes the output file.
 *
 * The function uses MPI_Barrier to synchronize all processes before proceeding.
 */
void join_export_state() {
#ifdef PROFILE_SOLVING
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size == 1) {
    rename("out/0state.csv", "out/state.csv");
    return;
  }
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    printf("join_export_state start\n");
    const char *outputFileName = "out/state.csv";
    FILE *outputFile = fopen(outputFileName, "w");
    for (int i = 0; i < size; i++) {
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
#endif
}

/**
 * Export statistics to a CSV file.
 *
 * This function exports statistics related to solving a problem to a CSV file
 * named "stats.csv". The statistics include the time taken by CVode solver
 * (`timeCVode`) and the average load balance (`avg_load_balance`). The function
 * checks if the current process rank is 0 before exporting the statistics. If
 * the file "stats.csv" does not exist or cannot be opened, an error message is
 * printed.
 *
 * @param sd A pointer to the SolverData structure containing the statistics to
 * be exported.
 */
void export_stats(SolverData *sd) {
#ifdef PROFILE_SOLVING
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    FILE *fptr;
    if (sd->iters_load_balance == 0)
      sd->iters_load_balance = 1;  // avoid division by 0
    if ((fptr = fopen("out/stats.csv", "w")) != NULL) {
      fprintf(fptr, "timeCVode,avg_load_balance\n");
      fprintf(fptr, "%.17le,%.17le", sd->timeCVode,
              sd->acc_load_balance / sd->iters_load_balance);
      fprintf(fptr, "\n");
      fclose(fptr);
    } else {
      printf("File '%s' does not exist.\n", "out/stats.csv");
    }
  }
#endif
}