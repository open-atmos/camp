/* Copyright (C) 2019 Christian Guzman
 * Licensed under the GNU General Public License version 1 or (at your
 * option) any later version. See the file COPYING for details.
 *
 * Debug and stats functions
 *
 */
// todo fix this...move all to this folder and compile correctly...
// todo: i think is not necessary include all time stdio.h.. reorder includes to
// left only camp_common with the common includes

// todo gprof to csv:
// https://stackoverflow.com/questions/28872400/convert-simple-ascii-table-to-csv
// Nvidia remote profiling:
// http://docs.nvidia.com/cuda/nsight-eclipse-edition-getting-started-guide/index.html#remote-development

#include "camp_debug_2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../camp_solver.h"

#ifdef PMC_USE_MPI
#include <mpi.h>
#endif

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



