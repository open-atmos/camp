/* Copyright (C) 2019 Christian Guzman
 * Licensed under the GNU General Public License version 1 or (at your
 * option) any later version. See the file COPYING for details.
 *
 * Header file for solver functions
 *
 */

#ifndef F_JAC_H_
#define F_JAC_H_

//#include <cusolverSp.h>
//#include <cuda_runtime_api.h>

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>


/*
extern "C" {
//#include "../../camp_solver.h"
#include "../cuda_structs.h"
#include "../rxns_gpu.h"
#include "../aeros/aero_rep_gpu_solver.h"
//#include "../time_derivative_gpu.h"
}
*/

/*
__device__
void cudaDevicef(
#ifdef CAMP_DEBUG_GPU
        int counterDeriv2,
#endif
        //check_model_state
        double threshhold, double replacement_value, int *flag,
        //f_gpu
        double time_step, int deriv_length_cell, int state_size_cell,
        int n_cells,
        int i_kernel, int threads_block, int n_shr_empty, double *y,
        ModelDataGPU md_object
); //Interface CPU/GPU
*/


#endif
