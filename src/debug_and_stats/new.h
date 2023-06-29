//
// Created by cguzman on 29/06/23.
//

#ifndef CAMP_NEW_H
#define CAMP_NEW_H
#include "../camp_common.h"
#include "camp_debug_2.h"

void rxn_calc_deriv_new(SolverData *sd, double time_step);

#endif  // CAMP_NEW_H

//Use as a temporary file to work on a new development. Try to use work in a single function, and
//Once development is finished move it to the proper file.
//This is better than working directly in the proper file because we avoid recompiling other functions.
//It also sets a place for new developments, where they should be easier to find.