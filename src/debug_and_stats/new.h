//
// Created by cguzman on 29/06/23.
//

#ifndef CAMP_NEW_H
#define CAMP_NEW_H

#ifdef NEW

#include "../camp_common.h"
#include "camp_debug_2.h"

void rxn_calc_deriv_new(SolverData *sd);
void rxn_get_ids(SolverData *sd);
void rxn_free();

#define RXN_ARRHENIUS 1
#define RXN_TROE 2
#define RXN_CMAQ_H2O2 3
#define RXN_CMAQ_OH_HNO3 4
#define RXN_PHOTOLYSIS 5
#define RXN_HL_PHASE_TRANSFER 6
#define RXN_AQUEOUS_EQUILIBRIUM 7
#define RXN_SIMPOL_PHASE_TRANSFER 10
#define RXN_CONDENSED_PHASE_ARRHENIUS 11
#define RXN_FIRST_ORDER_LOSS 12
#define RXN_EMISSION 13
#define RXN_WET_DEPOSITION 14

#endif
#endif  // CAMP_NEW_H

//Use as a temporary file to work on a new development. Try to use work in a single function, and
//Once development is finished move it to the proper file.
//This is better than working directly in the proper file because we avoid recompiling other functions.
//It also sets a place for new developments, where they should be easier to find.