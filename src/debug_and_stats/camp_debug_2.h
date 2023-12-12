/*
 * -----------------------------------------------------------------
 * Programmer(s): Christian G. Ruiz and Mario Acosta
 * -----------------------------------------------------------------
 * Copyright (C) 2022 Barcelona Supercomputing Center
 * SPDX-License-Identifier: MIT
 */

#ifndef CAMP_DEBUG_2_H
#define CAMP_DEBUG_2_H

#include "../camp_common.h"

void init_export_state();
void export_state(SolverData *sd);
void join_export_state();
void export_stats(SolverData *sd);

#endif  // CAMP_DEBUG_2_H
