/*
 * Copyright (C) 2022 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#ifndef CAMP_EBI_STRUCTS_H
#define CAMP_EBI_STRUCTS_H

typedef struct
{

    const int N_GC_SPC = 53;
    const int N_GC_SPC_CHEM = 72;
    char[100][16];
    float GC_MOLWT[100];

} MODULE_BSC_CHEM_DATA;

#endif //CAMP_EBI_STRUCTS_H
