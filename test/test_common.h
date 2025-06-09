/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */
/** \file
 * \brief common functions for c tests
 */
#ifndef CAMP_TEST_COMMON_H
#define CAMP_TEST_COMMON_H

#include <camp/camp_common.h>

// Tolerances
#define CAMP_TEST_COMMON_ABS_TOL 1.0e-30
#define CAMP_TEST_COMMON_REL_TOL 1.0e-10

// Assert function
#define ASSERT_MSG(y, z) camp_assert(__func__, __LINE__, y, z);
#define ASSERT(y) camp_assert(__func__, __LINE__, y, "Unknown error");
#define ASSERT_CLOSE_MSG(x, y, z)                                              \
  camp_assert_close(__func__, __LINE__, x, y, z);
#define ASSERT_CLOSE(x, y)                                                     \
  camp_assert_close(__func__, __LINE__, x, y, "Unknown error");

// Assert function def
int camp_assert(const char *func, const int line, bool eval,
                const char *message) {
  if (eval) {
    return 0;
  }
  printf("\n[ERROR] line %4d in %s(): %s", line, func, message);
  return 1;
}

// Assert close function def
int camp_assert_close(const char *func, const int line, double val1,
                      double val2, const char *message) {
  bool eval = val1 == val2 ? true
                           : fabs(val1 - val2) <= CAMP_TEST_COMMON_ABS_TOL ||
                                 2.0 * fabs(val1 - val2) / fabs(val1 + val2) <=
                                     CAMP_TEST_COMMON_REL_TOL;
  return camp_assert(func, line, eval, message);
}

#endif
