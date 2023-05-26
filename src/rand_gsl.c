/* Copyright (C) 2010-2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

/** \file
 * \brief Wrapper routines for GSL random number functions.
 */

/* clang-format off */

#include <gsl/gsl_errno.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <stdio.h>
#include <time.h>

/** \brief Private internal-use variable to store the random number
 * generator.
 */
static gsl_rng *camp_rand_gsl_rng = NULL;

/** \brief Result code indicating successful completion.
 */
#define CAMP_RAND_GSL_SUCCESS      0
/** \brief Result code indicating initialization failure.
 */
#define CAMP_RAND_GSL_INIT_FAIL    1
/** \brief Result code indicating the generator was not initialized
 * when it should have been.
 */
#define CAMP_RAND_GSL_NOT_INIT     2
/** \brief Result code indicating the generator was already
 * initialized when an initialization was attempted.
 */
#define CAMP_RAND_GSL_ALREADY_INIT 3

/** \brief Initialize the random number generator with the given seed.
 *
 * This must be called before any other GSL random number functions
 * are called.
 *
 * \param seed The random seed to use.
 * \return CAMP_RAND_GSL_SUCCESS on success, otherwise an error code.
 * \sa camp_rand_finalize_gsl() to cleanup the generator.
 */
int camp_srand_gsl(int seed)
{
        if (camp_rand_gsl_rng) {
                return CAMP_RAND_GSL_ALREADY_INIT;
        }
        gsl_set_error_handler_off(); // turn off automatic error handling
        camp_rand_gsl_rng = gsl_rng_alloc(gsl_rng_mt19937);
        if (camp_rand_gsl_rng == NULL) {
                return CAMP_RAND_GSL_INIT_FAIL;
        }
        gsl_rng_set(camp_rand_gsl_rng, seed);
        return CAMP_RAND_GSL_SUCCESS;
}

/** \brief Cleanup and deallocate the random number generator.
 *
 * This must be called after camp_srand_gsl().
 *
 * \return CAMP_RAND_GSL_SUCCESS on success, otherwise an error code.
 */
int camp_rand_finalize_gsl()
{
        if (!camp_rand_gsl_rng) {
                return CAMP_RAND_GSL_NOT_INIT;
        }
        gsl_rng_free(camp_rand_gsl_rng);
        camp_rand_gsl_rng = NULL;
        return CAMP_RAND_GSL_SUCCESS;
}

/** \brief Generate a uniform random number in \f$[0,1)\f$.
 *
 * \param harvest A pointer to the generated random number.
 * \return CAMP_RAND_GSL_SUCCESS on success, otherwise an error code.
 */
int camp_rand_gsl(double *harvest)
{
        if (!camp_rand_gsl_rng) {
                return CAMP_RAND_GSL_NOT_INIT;
        }
        *harvest = gsl_rng_uniform(camp_rand_gsl_rng);
        return CAMP_RAND_GSL_SUCCESS;
}

/** \brief Generate a uniform random integer in \f$[1,n]\f$.
 *
 * \param n The upper limit of the random integer.
 * \param harvest A pointer to the generated random number.
 * \return CAMP_RAND_GSL_SUCCESS on success, otherwise an error code.
 */
int camp_rand_int_gsl(int n, int *harvest)
{
        if (!camp_rand_gsl_rng) {
                return CAMP_RAND_GSL_NOT_INIT;
        }
        *harvest = gsl_rng_uniform_int(camp_rand_gsl_rng, n) + 1;
        return CAMP_RAND_GSL_SUCCESS;
}

/** \brief Generate a normally-distributed random number.
 *
 * \param mean The mean of the distribution.
 * \param stddev The standard deviation of the distribution.
 * \param harvest A pointer to the generated random number.
 * \return CAMP_RAND_GSL_SUCCESS on success, otherwise an error code.
 */
int camp_rand_normal_gsl(double mean, double stddev, double *harvest)
{
        if (!camp_rand_gsl_rng) {
                return CAMP_RAND_GSL_NOT_INIT;
        }
        *harvest = gsl_ran_gaussian(camp_rand_gsl_rng, stddev) + mean;
        return CAMP_RAND_GSL_SUCCESS;
}

/** \brief Generate a Poisson-distributed random integer.
 *
 * \param mean The mean of the distribution.
 * \param harvest A pointer to the generated random number.
 * \return CAMP_RAND_GSL_SUCCESS on success, otherwise an error code.
 */
int camp_rand_poisson_gsl(double mean, int *harvest)
{
        if (!camp_rand_gsl_rng) {
                return CAMP_RAND_GSL_NOT_INIT;
        }
        *harvest = gsl_ran_poisson(camp_rand_gsl_rng, mean);
        return CAMP_RAND_GSL_SUCCESS;
}

/** \brief Generate a Binomial-distributed random integer.
 *
 * \param n The sample size for the distribution.
 * \param p The sample probability for the distribution.
 * \param harvest A pointer to the generated random number.
 * \return CAMP_RAND_GSL_SUCCESS on success, otherwise an error code.
 */
int camp_rand_binomial_gsl(int n, double p, int *harvest)
{
        unsigned int u;

        if (!camp_rand_gsl_rng) {
                return CAMP_RAND_GSL_NOT_INIT;
        }
        u = n;
        *harvest = gsl_ran_binomial(camp_rand_gsl_rng, p, u);
        return CAMP_RAND_GSL_SUCCESS;
}

/* clang-format on */
