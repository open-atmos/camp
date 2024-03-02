#include "openmp.h"

void openmp_bcg(){
    int iam = 0, np = 1;
#pragma omp parallel default(shared) private(iam, np)
    {
#if defined (_OPENMP)
        np = omp_get_num_threads();
     iam = omp_get_thread_num();
#endif
        printf("Hello from thread %d out of %d\n", iam, np);
    }
    exit(0);
}