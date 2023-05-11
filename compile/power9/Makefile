
CAMP_INC   =  -I/gpfs/scratch/bsc32/bsc32815/gpupartmc/camp/build -I/gpfs/scratch/bsc32/bsc32815/gpupartmc/json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0/lib -I/gpfs/scratch/bsc32/bsc32815/gpupartmc/SuiteSparse/include -I/gpfs/scratch/bsc32/bsc32815/gpupartmc/cvode-3.4-alpha/install/include
CAMP_LIB   = -L/gpfs/scratch/bsc32/bsc32815/gpupartmc/camp/build -lcamp -lcuda -lcudart -L/gpfs/scratch/bsc32/bsc32815/gpupartmc/json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0/lib -ljsonfortran -L/gpfs/scratch/bsc32/bsc32815/gpupartmc/SuiteSparse/lib -lklu -L/gpfs/scratch/bsc32/bsc32815/gpupartmc/cvode-3.4-alpha/install/lib -lsundials_nvecserial -lsundials_cvode -lsundials_sunlinsolklu -lsundials_sunmatrixsparse
EXTLIBS = $(CAMP_LIB)
FINCS = $(CAMP_INC)
FC = mpifort
FFLAGS = -g -Wl,--copy-dt-needed-entries,-rpath=/gpfs/scratch/bsc32/bsc32815/gpupartmc/json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0/lib $(FINCS)
SRC = /gpfs/scratch/bsc32/bsc32815/gpupartmc/camp/test/monarch/mock_monarch.F90 /gpfs/scratch/bsc32/bsc32815/gpupartmc/camp/test/monarch/camp_monarch_interface.F90
OBJ = $(SRC:.c=.o)
MOD = /gpfs/scratch/bsc32/bsc32815/gpupartmc/camp/build/camp_constants.mod
EXEC = mock_monarch

all: $(EXEC)

$(EXEC): $(OBJ)
	$(FC) $(FFLAGS) $(OBJ) $(EXTLIBS) -o $(EXEC)
clean:
	$(RM) $(OBJ)

.PHONY: clean