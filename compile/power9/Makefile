#FINE

#PA = /gpfs/scratch/bsc32/bsc32815/gpupartmc/
PA = /home/cguzman/CLionProjects/gpupartmc/
#PB = build
PB = cmake-build-debug
CAMP_INC   =  -I$(PA)camp/$(PB) -I$(PA)json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0/lib -I$(PA)SuiteSparse/include -I$(PA)cvode-3.4-alpha/install/include
CAMP_LIB   = -L$(PA)camp/$(PB) -lcamp -lcuda \
	-L$(PA)json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0/lib -ljsonfortran \
	 -L$(PA)SuiteSparse/lib $(PA)SuiteSparse/lib/libklu.so -L$(PA)cvode-3.4-alpha/install/lib \
	 -lsundials_nvecserial -lsundials_cvode -lsundials_sunlinsolklu -lsundials_sunmatrixsparse \
#-lklu or $(PA)SuiteSparse/lib/libklu.so , both fine

EXTLIBS = $(CAMP_LIB)
FINCS = $(CAMP_INC)
FC = mpifort
FFLAGS = -g -O0 -lstdc++ \
-Wl,-rpath=$(PA)SuiteSparse/lib \
-Wl,-rpath=$(PA)json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0/lib \
-Wl,-rpath=$(PA)cvode-3.4-alpha/install/lib \
 $(FINCS)
SRC = $(PA)camp/test/unit_rxn_data/test_rxn_photolysis.F90
OBJ = $(SRC:.c=.o)
MOD = $(PA)camp/$(PB)/camp_constants.mod
EXEC = mock_monarch

all: $(EXEC)

$(EXEC): $(OBJ)
	$(FC) $(FFLAGS) $(OBJ) $(EXTLIBS) -o $(EXEC)
clean:
	$(RM) $(EXEC)

.PHONY: clean