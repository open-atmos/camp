
PA = /gpfs/scratch/bsc32/bsc32815/gpupartmc/
CAMP_INC   =  -I$(PA)camp/build -I$(PA)json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0/lib -I$(PA)SuiteSparse/include -I$(PA)cvode-3.4-alpha/install/include
CAMP_LIB   = -L$(PA)camp/build -lcamp -lcuda -lcudart -L$(PA)json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0/lib -ljsonfortran -L$(PA)SuiteSparse/lib -lklu -L$(PA)cvode-3.4-alpha/install/lib -lsundials_nvecserial -lsundials_cvode -lsundials_sunlinsolklu -lsundials_sunmatrixsparse
EXTLIBS = $(CAMP_LIB)
FINCS = $(CAMP_INC)
FC = mpifort
#FFLAGS = -g -Wl,--copy-dt-needed-entries,-rpath=$(PA)json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0/lib $(FINCS)
FFLAGS = -g -lstdc++ -Wl,-rpath=$(PA)json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0/lib $(FINCS)
SRC = $(PA)camp/test/monarch/mock_monarch.F90 $(PA)camp/test/monarch/camp_monarch_interface.F90
OBJ = $(SRC:.c=.o)
MOD = $(PA)camp/build/camp_constants.mod
EXEC = mock_monarch

all: $(EXEC)

$(EXEC): $(OBJ)
	$(FC) $(FFLAGS) $(OBJ) $(EXTLIBS) -o $(EXEC)
clean:
	$(RM) $(OBJ)

.PHONY: clean