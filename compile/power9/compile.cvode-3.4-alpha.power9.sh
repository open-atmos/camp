

if [ -z "$SUITE_SPARSE_CAMP_ROOT" ]; then
	SUITE_SPARSE_CAMP_ROOT=$(pwd)/../../../SuiteSparse
fi

#MPI integration:
#-D MPI_ENABLE=ON \
#-D CMAKE_C_COMPILER=$(which mpicc) \

#Dont remember why -O0 is needed
#-D CMAKE_CXX_FLAGS="-O0" \
#-D CMAKE_C_FLAGS="-O0" \

#tar -zxvf camp/cvode-3.4-alpha.tar.gz
cd ../../../cvode-3.4-alpha
#rm -r build
mkdir build
mkdir install
mkdir install/examples
cd build
#use -O0 to keep same results
cmake -D CMAKE_BUILD_TYPE=debug \
-D MPI_ENABLE:BOOL=TRUE \
-D KLU_ENABLE:BOOL=TRUE \
-D CUDA_ENABLE:BOOL=FALSE \
-D CAMP_PROFILING=ON \
-D CMAKE_C_COMPILER=$(which mpicc) \
-D EXAMPLES_ENABLE_CUDA=OFF \
-D KLU_LIBRARY_DIR=$SUITE_SPARSE_CAMP_ROOT/lib \
-D KLU_INCLUDE_DIR=$SUITE_SPARSE_CAMP_ROOT/include \
-D CMAKE_INSTALL_PREFIX=$(pwd)/../install \
-D EXAMPLES_ENABLE_C=OFF \
..
#-D EXAMPLES_INSTALL_PATH=$(pwd)/../install/examples .. \
#-D CMAKE_CXX_FLAGS="-O3 -lcudart -lcublas" \
#-D CMAKE_C_FLAGS ="-O3 -lcudart -lcublas" \
#-D CMAKE_CUDA_FLAGS="-Xcompiler="-fpermissive" -lcudart -lcublas" \
#-D EXAMPLES_ENABLE_C=OFF \
..

#make VERBOSE=1
#make -j 4 #not working
make install
cd ../../camp/build/compile


#./cvode-3.4-alpha/build/examples/cvode/serial/cvRoberts_klu
