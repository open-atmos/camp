cd ../../../json-fortran-6.1.0
mkdir build
mkdir install
cd build
cmake -D SKIP_DOC_GEN:BOOL=TRUE -D CMAKE_INSTALL_PREFIX=$(pwd)/../install ..
make install
cd ../../camp/build/compile
