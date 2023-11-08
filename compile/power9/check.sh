
cd ../../build
make -j 4
#make test
#./unit_test_aero_rep_single_particle
cd ../test/monarch
./checkGPU.sh
#python checkGPU.py