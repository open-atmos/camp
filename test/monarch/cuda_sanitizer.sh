set -e
compile_in_mode_debug(){
  BUILD_TYPE=$(grep CMAKE_BUILD_TYPE ../../build/CMakeCache.txt | cut -d= -f2)
  if [[ "${BUILD_TYPE,,}" == "debug" ]]; then
    initial_dir=$(pwd)
    cd  ../../build
    cd $initial_dir
  else
    initial_dir=$(pwd)
    cd ../../compile
    ./debug.compile.camp.sh
    cd $initial_dir
  fi
}
compile_in_mode_debug

# Redirect all stdout and stderr to both terminal and log file
exec > >(tee ../../build/log_cuda_sanitizer.txt) 2>&1

# TODO: Filter CUDA_ERROR_INVALID_CONTEXT

mpirun -np 1 --bind-to core compute-sanitizer --tool memcheck --leak-check=full --suppressions=suppressions.txt --error-exitcode=1 ../../build/mock_monarch

#mpirun -np 1 --bind-to core compute-sanitizer --tool synccheck ../../build/mock_monarch
#sed -i 's/"load_gpu": [^,}]*,/"load_gpu": 100,/' settings/TestMonarch.json  # Set load_gpu=100 for --tool initcheck
#mpirun -np 1 --bind-to core compute-sanitizer --tool initcheck --track-unused-memory yes ../../build/mock_monarch 2>&1 | tee -a ../../build/log_cuda_sanitizer.txt #Use with load_gpu=100
#mpirun -np 1 --bind-to core compute-sanitizer --tool racecheck ../../build/mock_monarch 2>&1 | tee -a ../../build/log_cuda_sanitizer.txt

echo "script continues"

compile_in_mode_release(){
  initial_dir=$(pwd)
  cd ../../compile
  ./compile.camp.sh
  cd $initial_dir
}
#compile_in_mode_release