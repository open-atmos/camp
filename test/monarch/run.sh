set -e
make_and_check() {
  curr_path=$(pwd)
  cd ../../build
  unbuffer make | tee output_make.log
  make_exit_status=${PIPESTATUS[0]}
  if [ $make_exit_status -ne 0 ]; then
    exit 1
  fi
  cd $curr_path
  if grep -q "Scanning dependencies" ../../build/output_make.log; then
    echo "Changes made by 'make' command."
    python checkGPU.py
  fi
}

make_run() {
  curr_path=$(pwd)
  cd ../../build
  make -j 4
  cd $curr_path
  #python TestMonarch.py
  python checkGPU.py
}

make_run
#make_and_check
