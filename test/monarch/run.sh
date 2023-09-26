make_and_check(){
  cd ../../build
  unbuffer make | tee output_make.log
  make_exit_status=${PIPESTATUS[0]}
  if [ $make_exit_status -ne 0 ]; then
    exit 1
  fi
  cd -
  if grep -q "Scanning dependencies" ../../build/output_make.log; then
    echo "Changes made by 'make' command."
    python checkGPU.py
  fi
}
#make_and_check
python checkGPU.py

#python TestMonarch.py