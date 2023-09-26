make_and_check(){
  cd ../../build
  unbuffer make | tee output_make.log
  make_exit_status=${PIPESTATUS[0]}
  if [ $make_exit_status -ne 0 ]; then
    exit 1
  fi
  if grep -q "Scanning dependencies" output_make.log; then
    echo "Changes made by 'make' command."
  else
    echo "No changes made by 'make' command."
  fi
  cd -
}
make_and_check


#python TestMonarch.py

make_and_checkNoOutput(){
  cd ../../build
  make_output="$(make)"
  make_exit_status=$?
  if [ $make_exit_status -ne 0 ]; then
    echo "Error: 'make' command failed."
    exit 1  # Exit the script with an error status
  else
    echo "$make_exit_status success"
  fi
  if [[ $make_output == *"Scanning dependencies"* ]]; then
      echo "Changes made by 'make' command."
  else
      echo "No changes made by 'make' command."
  fi
}

make_and_checkNoFail(){
  cd ../../build
  make_output="$(make)"
  if [[ $make_output == *"Scanning dependencies"* ]]; then
      echo "Changes made by 'make' command."
  else
      echo "No changes made by 'make' command."
  fi
}

make_and_checkNoColors(){
  cd ../../build
  make 2>&1 | tee output_make.log
  make_exit_status=${PIPESTATUS[0]}  # Capture the exit status of the 'make' command
  if [ $make_exit_status -ne 0 ]; then
    echo "Error: 'make' command failed."
    exit 1  # Exit the script with an error status
  else
    echo "$make_exit_status success"
  fi
  if grep -q "Scanning dependencies" output_make.log; then
    echo "Changes made by 'make' command."
  else
    echo "No changes made by 'make' command."
  fi
  cd -
}



make_and_check2(){
  #make_camp
  #make_output=$(make_camp)
  make_output=""
  make_camp | tee /dev/tty | make_output
}

make_and_check0(){
  make_output=$(make_camp)
  if [[ $make_output == *"Scanning dependencies"* ]]; then
      echo "Changes made by 'make' command."
  else
      echo "No changes made by 'make' command."
  fi
}
