make_camp(){
  curr_path=$(pwd)
  cd  ../../build
  if ! make -j ${NUMPROC}; then
    exit
  fi
  cd $curr_path
}
make_camp

python TestMonarch.py