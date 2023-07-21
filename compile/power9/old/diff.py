
import datetime
import time
import diff_main
import subprocess
import os

def diff():
  st = datetime.datetime.now()
  os.system("/home/cguzman/CLionProjects/gpupartmc/venv/bin/python /home/cguzman/CLionProjects/gpupartmc/camp/compile/power9/diff.sh")

  et = datetime.datetime.now()
  elapsed_time = et - st
  print('Execution time:', elapsed_time, 'seconds')

if __name__ == "__main__":
  diff()

def diff2():
  st = datetime.datetime.now()
  #diff_main.main()

  cpp_file_path = 'diff.cpp'
  # Compile the C++ code
  executable_file_path = 'file_comparison'
  compile_command = f'g++ -o {executable_file_path} {cpp_file_path}'
  subprocess.run(compile_command, shell=True, check=True)

  # Run the compiled C++ program
  run_command = f'./{executable_file_path}'
  result = subprocess.run(run_command, shell=True, capture_output=True, text=True)

  # Print the result
  print(result.stdout)

  et = datetime.datetime.now()
  elapsed_time = et - st
  print('Execution time:', elapsed_time, 'seconds')
