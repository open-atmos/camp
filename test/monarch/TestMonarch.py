#
# Copyright (C) 2022 Barcelona Supercomputing Center and University of
# Illinois at Urbana-Champaign
# SPDX-License-Identifier: MIT
#

from mainMonarch import *


def all_timesteps():
  conf = TestMonarch()
  conf.casesOptim = []
  conf.timeSteps = 1
  conf.loads_gpu = [99]
  conf.cells = [1000]
  conf.mpiProcessesCaseBase = 1
  conf.caseBase = "CPU One-cell"
  #conf.caseBase = "GPU BDF"
  conf.mpiProcessesCaseOptimList = [1]
  conf.casesOptim = ["GPU BDF"]
  #conf.is_import = True
  conf.is_import_base = True
  #conf.profileCuda = "ncu"
  #conf.profileCuda = "nsys"
  datay = run_main(conf)
  plot_cases(conf, datay)


if __name__ == "__main__":
  all_timesteps()

"""
#TODO USEFUL IPC:
- Copy all from /gpfs/scratc^Cbsc32/bsc032756/ESIWACE_CAMP_PROFILING/camp/test/monarch
- Ejecutar tal cual y generara 3 ficheros
- Bajar en local los ficheros .pcf prv y row  
-(descargar con ssh de la maquina de transfer el paraver)
- Con el paraver en local abrir el .prv 
2n paso si funciona:
- modificar la trace_f.sh y cambiar el LD_PRELOAD: #probar con cuda lib: libcudampitrace.so export LD_PRELOAD=${EXTRAE_HOME}/lib/libmpitracef.so # For Fortran apps
- añadir -finstrument-functions a compilacion
- Añadir en el .xml las user functions que quiero profilear
- ejecutar "nm -a binario (mock_monarch)" y buscar los de cuda que quiero mirar, los nombres del binario ponerlos al profile
- añadir la de run_solver
- Si en la traza anterior no se han visto las de GPU, añadirlas en user functions. 

#Diferencia IPC y Useful IPC:
- El Useful IPC te elimina del contaje los eventos MPI que no son runnings (como MPI wait), ya que lo modifican erroneamente (un wait es muy rapido pero en verdad tiene mucha espera),
lo que no sabemos si detecta el CUDA.
"""
