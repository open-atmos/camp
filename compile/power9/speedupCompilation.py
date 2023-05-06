import os
import re

# class Conf:
# pass

def disable_dc():
    with open("../../CMakeLists.txt", "r") as file:
        contents = file.read()
    new_contents = contents.replace('#set_target_properties(camplib  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)',
                                    'set_target_properties(camplib  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)')
    with open("../../CMakeLists.txt", "w") as file:
        file.write(new_contents)

def enable_dc():
    with open("../../CMakeLists.txt", "r") as file:
        contents = file.read()
    new_contents = contents.replace('set_target_properties(camplib  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)',
                                    '#set_target_properties(camplib  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)')
    with open("../../CMakeLists.txt", "w") as file:
        file.write(new_contents)

if __name__ == "__main__":
    # print("main start")
    # with open("../../src/cuda/cvode_cuda.cu")
    #remove_dc()
     #enable_dc()
