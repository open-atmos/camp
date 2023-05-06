
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

def split_functions_into_files():
    import os
    import re
    os.chdir("../../src/cuda")
    # Create the output directory if it doesn't exist
    output_dir = 'cvode_cuda'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Open the input file for reading
    with open('cvode_cuda.cu', 'r') as infile:
        # Read the contents of the file
        contents = infile.read()

        # Find all function definitions using a regular expression
        pattern = r'^(void|int|__device__)\s+(\w+)\s*\('
        matches = re.findall(pattern, contents, flags=re.MULTILINE)

        # Iterate over the matches and write each function to a new file
        for func_qualifier, func_name in matches:
            # Construct the output filename
            out_filename = os.path.join(output_dir, func_name + '.cu')

            # Find the start and end positions of the function definition
            pattern = r'^' + re.escape(func_qualifier) + r'\s+' + func_name + r'\s*\('
            start_match = re.search(pattern, contents, flags=re.MULTILINE)
            end_match = re.search(r'}\s*$', contents[start_match.end():], flags=re.MULTILINE)
            end_pos = start_match.end() + end_match.end()

            # Write the function to the output file
            with open(out_filename, 'w') as outfile:
                outfile.write(contents[start_match.start():end_pos])

if __name__ == "__main__":
    # print("main start")
    # with open("../../src/cuda/cvode_cuda.cu")
    #remove_dc()
     #enable_dc()
    split_functions_into_files()