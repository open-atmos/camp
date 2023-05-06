import os

os.chdir("../../src/cuda")


import os
import re

# Define the header file name
header_file = 'cvode_cuda_functions.h'

# Create the output directory if it doesn't exist
output_dir = 'cvode_cuda'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Open the header file for writing
with open(os.path.join(output_dir, header_file), 'w') as header:
    # Include the main header file at the top of the new header file
    header.write('#include "../cvode_cuda.h"\n\n')

    # Open the input file for reading
    with open('cvode_cuda.cu', 'r') as infile:
        # Read the contents of the file
        contents = infile.read()

        # Find all function definitions using a regular expression
        pattern = r'^(__device__\s+|__global__\s+)?(void|int)\s+(\w+)\s*\((.*?)\)'
        matches = re.findall(pattern, contents, flags=re.MULTILINE | re.DOTALL)

        # Write the function signatures to the header file with the correct qualifiers and parameters
        for func_qualifier, func_type, func_name, func_params in matches:
            # Remove any comments from the parameter list
            func_params = re.sub(r'/\*.*?\*/', '', func_params, flags=re.DOTALL)
            # Remove any whitespace or newlines from the parameter list
            func_params = re.sub(r'\s+', ' ', func_params).strip()
            # Write the function signature to the header file
            header.write(f'{func_qualifier}{func_type} {func_name}({func_params});\n')

        # Include the header file in all source files and write it only once
        for filename in os.listdir(output_dir):
            if filename.endswith('.cu'):
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'r+') as source:
                    source_contents = source.read()
                    # Check if the #include statement for the header file is already in the source file
                    if not re.search(r'#include\s+"%s"\s*\n?' % header_file, source_contents):
                        # If not, add the #include statement at the top of the file
                        source.seek(0)
                        source.write('#include "%s"\n\n%s' % (header_file, source_contents))
