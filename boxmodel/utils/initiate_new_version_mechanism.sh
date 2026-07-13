#!/bin/sh

####################################################
#        INITIATE NEW VERSION MECHANISM
####################################################
#
# This script copies the directory of a selected
# gas-, aerosol- or aqueous-phase mechanism with
# symbolic links, creating a new version from a
# selected earlier version. It allows the user
# to start a new version of the mechanism,
# from an existing one, using symbolic links
# in order to limit the disk space use for the
# CAMP data directory.
# From this
#
# Versions:
#       08/05/2024: Creation (Herve Petetin, herve.petetin@bsc.es)
#
####################################################

# define colors for bash
RED='\033[0;31m' #(red)
NC='\033[0m'     #(default color)

# usage instructions and/or other information
echo -e "$RED|==========================================$NC"
echo -e "$RED|       INITIATE NEW VERSION MECHANISM$NC"
echo -e "$RED|==========================================$NC"
if [ "$#" -eq 0 ]; then
    echo -e "$RED| Usage:  ./initiate_new_version_mechanism.sh <type> <mechanism> <version> <new_version>$NC"
    echo -e "$RED|        with <type> the mechanism type (gas, aerosol, aqueous)$NC"
    echo -e "$RED|             <mechanism> the name of the mechanism$NC"
    echo -e "$RED|             <version> the input version (by default the last one)$NC"
    echo -e "$RED|             <new_version> the name given to the newly created version$NC"
    echo -e "$RED| Examples:$NC"
    echo "./initiate_new_version_mechanism.sh gas cb05 v1.0.0 v2.0.0_test"
    exit 0
fi

# read arguments
type=$1
mechanism=$2
version=$3
new_version=$4
echo -e "$RED| Inputs:$NC"
echo -e "$RED|      Type        : $type$NC"
echo -e "$RED|      Mechanism   : $mechanism$NC"
echo -e "$RED|      Version     : $version$NC"
echo -e "$RED|      New version : $new_version$NC"

# current directory
initial_dir=$(pwd)

# create directory for the camp suite
echo -e "$RED| Create the output directory$NC"
input_dir=/gpfs/projects/bsc32/models/camp/data/mechanisms/$type/$mechanism/$version
output_dir=/gpfs/projects/bsc32/models/camp/data/mechanisms/$type/$mechanism/${new_version}
echo -e "$RED|     $output_dir$NC"
if [ -d ${output_dir} ]; then
    echo -e "$RED| ERROR: The version name you specified (${new_version}) already exists, please choose another one. Exiting...$NC"
    exit 0
fi
mkdir -p ${output_dir}
if [ ! -d ${output_dir} ]; then
    echo -e "$RED| ERROR: The new mechanism directory could not be created (${output_dir}), please check that you have write permissions.$NC"
    echo -e "$RED| Exiting...$NC"
    exit 0
fi

# loop on files
cd ${output_dir}
for file in "${input_dir}"/*; do
    if [ -f "$file" ]; then
        echo -e "$RED| Creating symbolic link for $file$NC"
        ln -s $file ./
    fi
done
cd ${initial_dir}

# Get the target file of the symbolic link
echo -e "$RED| Some potentially useful commands for next developments:$NC"
echo -e "$RED|    to replace link(s) by hard copy(ies):$NC"
echo "cd $output_dir"
for file in "${input_dir}"/*; do
    file=$(basename $file)
    echo "file=$file"' ; target=$(readlink -f "$file") ; rm $file ; cp $target ./'
done
echo -e "$RED|    to put files in read-only mode:$NC"
for file in "${input_dir}"/*; do
    echo "file=$file"' ; chmod a=r "$file"'
done

echo -e "$RED| Successfully completed.$NC"
