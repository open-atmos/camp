#!/bin/sh

####################################################
#        UPDATE SPECIES NAME
####################################################
#
# This script modifies the species name in the aerosol
# JSON configuration files of the selected aerosol mechanis
# to be consistent with the selected gas-phase mechanism.
# It relies on a text file located in the aerosol mechanism
# directory that indicates the name correspondance with one
# or several different gas-phase mechanism. If the selected
# aerosol mechanism is not yet mapped with the selected
# gas-phase mechanism, then the configuration cannot be used
# in CAMP-boxmodel.
#
# Versions:
#       08/05/2024: Creation (Herve Petetin, herve.petetin@bsc.es)
#
####################################################


RED='\033[0;31m'  #(red)
NC='\033[0m'      #(default color)

# usage instructions and/or other information
echo -e "$RED|==========================================$NC"
echo -e "$RED|       UPDATE SPECIES NAMES$NC"
echo -e "$RED|==========================================$NC"
if [ "$#" -ne 3 ] ; then
    echo -e "$RED| Usage:  ./update_species_names.sh <mechanism_gas>[:<version_gas>] <mechanism_aerosol>[:<version_aerosol>] <output_file>$NC"
    echo -e "$RED|     with <mechanism_gas> the name of the gas-phase mechanism$NC"
    echo -e "$RED|          <version> its version (by default the last version is chosen)$NC"
    echo -e "$RED|          <mechanism_aerosol> the name of the aerosol-phase mechanism$NC"
    echo -e "$RED|          <version_aerosol> its version (by default the last version is chosen)$NC"
    echo -e "$RED|          <output_file> the path of the updated aerosol mechanism JSON file$NC"
    echo -e "$RED| Examples:$NC"
    echo -e "./update_species_names.sh cb05:v1.0.0 shrivastava_vbs_SOA_scheme ./mechanism_cb05shri.json"
    exit 0
fi

# read arguments
mechanism_gas_version=$1
mechanism_aerosol_version=$2
updated_mechanism_aerosol_jsonfile=$3

# specify CAMP mechanism directory
camp_mechanism_dir=/gpfs/projects/bsc32/models/camp/data/mechanisms

# get gas/aerosol mechanism and version (last version by default)
mechanism_gas=$(echo "${mechanism_gas_version}" | awk -F':' '{print $1}')
if [[ $mechanism_gas_version == *:* ]] ; then
    version_gas=$(echo "${mechanism_gas_version}" | awk -F':' '{print $2}')
else
    version_gas=$(ls "${camp_mechanism_dir}/gas/${mechanism_gas}" | sort | tail -n 1)
fi
mechanism_aerosol=$(echo "${mechanism_aerosol_version}" | awk -F':' '{print $1}')
if [[ $mechanism_aerosol_version == *:* ]] ; then
    version_aerosol=$(echo "${mechanism_aerosol_version}" | awk -F':' '{print $2}')
else
    version_aerosol=$(ls "${camp_mechanism_dir}/aerosol/${mechanism_aerosol}" | sort | tail -n 1)
fi


# print inputs and other information
mechanism_aerosol_jsonfile=${camp_mechanism_dir}/aerosol/${mechanism_aerosol}/${version_aerosol}/mechanism_cb05.json
echo -e "$RED| Inputs:$NC"
echo -e "$RED|     Input  JSON file: $mechanism_aerosol_jsonfile$NC"
echo -e "$RED|     Output JSON file: $mechanism_aerosol_jsonfile_updated$NC"
echo -e "$RED| Gas-phase     mechanism: $mechanism_gas (version $version_gas)$NC"
echo -e "$RED| Aerosol-phase mechanism: $mechanism_aerosol (version $version_aerosol)$NC"

# path specifying the species name in the aerosol mechanism and their equivalent in gas-phase mechanism(s)
naming_table_file=${camp_mechanism_dir}/aerosol/${mechanism_aerosol}/${version_aerosol}/naming_table.csv

# loop on species name to change
echo -e "$RED| Reading naming table ($naming_table_file)$NC"
while read -r line; do

    if [[ $line == *reactants* ]] ; then
	# get column index of the selected gas-phase mechanism
	index=$(awk -v mech="$mechanism_gas" -F';' '{for(i=1;i<=NF;i++) if($i==mech) print i}' <<< "$line")

	# copy aerosol mechanism json file
	cp $mechanism_aerosol_jsonfile $updated_mechanism_aerosol_jsonfile
	
    else
	# identify species name in aerosol and gas mechanism
	species_aerosol='"'$(awk -v idx="$index" -F';' '{print $1}' <<< "$line")'"'
	species_gas='"'$(awk -v idx="$index" -F';' '{print $idx}' <<< "$line")'"'

	# updating the aerosol mechanism json file
	echo -e "$RED|     Changing $species_aerosol into $species_gas$NC"
	sed -i "s|${species_aerosol}|${species_gas}|g" $updated_mechanism_aerosol_jsonfile
    fi
	

done < $naming_table_file

echo -e "$RED| Check differences with:$NC"
echo "diff $mechanism_aerosol_jsonfile $updated_mechanism_aerosol_jsonfile"
echo -e "$RED| Successfully completed.$NC"
