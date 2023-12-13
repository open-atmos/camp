#IN DEVELOPMENT
set -e
#count (change as desire)
curr_path=$(pwd)
cd ../../test/monarch
file="../../src/cuda/cvode_cuda.cu"
text="__syncthreads();"
tmp_file="../../compile/power9/tmp.txt"

count=$(grep -c "$text" "$file")
echo "Initial count of '$text': $count"

count=1

i=0
ne=1 #next occurrence
while [ $i -lt $count ]; do
    # Backup the file before making changes
    cp "$file" "$file.backup"

    # Remove next occurrence
    sed -z "0,/$text/s/$text/$ne" "$file"

    #run
    #./run.sh >&1 | tee $tmp_file
    echo "run finished."

    # Check NRMSE and recover file if necessary
    #check_and_recover
    if [ ! -f "$tmp_file" ]; then
        echo "File $tmp_file not found. Skipping recovery."
        return
    fi
    nrmse_line=$(grep "NRMSE: " "$tmp_file")
    if [ -z "$nrmse_line" ]; then
        echo "No 'NRMSE' found in $tmp_file. Skipping recovery."
        return
    fi
    nrmse_value=$(echo "$nrmse_line" | awk '{print $2}')
    if (( $(echo "$nrmse_value > 9.0E-12" | bc -l) )); then
        echo "NRMSE is greater than 9.0E-12. Recovering file and skipping iteration."
        cp "$tmp_file.backup" "$file"
        exit 1
    fi

    echo "occurrences removed."
    i=$((i + 1))
done

echo "All occurrences of '$text' removed."



