set -e

#count (change as desire)
curr_path=$(pwd)
cd ../../src/cuda
file1="cvode_cuda.cu"
text="__syncthreads();"
count=$(grep -c "$text" "$file1")
echo "Initial count of '$text': $count"

count=1
#remove, run, check
while [ $count -gt 0 ]; do
    #remove
    cd ../../src/cuda
    sed -i "0,/$text/s/$text//" "$file1"
    cd $curr_path
    #run
    cd ../../test/monarch
    file="../../compile/power9/tmp.txt"
    ./run.sh >&1 | tee $file
    #check
    search_text="NRMSE: "
    if ! grep -q "$search_text" "$file"; then
        echo "Text '$search_text' not found in $file. Exiting."

    fi

    cd $curr_path
    # Update count
    count=$((count - 1))
    echo removed 1
done


