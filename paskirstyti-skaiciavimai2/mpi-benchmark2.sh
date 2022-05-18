#!/bin/bash
# this script intended run on linux
echo "Script to calculate mpirun times"

truncate --size 0 benchmark-times-mpi-2

for i in {1..12}; do

 # stderr print execution time @ rank=0
 dt=$( mpirun -np $i ./kmeans-mpi-2.py --data-file data-big.npy --batch-seed 142 --seed 1 --stop-sad 0.05 --max-iters 10000 --batch-size 2048 2>&1 >/dev/null )
 echo "$i $dt" >> benchmark-times-mpi-2

done