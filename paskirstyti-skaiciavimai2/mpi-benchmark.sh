#!/bin/bash
# this script intended run on linux
echo "Script to calculate mpirun times"

truncate --size 0 benchmark-times-mpi-1

for i in {1..12}; do

 # stderr print execution time @ rank=0
 dt=$( mpirun -np $i kmeans-mpi-1.py 2>&1 >/dev/null )
 echo "$i $dt" >> benchmark-times-mpi-1

done