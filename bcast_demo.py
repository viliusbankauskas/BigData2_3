#!/usr/bin/env python
"""
Bcast demo
"""

from mpi4py import MPI
import sys

import numpy as np

comm = MPI.COMM_WORLD

name = MPI.Get_processor_name()
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

if rank == 0:
  data = np.array([1,2,3,4,42])
else:
  data = None

comm.barrier()
print("before %s %03d/%03d: data %r" % (name,rank,size,data))

comm.barrier()

data = comm.bcast(data, root=0)

print("after  %s %03d/%03d: data %r" % (name,rank,size,data))

