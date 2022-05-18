#!/usr/bin/env python

from mpi4py import MPI
import sys

import numpy as np
comm = MPI.COMM_WORLD

name = MPI.Get_processor_name()
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

root_rank = 0

data = np.random.randint(1,100)

comm.barrier() # not necessary just for nice printing

print("%03d/%03d data before reduce: %s" % (rank,size,data))

comm.barrier()
data = comm.reduce(data,MPI.SUM, root=root_rank)

comm.barrier()

print("%03d/%03d data after reduce: %s" % (rank,size,data))