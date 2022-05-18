#!/usr/bin/env python
"""
Barrier demonstracija
"""

from mpi4py import MPI
import sys, datetime, os,time

from termcolor import colored

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()


start = time.monotonic()

random_number = 1 +  (rank % 5)
time.sleep(random_number)


print("%s %3d/%3d: Big process time before barrier : %3.3f ms" % (colored(name,attrs=['bold']),rank,size,(time.monotonic() - start) * 1000))

MPI.COMM_WORLD.barrier()

print("%s %3d/%3d: Big process time !!!after!!! barrier: %3.3f ms" % (colored(name,attrs=['bold']),rank,size,(time.monotonic() - start) * 1000))
