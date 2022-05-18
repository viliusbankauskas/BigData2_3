#!/usr/bin/env python
"""
allreduce demo
"""

from mpi4py import MPI
import sys

import numpy as np
comm = MPI.COMM_WORLD

name = MPI.Get_processor_name()
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

root_rank = 0


#data = np.random.randint(1,100)
data = rank

comm.barrier() # not necessary just for nice printing

print("%03d/%03d data before allreduce: %s" % (rank,size,data))


comm.barrier()


def reduce_max(a,b,dt):
  """
  aggregate function
  :param a: first operand
  :param b: second operan
  :param dt: mpi datatype
  
  """
  binlen_a = a
  binlen_b = b

  res = max(binlen_a,binlen_b)
  print("%d/%d executing `%r` op `%r` -> `%r`" % (rank,size,a,b,res))

  return res

def reduce_binmax(a,b,dt):
  """
  aggregate function
  :param a: first operand
  :param b: second operan
  :param dt: mpi datatype
  
  """
  if a > 0:
    binlen_a = -(len(bin(a)) -2 ) # bin(2) == 0b10
  else:
    binlen_a = a


  if b > 0:
    binlen_b = -(len(bin(a)) -2 )
  else:
    binlen_b = b

  res = min(binlen_a,binlen_b)
  print("%d/%d executing `%r` op `%r` -> `%r`" % (rank,size,a,b,res))
  
  return res 

op_max = MPI.Op.Create(reduce_max, commute=True)
op_binmax = MPI.Op.Create(reduce_binmax, commute=True)


#data = comm.allreduce(data,MPI.SUM)
#data = comm.allreduce(data,op_binmax)

data = comm.allreduce(data,op_max)

comm.barrier()

print("%03d/%03d data after allreduce: %s" % (rank,size,data))