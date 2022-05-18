#!/usr/bin/env python
"""
Calculates pagerank
"""
from mpi4py import MPI
import sys, datetime, os,time
import lzma,glob, pickle,sys

import scipy.sparse

from pprint import pprint
import re
import numpy as np

import itertools


comm = MPI.COMM_WORLD

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()


if rank == 0:
  with open("wiki-graph.pickle",'rb') as f:
    graph = pickle.load(f)

  with open("lookup-articles.pickle",'rb') as f:
    lookup = pickle.load(f)

  N = len(lookup['forward'])
  N = ( (N + size - 1) // size) * size


  PR = np.ones(N)/N

  ranges_lower = np.array(range(0,N,N//size))
  ranges_higher = ranges_lower + (N//size)

else:
  N = None
  ranges_lower = None
  ranges_higher = None
  PR = None
  graph = None


N = comm.bcast(N, root=0)
PR = comm.bcast(PR, root = 0)
graph = comm.bcast(graph, root=0)

d = 0.85
const1 = (1 - d)/N


if rank == 0:
  print("start articles:",ranges_lower)
  print("end articles:  ",ranges_higher,'not inclusive')
  print("**********")

comm.barrier()

# divide ranges
ranges_lower = comm.scatter(ranges_lower,root=0)
ranges_higher = comm.scatter(ranges_higher,root=0)

# print some info
print("%d/%d (%s) node prossing [%d-%d) %d articles" % (rank, size, name,ranges_lower, ranges_higher, ranges_higher - ranges_lower))
comm.barrier()

for num_iter in range(32):

  PR = comm.allgather(PR[ranges_lower:ranges_higher])
  PR = list(itertools.chain(*PR))

  if num_iter > 0 and rank == 0: # calculate matrix difference
    a = np.array(PR)
    b = np.array(PR_old)
    print("@ %d iter sum(PR - PR_old) =" % (num_iter - 1),np.sum(a - b))
  PR_old = PR.copy()

  for i in range(ranges_lower, ranges_higher):
    s = 0
    if i in graph['b']:
      for j in graph['b'][i]:
        s = s + ( PR_old[j]/len(graph['f'][j]))
    else:
      s = s + (1/N)
    PR[i] = const1 + d * s


  if rank == 0:
    print("Iter %d finished" % num_iter)


# gathering data

PR = comm.gather(PR[ranges_lower:ranges_higher],root=0)

if  rank == 0: # calculate matrix difference
    PR = list(itertools.chain(*PR))

    a = np.array(PR)
    b = np.array(PR_old)
    print("@ %d iter sum(PR - PR_old) =" % num_iter,np.sum(a - b))

    print("saving PR data")

    with open("page-rank-calculated.pickle",'wb') as f:
        pickle.dump(PR,f)
