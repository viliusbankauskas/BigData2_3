#!/usr/bin/env python
"""
Reference kmeans implementation using MPI interface naive implementation
"""
import matplotlib
matplotlib.use('Agg')


import numpy as np
import argparse
from pprint import pprint

from matplotlib import pyplot as plt

from mpi4py import MPI

import time,sys

# load mpi

comm = MPI.COMM_WORLD

name = MPI.Get_processor_name()
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()




def assign_data_to_cluster(data, clusters_loc, nclusters):
    z = np.zeros((data.shape[0], nclusters))

    for i in range(nclusters):
        z[:, i] = np.linalg.norm(data - clusters_loc[i], axis=1)

    return z.argmin(axis=1)

def calc_split(anum, n):
    """
    calculate splits into [a_lower, a_higher) intervals
    """
    k, m = divmod(anum, n)
    a_lower  = [ i*k+min(i,m) for i in range(n) ]
    a_higher = [(i+1)*k+min(i+1,m)  for i in range(n)]

    return a_lower,a_higher

def perform_mpi1_kmeans(data, nclusters, iters=4):
    global rank,size,comm

    if rank == 0:
        clusters_loc = np.random.randint(0, data.shape[0], nclusters)
        clusters_loc = data[clusters_loc, :] + np.random.uniform(-1,1,(nclusters,2))
        print("Initial cluster location")
        pprint(clusters_loc)
        
        a_lower, a_higher = calc_split(len(data),size)
        
        print("tasks parts lower: ",a_lower)
        print("tasks parts higher:",tuple(a_higher))
    else:
      a_lower = None
      a_higher = None
      clusters_loc = None
      clusters_loc_old = None

    start = time.monotonic()

    a_lower  = comm.scatter(a_lower,root=0)
    a_higher = comm.scatter(a_higher,root=0)

    clusters_loc = comm.bcast(clusters_loc,root=0)

    for i in range(iters):
        # do job
        assigned = assign_data_to_cluster(data[a_lower:a_higher], clusters_loc, nclusters)

        assigned = comm.gather(assigned,root=0) # gather all parts
        if rank == 0:
          assigned = np.concatenate(assigned).flatten() # technically we are gathering objects (binary blobs, not arrays !)

        if rank == 0:
          clusters_loc_old = clusters_loc.copy()
          # reassign centeroids
          for j in range(nclusters):
            mean = data[assigned == j].mean(axis=0)
            if not np.any(np.isnan(mean)):
                # gradient descent
                #clusters_loc[j] = clusters_loc[j]*0.7 + 0.3*mean
                # just mean
                clusters_loc[j] = mean

        clusters_loc_old = comm.bcast(clusters_loc_old,root=0)
        clusters_loc = comm.bcast(clusters_loc,root=0)

        # not optimal, but straightforward
        if (np.sum(clusters_loc_old - clusters_loc) == 0):
            print("No changes detected - breaking")
            break
    stop = time.monotonic()
    
    print("%s %d/%d took %.3f s" % (name,rank,size,(stop - start)))
    if rank ==0:
      print((stop - start),file=sys.stderr)
    sys.stdout.flush()
    comm.barrier()
    if rank == 0:
      fig = plt.figure()
      ax = plt.subplot(111)
      plt.title("kmeans-mpi-1.py (time %.3f s) " % ((stop - start)))

      from scipy.spatial import Voronoi, voronoi_plot_2d
      vor = Voronoi(clusters_loc)
      voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False)

      assigned = assign_data_to_cluster(data, clusters_loc, nclusters)
      ax.scatter(data[:, 0], data[:, 1], c=assigned, s=50)

      plt.scatter(clusters_loc[:,0],clusters_loc[:,1],color='red')

      plt.xlim(data[:,0].min(), data[:,0].max())
      plt.ylim(data[:,1].min(), data[:,1].max())

      plt.savefig("kmeans-mpi-1.png",dpi=300)


def main():
    parser = argparse.ArgumentParser(
        description='Kmean naive solver',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--data-file', type=str,
                        help='data file', default='kmeans-data.npy')

    parser.add_argument('--data-ytrue', type=str,
                        help='number of cluster', default='kmeans-ytrue.npy')

    parser.add_argument('--clusters', type=int,
                        help='number of clusters', default=16)

    parser.add_argument('--seed',type=int,
                        help='random seed', default=42)

    parser.add_argument('--max-iters',type=int,
                        help='random batch size', default=1000)

    args = parser.parse_args()

    data = np.lib.format.open_memmap(args.data_file)

    np.random.seed(args.seed)
    perform_mpi1_kmeans(data, args.clusters, args.max_iters)


if __name__ == '__main__':
    main()
