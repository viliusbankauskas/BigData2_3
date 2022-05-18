#!/usr/bin/env python
"""
Reference kmeans implementation using MPI interface using minibatch with averanging
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

def perform_mpi2_kmeans(data, nclusters, batch_seed,batch_size =512, iters=4,/, stop_sad = 0.02):
    global rank,size,comm

    if rank == 0:
        clusters_loc = np.random.randint(0, data.shape[0], nclusters)
        clusters_loc = data[clusters_loc, :] + np.random.uniform(-1,1,(nclusters,2))
        print("Initial cluster location")
        pprint(clusters_loc)

    else:
      clusters_loc = None
      clusters_loc_old = None

    start = time.monotonic()

    clusters_loc = comm.bcast(clusters_loc,root=0)
    rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(batch_seed)))    # synchornized seed

    for i in range(iters):
      permatations = rng.choice(data.shape[0],batch_size * size)
      permatations = np.array(permatations[rank::size]) # simple method to cut the necessary slice

      dperm = data[permatations]

      clusters_loc_old = clusters_loc.copy()

      assigned = assign_data_to_cluster(dperm, clusters_loc,nclusters)

      for j in range(nclusters):
            mean_data = dperm[assigned == j]
            if mean_data.shape[0] > 0:
              mean = mean_data.mean(axis=0)
            if not np.any(np.isnan(mean)) and mean_data.shape[0] > 0:
                miu = 1.0 / (( mean_data.shape[0] + 1.0))
                clusters_loc[j] = (1 - miu)*clusters_loc[j]  + miu*mean

      clusters_loc = comm.allreduce(clusters_loc,MPI.SUM) / size # get average centroids

      sm = np.sum(np.abs(clusters_loc_old - clusters_loc))
      if ( sm < stop_sad):  # stop condition
        if rank == 0:
          print("meeting stopping condition (absolute coordinate difference between all clusters) diff=%f stop_condition=%f" % (sm,stop_sad))
        break

    #print(clusters_loc)
    stop = time.monotonic()

    print("%s %d/%d took %.3f s" % (name,rank,size,(stop - start)))
    if rank ==0:
      print((stop - start),file=sys.stderr)
    sys.stdout.flush()
    comm.barrier()
    if rank == 0:
      fig = plt.figure()
      ax = plt.subplot(111)
      plt.title("kmeans-mpi-2.py (time %.3f s) " % ((stop - start)))

      from scipy.spatial import Voronoi, voronoi_plot_2d
      vor = Voronoi(clusters_loc)
      voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False)

      assigned = assign_data_to_cluster(data, clusters_loc, nclusters)
      ax.scatter(data[:, 0], data[:, 1], c=assigned, s=50)

      plt.xlim(data[:,0].min(), data[:,0].max())
      plt.ylim(data[:,1].min(), data[:,1].max())

      plt.scatter(clusters_loc[:,0],clusters_loc[:,1],color='red')

      plt.savefig("kmeans-mpi-2.png",dpi=300)


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

    parser.add_argument('--batch-seed',type=int,
                        help='random batch seed', default=43)

    parser.add_argument('--batch-size',type=int,
                        help='random batch size', default=512)

    parser.add_argument('--stop-sad',type=float,
                        help='random batch size', default=0.05)

    parser.add_argument('--max-iters',type=int,
                        help='random batch size', default=1000)

    args = parser.parse_args()

    data = np.lib.format.open_memmap(args.data_file)

    np.random.seed(args.seed)
    perform_mpi2_kmeans(data, args.clusters, args.batch_seed, args.batch_size, args.max_iters, stop_sad = args.stop_sad)


if __name__ == '__main__':
    main()
