#!/usr/bin/env python
"""
Reference naive reference kmeans implementation
"""

import matplotlib
matplotlib.use('Agg')

import time
import numpy as np
import argparse
from pprint import pprint

from matplotlib import pyplot as plt

# data loading


def assign_data_to_cluster(data, clusters_loc, nclusters):
    z = np.zeros((data.shape[0], nclusters))

    for i in range(nclusters):
        z[:, i] = np.linalg.norm(data - clusters_loc[i], axis=1)

    return z.argmin(axis=1)


def perform_naive_kmeans(data, nclusters, iters=4):
    clusters_loc = np.random.randint(0, data.shape[0], nclusters)

    clusters_loc = data[clusters_loc, :]
    print("Initial cluster location")
    pprint(clusters_loc)

    start = time.monotonic()

    for i in range(iters):
        assigned = assign_data_to_cluster(data, clusters_loc, nclusters)
        clusters_loc_old = clusters_loc.copy()
        # reassign centeroids
        for j in range(nclusters):
            mean = data[assigned == j].mean(axis=0)
            if not np.any(np.isnan(mean)):
                # gradient descent
                #clusters_loc[j] = clusters_loc[j]*0.7 + 0.3*mean
                # just mean
                clusters_loc[j] = mean

        if (np.sum(clusters_loc_old - clusters_loc) == 0):
            print("No changes detected - breaking")
            break
    stop = time.monotonic()

    fig = plt.figure()
    plt.title("kmeans-ref.py (time %.3f s, %d points) " % ((stop - start),len(data)))
    ax = plt.subplot(111)
    from scipy.spatial import Voronoi, voronoi_plot_2d
    vor = Voronoi(clusters_loc)
    voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False)

    assigned = assign_data_to_cluster(data, clusters_loc, nclusters)
    ax.scatter(data[:, 0], data[:, 1], c=assigned, s=50)

    plt.scatter(clusters_loc[:,0],clusters_loc[:,1],color='red')

    plt.xlim(data[:,0].min(), data[:,0].max())
    plt.ylim(data[:,1].min(), data[:,1].max())

    plt.savefig("kmeans-ref.png",dpi=300)


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


    args = parser.parse_args()

    data = np.lib.format.open_memmap(args.data_file)

    np.random.seed(args.seed)
    perform_naive_kmeans(data, args.clusters, 1000)


if __name__ == '__main__':
    main()
