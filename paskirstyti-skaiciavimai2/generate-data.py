#!/usr/bin/env python
"""
Result generator
"""
import argparse
from sklearn.datasets import make_blobs
import numpy as np

parser = argparse.ArgumentParser(description='Kmeans initial data generator',formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

parser.add_argument('-n', type=int,default=100000,
                    help='number of samples',)
parser.add_argument('--clusters', type=int,default=16,
                    help='number of cluster',)
parser.add_argument('--cluster_std', type=float,default=0.5,
                    help='number of cluster',)


parser.add_argument('--data-file', type=str,required=True,
                    help='data file',default='kmeans-data.npy')

parser.add_argument('--data-ytrue', type=str,required=True,
                    help='number of cluster',default='kmeans-ytrue.npy')

parser.add_argument('--random-state', type=int,
                    help='random seed',default=0)


args = parser.parse_args()

print("Executing kmeans data generator")
print(args)


X, y_true = make_blobs(n_samples=args.n, centers=args.clusters,
                       cluster_std=args.cluster_std, random_state=0)



print("Saving data to %s and y_true to %s" % (args.data_file,args.data_ytrue))
np.save(args.data_file,X)
np.save(args.data_ytrue,y_true)

