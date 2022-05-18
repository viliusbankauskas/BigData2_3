#!/usr/bin/env python
"""
This program draw npy data
"""
import argparse

import matplotlib
matplotlib.use('Agg')


import numpy as np
import argparse
from pprint import pprint

from matplotlib import pyplot as plt


parser = argparse.ArgumentParser(description='Kmeans initial data generator',formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

parser.add_argument('--data-file', type=str,
                    help='data file',default='kmeans-data.npy')

parser.add_argument('--data-ytrue', type=str,
                    help='number of cluster',default='kmeans-ytrue.npy')


args = parser.parse_args()

data = np.load(args.data_file)

assigned = np.load(args.data_ytrue)

fig = plt.figure()
plt.title("refrence %s " % (args.data_file,))
ax = plt.subplot(111)

ax.scatter(data[:, 0], data[:, 1], c=assigned, s=50)



plt.xlim(data[:,0].min(), data[:,0].max())
plt.ylim(data[:,1].min(), data[:,1].max())

plt.savefig("kmeans-draw.png",dpi=300)


