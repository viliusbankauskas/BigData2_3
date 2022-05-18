#!/usr/bin/env python

from matplotlib import pyplot as plt;
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np

clusters = 16

X = np.load("kmeans-data.npy")
y_true = np.load("kmeans-ytrue.npy")

fig = plt.figure()
ax = plt.subplot(111)



kmeans = KMeans(n_clusters=clusters, random_state=0).fit(X)


vor = Voronoi(kmeans.cluster_centers_)

print(vor)


voronoi_plot_2d(vor,ax=ax,show_points=False,show_vertices=False)


ax.scatter(X[:, 0], X[:, 1], c=kmeans.predict(X),s=50);

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='red')
plt.show()

