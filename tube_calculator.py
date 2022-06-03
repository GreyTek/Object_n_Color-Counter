import numpy as np
from skimage import io, morphology, measure
from sklearn.cluster import KMeans

img = io.imread('stuffed_pool1.png')
rowPixels, colPixels, bands = img.shape
MainCounter = img.reshape(rowPixels * colPixels, bands)

kmeans = KMeans(n_clusters=5, random_state=0).fit(MainCounter)
labels = kmeans.labels_.reshape(rowPixels, colPixels)

for i in np.unique(labels):
    Opening_Labels = np.int_(morphology.binary_opening(labels == i))
    color = np.around(kmeans.cluster_centers_[i])
    counter = len(np.unique(measure.label(Opening_Labels))) - 1
    print('Color: {}  >>  Objects: {}'.format(color, counter))
