import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as scidist
import operator

import sys
sys.path.append("..")
from Data import ClusteredDataset


def FuzzyKNN(newData, dataset, k=2):
	# extract the labels
	labels = dataset.keys()
	
	# track the closest points efficiently
	closest = [(np.inf,labels[0])]*k
	# iterate through the labels
	for i in xrange(len(labels)):
		distances = scidist.cdist(np.transpose(newData[:,np.newaxis]), np.transpose(dataset[labels[i]]))
		for m in xrange(k):
			# calculate the minimum distance
			j = np.argmin(distances)
			d = distances[0,j]
			if d<closest[-1][0]:
				closest[-1] = (d,labels[i])
				closest.sort()
				distances[0,j] = float(np.inf)
			else:
				break
	# count the number of occurances for each label
	numLabels = dict(zip(labels,[0]*len(labels)))
	
	# compute and return the probabilities
	for _,label in closest:
		numLabels[label] += 1./k
	return numLabels
	

if __name__ =='__main__':
	# create the dataset
	labels = ['blue','green','red']
	dataset = ClusteredDataset(labels)

	# plot the original data
	for i, label in enumerate(labels):
		data = dataset.data[label]
		x, y = data[0,:], data[1,:]
		plt.plot(x,y,'.')
	plt.title('OriginalData')
	# plt.savefig('OriginalData')

	#determine the max/min of the dataset
	maxX = -np.inf
	maxY = -np.inf
	minX = np.inf
	minY = np.inf

	for data in dataset.values():
		maxX_t, maxY_t = np.amax(data,axis=1)
		minX_t, minY_t = np.amin(data,axis=1)
		maxX = max(maxX, maxX_t)
		maxY = max(maxY, maxY_t)
		minX = min(minX, minX_t)
		minY = min(minY, minY_t)

	# create test data
	x = np.linspace(minX,maxX)
	y = np.linspace(minY,maxY)
	X, Y = np.meshgrid(x,y)

	prob = dict(zip(labels,[np.zeros(X.shape) for i in xrange(len(labels))]))
	Ks = [10]#[1,3,5,10]
	for k in Ks:
		# classify the test data
		for i in xrange(X.shape[0]):
			for j in xrange(X.shape[1]):
				temp = FuzzyKNN(np.array([X[i,j],Y[i,j]]),dataset,k)
				for label in labels:
					prob[label][i,j] = temp[label]

	for label in labels:
		plt.figure()
		plt.pcolor(x,y,prob[label])
		plt.colorbar()
		plt.title('Probability of Label %s' % (label,))
		# plt.savefig('Probability of Label %s' % (label,))
	plt.show()