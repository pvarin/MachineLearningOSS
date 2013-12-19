import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as scidist
import operator

import sys
sys.path.append("..")
from Data import ClusteredDataset


def ClassifyKNN(newData, dataset, k=2):
	# extract the labels
	labels = dataset.keys()
	
	# track the closest points efficiently
	closest = []
	for i in xrange(len(labels)):
		distances = scidist.cdist(np.transpose(newData[:,np.newaxis]), np.transpose(dataset[labels[i]]))
		for l in np.transpose(dataset[labels[i]]):
			j = np.argmin(distances)
			d = distances[0,j]
			if len(closest)<k or d<closest[-1][0]:
				closest.append((d,labels[i]))
				closest.sort()
				distances[0,j] = float(np.inf)
			else:
				break

	# count the number of occurances for each label
	numLabels = {}
	for _,label in closest:
		numLabels[label] = numLabels.get(label,0) + 1
	
	# find the maximum occurance
	return max(numLabels.iteritems(), key=operator.itemgetter(1))[0]
	
if __name__ == '__main__':
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

	X = np.ndarray.flatten(X)
	Y = np.ndarray.flatten(Y)

	testData = np.vstack((X,Y))

	Ks = [1,3,5,10]
	for k in Ks:
		# classify the test data
		classifiedData = {}
		for label in labels:
			classifiedData[label] = []

		for i,data in enumerate(np.transpose(testData)):
			if i%100000:
				print i/float(testData.shape[1])
			classifiedData[ClassifyKNN(data,dataset,k)].append(data)

		# plot the classified data
		plt.figure()
		for key,data in classifiedData.iteritems():
			data = np.array(data)
			x, y = data[:,0], data[:,1]
			print x,y
			plt.plot(x,y,'.')
		plt.title('KNN-Classified Data K=%s' % k)
		# plt.savefig('KNN-Classified Data K=%s' % k)