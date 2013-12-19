from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from Data import ClusteredDataset

def GaussianProb(data,mean,variance):
	return 1/(np.sqrt(2*np.pi*variance))*\
	np.exp(-(data-mean)**2/(2*variance**2))

class NaiveBayesClassifier(object):
	def __init__(self,dataset):
		self.dataset = dataset
		self.labels = dataset.keys()
		self.params = []
		for label in self.labels:
			data = dataset[label]
			self.params.append([(np.mean(feature), np.var(feature)) for feature in data])

	def classify(self,data):
		prob = np.ones(len(self.params))
		for label, params in enumerate(self.params):
			# print label, params
			for mean, variance in params:
				for p in GaussianProb(data,mean,variance):
					prob[label] *= p
		return self.labels[np.argmax(prob)]



if __name__ == '__main__':
	# create the dataset
	labels = ['blue','green','red']
	dataset = ClusteredDataset(labels)

	# plot the original data
	for i, label in enumerate(labels):
		data = dataset.data[label]
		x, y = data[0,:], data[1,:]
		plt.plot(x,y,'.')
	plt.title('Original Data')
	plt.savefig('Original Data')

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

	# classify the test data
	NB = NaiveBayesClassifier(dataset)
	classifiedData = {}
	for label in labels:
		classifiedData[label] = []

	for data in testData.T:
		classifiedData[NB.classify(data)].append(data)

	# plot the classified data
	plt.figure()
	for key,data in classifiedData.iteritems():
		if len(data):
			print "before: %s" % data
			data = np.array(data)
			print "after %s" % data
			x, y = data[:,0], data[:,1]
		else:
			x,y = np.array([]), np.array([])
		plt.plot(x,y,'.')
	plt.title('Naive-Bayes Classified Data')
	plt.savefig('Naive-Bayes Classified Data')

	plt.show()