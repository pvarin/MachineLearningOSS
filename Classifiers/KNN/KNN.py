import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as scidist
from Data import FullDataSet

# a class that represents a dataset for the KNN algorithm
# an instance of this class is able to classify new data
class KNNData:
	def __init__(self, k, dataset, dataSize = None):
		self.k = k
		self.dataset = dataset
		if dataSize == None:
			dataSize = [100]*len(dataset)

	def classify(self, point):
		# initialize the list and the furthest recorded distance
		kNearest = []

		for label,data in self.dataset.iteritems():
			print label, data
			distances = scidist.cdist(point, data.generateData(), 'euclidean')
			print label, distances

		return distances

		# # iterate through each of the points in the dataset
		# for d in self.data
		# 	dist = distance(d, data) #TODO make this function call work
		# 	if dist < furthest:
		# 		# remove the element and append the new data
		# 		if len(k_nearest) >= self.k
		# 			kNearest.pop()
		# 			kNearest.append((dist, d))
		# 		# resort the list and update furthest
		# 		kNearest.sort()
		# 		furthest = kNearest[-1][0]

		# #classify the data based on kNearest
		# labels = dict()
		# for d in kNearest:
		# 	labels[d.label] = labels.get(d.label, 0)+1 #TODO make d.label work
		
		# return max(labels.iteritems(), key=operator.itemgetter(1))[0]




if __name__ == '__main__':
	# create the dataset
	labels = ['red','green','blue']
	dataset = FullDataSet(labels)

	d = KNNData(1,dataset)
	d.classify(np.array([[1,1]]))
	#TODO import data
	#add data to KNNData object