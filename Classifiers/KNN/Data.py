import numpy as np
import matplotlib.pyplot as plt
import UserDict

class ClusteredDataModel(object):
	def __init__(self, center, std_dev=1):
		self.center = center
		self.std_dev = std_dev
		self.dim = len(center)

	def generateData(self,length):
		data = self.std_dev*np.random.randn(self.dim,length) + self.center[:,np.newaxis]
		return data

class ClusteredDataset(UserDict.IterableUserDict):
	def __init__(self, labels, dim=2, centers=None, std_devs=None, lengths=None):
		self.data = {}

		# create distributions
		if centers is not None:
			assert len(labels) == centers.shape[0] and centers.shape[1] == dim
		else:
			centers = np.random.rand(len(labels),dim)*4
			print centers
		if std_devs is not None:
			assert len(labels) == std_devs.shape[0]
		else:
			std_devs = np.random.rand(len(labels))
		if lengths is not None:
			assert len(labels) == lengths.shape[0]
		else:
			lengths = np.random.multinomial(1000,[1/float(len(labels))]*len(labels))

		for label, center, std_dev, length in zip(labels, centers, std_devs, lengths):
			self.data[label] = ClusteredDataModel(center, std_dev=std_dev).generateData(length)

		# print self.data

if __name__ == '__main__':
	# create the dataset
	labels = ['red','green','blue']
	dataset = ClusteredDataset(labels)

	# plot the data
	for i, label in enumerate(labels):
		data = dataset.data[label]
		x, y = data[0,:], data[1,:]
		plt.plot(x,y,'.')
	plt.show()
