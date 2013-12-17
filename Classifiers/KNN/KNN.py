# calculates the distance between two hyperdimensional points
def distance(point1, point2):
	s = 0
	for a,b in zip(point1, point2):
		s += (a-b)**2
	return sqrt(s)

# a class that represents a dataset for the KNN algorithm
# an instance of this class is able to classify new data
class KNNData:
	def __init__(self, k, data, labels):
		self.k
		self.data = data

	def classify(data):
		# initialize the list and the furthest recorded distance
		kNearest = []
		furthest = float('inf')

		# iterate through each of the points in the dataset
		for d in self.data
			dist = distance(d, data) #TODO make this function call work
			if dist < furthest:
				# remove the element and append the new data
				if len(k_nearest) >= self.k
					kNearest.pop()
					kNearest.append((dist, d))
				# resort the list and update furthest
				kNearest.sort()
				furthest = kNearest[-1][0]

		#classify the data based on kNearest
		labels = dict()
		for d in kNearest:
			labels[d.label] = labels.get(d.label, 0)+1 #TODO make d.label work
		
		return max(labels.iteritems(), key=operator.itemgetter(1))[0]




if __name__ == '__main__':
	#TODO import data
	#add data to KNNData object