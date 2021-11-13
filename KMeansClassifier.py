import math
import sys
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

class KMeansClassifier():

	def __init__(self,k):
		self.k = k
		self.centroids = np.zeros(self.k)
		self.max_iters = 100
		self.training_clusters = None

	def fit(self,x_train):
		centroid_indexes = [random.randrange(0,x_train.shape[0]) for i in range(0, self.k)]
		self.centroids = x_train[centroid_indexes]
		self.training_clusters = np.zeros(X.shape[0])

		iters = 0
		while(True):
			#calculate the closest cluster and assign
			for i in range(0,x_train.shape[0]):
				closest_centroid_distance = sys.maxsize
				closest_cluster = sys.maxsize
				for j in range(0,self.k):
					distance = np.linalg.norm(self.centroids[j] - x_train[i])
					if (distance < closest_centroid_distance):
						closest_centroid_distance = distance
						closest_cluster = j
				self.training_clusters[i] = closest_cluster

			#update cluster means
			new_centroids = []
			for i in range(0,self.k):
				cluster_points = x_train[np.where(self.training_clusters == i)]
				new_centroids.append(cluster_points.mean(axis=0))

			#check if any cluster centroids have changed since last iteration
			if np.array_equal(np.array(new_centroids),self.centroids):
				break
			else:
				self.centroids = np.array(new_centroids)

			iters += 1

			#check if the maximum number of iterations has been exceeded
			if iters > self.max_iters:
				break

		print(self.training_clusters)

	def assign(self,x):
		#calculate the closest cluster and assign x to that cluster
		closest_centroid_distance = sys.maxsize
		closest_cluster = sys.maxsize
		for j in range(0,self.k):
			distance = np.linalg.norm(self.centroids[j] - x)
			if (distance < closest_centroid_distance):
				closest_centroid_distance = distance
				closest_cluster = j
		return closest_cluster


if __name__ == '__main__':
	digits_datasets = load_digits()
	digits_df = pd.DataFrame(digits_datasets.data,columns=digits_datasets.feature_names)

	X = digits_df.to_numpy()

	kmc = KMeansClassifier(10)
	kmc.fit(X)
	#print(kmc.assign(X[1]))
	#print(kmc.assign(X[11]))
