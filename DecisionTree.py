import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder


class Node:
	def __init__(self):
		self.left = None #branch for if the attribute is yes
		self.right = None #branch for if the attribute is no
		self.feature_index = None
		self.prediction = None
		self.entropy = None

class ScratchDecisionTree:
	def __init__(self,max_depth = 7):
		self.decision_tree = None
		self.max_depth = max_depth
		self.tree_depth = 0
		self.num_leaf_nodes = 0

	#builds the decision tree
	def fit(self,X_train,Y_train):
		self.decision_tree = self.branch(X_train,Y_train,0)

	#predicts the output given a new data vector X
	def predict(self,X):
		current_node = self.decision_tree
		while(True):
			feature_index = current_node.feature_index
			if current_node.left or current_node.right:
				if X[feature_index] == 1:
					current_node = current_node.left
				elif X[feature_index] == 0:
					current_node = current_node.right
			else:
				return current_node.prediction

	#returns the prediction of a leaf node
	def leaf_prediction(self,Y_train):
		return pd.value_counts(Y_train).idxmax()

	#creates a brannch of the tree
	def branch(self,X_train,Y_train,depth):
		node = Node()
		node.entropy = self.calculate_entropy(Y_train) 

		if self.tree_depth < depth:
			self.tree_depth = depth

		if node.entropy == 0 or depth > self.max_depth:
			node.prediction = self.leaf_prediction(Y_train)
			self.num_leaf_nodes = self.num_leaf_nodes + 1 
			return node

		best_information_gain_index = 0
		best_information_gain = 0
		#for each feature we calculate the information gain
		for feature_index in range(0,X_train.shape[1]):
			information_gain = self.calculate_information_gain(X_train,Y_train,node.entropy,feature_index)
			if information_gain > best_information_gain:
				best_information_gain = information_gain
				best_information_gain_index = feature_index
		node.feature_index = best_information_gain_index

		x_left, y_left, x_right, y_right = self.split_data(X_train,Y_train,node.feature_index)

		node.left = self.branch(x_left,y_left,depth+1)
		node.right = self.branch(x_right,y_right,depth+1,)
		
		return node

	#splits the data on index
	def split_data(self,x,y,index):
		x_col  = x[...,index]
		x_left = x[np.where(x_col == 1)]
		y_left = y[np.where(x_col == 1)]
		x_right = x[np.where(x_col == 0)]
		y_right = y[np.where(x_col == 0)]
		return x_left, y_left, x_right, y_right

	#calculates the entropy of a node
	def calculate_entropy(self,Y_train):
		entropy = 0
		for y in np.unique(Y_train):
			total_samples = Y_train.shape[0]
			total_samples_with_class_y = len(np.where(Y_train==y)[0])
			entropy = entropy - ((total_samples_with_class_y/total_samples) * math.log(total_samples_with_class_y/total_samples,2))
		return entropy

	#calculates the information gain if tree is split on attribute in feature_index
	def calculate_information_gain(self,X_train,Y_train,entropy,feature_index):
		x_col = X_train[...,feature_index]
		information_gain = 0
		#for each feature in X_train
		for a in np.unique(x_col):	
			total_samples_with_attr_a = len(np.where(x_col == a)[0])
			total_samples = Y_train.shape[0]
			Y_train_a = Y_train[np.where(x_col == a)] #retrieve the values of Y_train where X == current attribute within feature
			attribute_entropy = self.calculate_entropy(Y_train_a)
			information_gain = information_gain + (total_samples_with_attr_a/total_samples) * attribute_entropy
		return entropy - information_gain

	#prints a visual of the tree
	def visualize_tree(self):
		root = self.decision_tree
		depth = 0
		def recurse_visual(node,depth):
			indent1 = ""
			for i in range(0,depth):
				indent1 = indent1 + "\t"

			print('\n' + indent1 + '|---------------------> Feature Index: {}'.format(node.feature_index) + '\n' + indent1 +'\t\t\tEntropy: {}'.format(node.entropy))

			if node.left or node.right:
				recurse_visual(node.left,depth+1)
				recurse_visual(node.right,depth+1)
			else:
				print(indent1 + '\t\t\tPrediction: {}'.format(node.prediction))

		recurse_visual(root,depth)

	#returns the depth of the tree
	def get_depth(self):
		return self.tree_depth

	#returns the number of leaf nodes within the tree
	def get_n_leaves(self):
		return self.num_leaf_nodes

if __name__ == '__main__':

	voting_records = pd.read_csv('house-votes-84.data',index_col=False)
    
	#separate X and Y
	X = voting_records.to_numpy()

	#remove Class-Name column from X
	X = np.delete(X,0,1)
	Y = voting_records['Class-Name'].to_numpy()

	transformer = ColumnTransformer(transformers=[("Convert",OneHotEncoder(),[x for x in range(0,X.shape[1])])], remainder='passthrough')
	X = transformer.fit_transform(X)

	#split data into training and test data
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 5)


	dtc = ScratchDecisionTree()
	dtc.fit(X_train,Y_train)

	#dtc.visualize_tree()

	correct = 0
	for m in range(0,X_test.shape[0]):
		prediction = dtc.predict(X_test[m])
		if prediction == Y_test[m]:
			correct = correct + 1

	print('Accuracy: {}'.format(correct/X_test.shape[0]))
	print('Number of Leaves: {}'.format(dtc.get_n_leaves()))
	print('Depth: {}'.format(dtc.get_depth()))
