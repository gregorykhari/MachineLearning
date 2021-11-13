import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

class ScratchLinearRegressor:

    def __init__(self,epochs=100,learning_rate=0.0001):
        self.epochs = epochs
        self.learning_rate = learning_rate

    def normalize(self,x):
        return np.array(list(map(f, x)))

    def f(self,x):
        MIN, MAX = np.amin(x), np.amax(x)
        return (x - MIN)/(MAX - MIN)

    def fit(self, x_train, y_train):

        for col in x_train.T:
            np.array(list(map(self.f, col)))

        #initialize weight vector to all zeros
        self.w = np.zeros(x_train.shape[1])

        def calculate_gradient(x_train, y_train):
            y_prediction = np.dot(x_train,self.w)
            y_error = (y_prediction - y_train) 
            gradient = np.dot(x_train.T,y_error)
            return gradient
        
        for iters in range(0,self.epochs):
            self.w = self.w - self.learning_rate * calculate_gradient(x_train,y_train)

    def predict(self, X):
        return np.dot(X, self.w)

if __name__ == '__main__':
    boston = load_boston()
    boston_dataset = pd.DataFrame(boston.data,columns=boston.feature_names)
    boston_dataset['MEDV'] = boston.target

    #separate X and Y
    X = boston_dataset.to_numpy()

    #remove MEDV column from X
    X = np.delete(X,13,1)
    Y = boston_dataset['MEDV'].to_numpy()

    #split data into training and test data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 5)
    regressor = ScratchLinearRegressor()
    regressor.fit(X_train, Y_train)

    for m in range(0,X_test.shape[0]):
        prediction = regressor.predict(X_test[m])
