import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import sklearn.neural_network
import sklearn.model_selection
import sklearn.metrics
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import metrics

from functions import *
from test3 import *


np.random.seed(2019)

iterations = 100
eta = 0.1
lmbd = 0.0
sigmoid = Sigmoid()
tanh = Tanh()
relu = Relu()
softmax = SoftMax()
MSE_loss = MeanSquareError()
crossentropy = CrossEntropy()

datapoints = 20
n_points = 5
x = np.linspace(0,1,datapoints)
y = np.linspace(0,1,datapoints)

x,y  = np.meshgrid(x,y)
z= FrankeFunction(x,y)

x= np.ravel(x)
y= np.ravel(y)
z = np.ravel(z)

#creates franke functions data
X= CreateFranke_data(x,y, n_points)
#split the data into test and train
X_train, X_test, z_train, z_test = sklearn.model_selection.train_test_split(X, z, test_size = 0.2)

#perform neural network on franke function
nn = NeuralNet(X_train, z_train, neutron_Length= [16,8], n_categories= 1, activations= sigmoid, output_activations= sigmoid, 
		 epochs = 100, eta= 0.001, lmbd =0.0)

#train the model
nn.train(iterations)
#predict the model
z_pred = nn.predict_regression_franke(X_test)

learning_rates = np.logspace(-5,-1,7)
lambda_values = np.logspace(-5,-1,7)

#perform grid search
NN_GRID = nn.GridSearch_regression(X_test, X_train, z_test, z_train, lambda_values, learning_rates, iterations=200)


#Confusion_Matrix(z_test, z_pred)















