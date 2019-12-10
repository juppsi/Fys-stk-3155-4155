import numpy as np
import pandas as pd
#import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from scipy.optimize import fmin_tnc
from sklearn.neural_network import MLPClassifier

from classes import * 
from test3 import *
from functions import *



def logreg():
	"""
	Running logistic regression on credit card data set.
	"""
	#reading credit card data 
	X, y= read_dataset()

	#split the data into test and train
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 123)

	#LogisticRegression_sklearn(X_train, X_test, y_train, y_test)
	#Defining learning rate and regularization parameter
	learning_rates = np.logspace(-6,1,8)
	lambda_values = np.logspace(-6,1,8)

	iterations = 10

	#performing logistic regression on credit card data
	LogisticRegression_self_test(X_train, X_test, y_train, y_test, learning_rates, epochs, iterations)


	print(df)


def run_NN():
	"""
	Running neural network on credit card data set
	"""
	X, y = read_dataset()

	#defining parameters
	n_inputs = X.shape[0]
	n_features = X.shape[1]

	iterations = 20
	eta = 0.1
	lmbd = 0.0
	
	sigmoid = Sigmoid()
	tanh = Tanh()
	relu = Relu()
	softmax = SoftMax()
	MSE_loss = MeanSquareError()
	crossentropy = CrossEntropy()

	#split the data into test and train
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

	
	onehotencoder = OneHotEncoder(categories ='auto', sparse =False)
	y_train_one = onehotencoder.fit_transform(y_train)
	sc = StandardScaler()
	#X_train= sc.fit_transform(X_train)
	#X_test = sc.fit_transform(X_test)

	#perform neural network on the credit card data set
	nn = NeuralNet(X_train, y_train, neutron_Length= [16,8], n_categories= 1, activations= sigmoid, output_activations= sigmoid, 
		 epochs = 100, eta= 1e-4, lmbd =0.00)

	#train the model
	nn.train(iterations)
	test_predict = nn.predict(X_test)


	learning_rates = np.logspace(-5,1,7)
	lambda_values = np.logspace(-5,1,7)
	

	#gridsearch of the model
	NN_GRID = nn.GridSearch(X_test, X_train, y_test, y_train, lambda_values, learning_rates, iterations=200)
	




logreg()
run_NN()




























