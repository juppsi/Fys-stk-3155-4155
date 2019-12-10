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
"""
r2_score = metrics.r2_score(z_test,z_pred)
MSE = metrics.mean_squared_error(z_test,z_pred)
#accuracy = metrics.accuracy_score(z_test, z_pred)

print('eta =', eta)
print('lmbd =', lmbd)
#print('accuracy =', accuracy)
print('MSE =', MSE)
print('R2 SCORE=', r2_score)

eta_vals = np.logspace(-5,1,7)
lmdb_vals = np.logspace(-5,1,7)

NN_GRID = nn.GridSearch_regression(X_test, X_train, z_test, z_train, lmbd_vals= np.logspace(-5,1,7), eta_vals= np.logspace(-5,1,7), iterations=100)

best_mse, best_lmbd, best_eta = NN_GRID.output_values()

print('best MSE:', best_mse)
print('lambda:', best_lmbd)
print('Learning rate:', best_eta)
"""

"""

#OLS METHOD

beta_OLS = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(z_train)
z_tilde_ols = X_train @ beta
z_pred_ols = X_test @ beta

print("MSE_test_ols:", MSE(z_test, z_pred_ols))
print("MSE_train_ols:", MSE(z_train, z_pred_ols))
print("R2_test_ols:", R2(z_train, z_pred_ols))
print("R2_train_ols:", R2(z_train, z_pred_ols))

Confusion_Matrix(z_test, z_pred_ols)
"""


"""
num_splits = 5
z= np.expand_dims(z, axis=1)

#cv_regression(num_splits, X, z, nn, epochs=100, eta=1e-5, lmbd=0.0, iterations =100)

learning_rates = np.logspace(-6,1,8)
lambda_values = np.logspace(-6,1,8)
epochs_ = 10

mse = np.zeros((len(learning_rates), len(lambda_values)))
for i, eta in enumerate(learning_rates):
	for j, lmbd in enumerate(lambda_values):
		cv_mse = cv_regression(num_splits, X, z, nn, epochs=10, eta=1e-5, lmbd=0.0, iterations =100)

		mse[i,j] = cv_mse

sns.heatmap(pd.DataFrame(mse),  annot= True, fmt='g')
plt.title('CV Frankes Function')
#plt.ylim(top = 0, bottom = learning_rates)
plt.ylabel('Learning rate: $\\eta$')
plt.xlabel('Regularization Term: $\\lambda$')
plt.show()

"""











