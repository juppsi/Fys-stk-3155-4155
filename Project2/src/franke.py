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

from functions import *


np.random.seed(2019)

datapoints = 100
n_points = 5
x = np.linspace(0,1,datapoints)
y = np.linspace(0,1,datapoints)

x,y  = np.meshgrid(x,y)
z= FrankeFunction(x,y)

my_col = cm.jet(z/np.amax(z))

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(x,y,z, facecolors = my_col)
ax.set_title("Franke's function")
ax.set_ylabel("$y$")
ax.set_xlabel("$x$")
ax.set_zlabel("$z$")
plt.show()

x_= np.ravel(x).reshape(-1,1)
y_= np.ravel(y).reshape(-1,1)
z_ = np.ravel(z).reshape(-1,1)

X= CreateFranke_data(x_,y_, n_points)
X_train, X_test, z_train, z_test = sklearn.model_selection.train_test_split(X, z_, test_size = 0.2)

#Regression analysis

#OLS
z_tilde, z_predict_ols = OLS(X_test, X_train, z_train)

print('Training MSE: {}'.format(MSE(z_train, z_tilde)))
print('Test MSE: {}'.format(MSE(z_test,z_predict_ols)))

print('Training R2: {}'.format(R2(z_train,z_tilde)))
print('Test R2: {}'.format(R2(z_test,z_predict_ols)))


#Implementing Neutral Network
regression = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(100,20), 
	learning_rate= 'adaptive', learning_rate_init = 0.01, max_iter= 2000, tol =1e-7, verbose = True)

regression = regression.fit(X_train,z_train)

#Statistical computations
predict = regression.predict(X_test)
MSE =sklearn.metrics.mean_squared_error(z_test, predict)
R2_Score = regression.score(X_test, z_test)
print('MSE ={}'.format(MSE))
print('R2= {}'.format(R2_Score))


predict = regression.predict(X)
z_pred = predict.reshape(datapoints, datapoints)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(x,y,z,facecolors = my_col)
ax.set_title("Franke's function")
ax.set_ylabel("$y$")
ax.set_xlabel("$x$")
ax.set_zlabel("$z$")
ax.plot_wireframe(x,y,z_pred)

fig=plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(x,y,np.abs(z-z_pred), facecolors = my_col)
ax.set_title("Franke's function")
ax.set_ylabel("$y$")
ax.set_xlabel("$x$")
ax.set_zlabel("$z$")
plt.show()









