import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, mean_squared_error, r2_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from scipy.optimize import fmin_tnc
import numpy as np 
from scikitplot.metrics import plot_confusion_matrix, plot_roc, plot_cumulative_gain
from scikitplot.helpers import cumulative_gain_curve

from classes import * 
from functions import *

"""

A neural network class used to perform with different kind of data sets. 
There are several functions in this class feed-forward, backpropagation, grid-search, create-bias-and-weights, predict that represents
different parts of the neural network performance.

"""

class NeuralNet:

	def __init__(self, 
			X_data, 
			y_data, 
			neutron_Length, 
			n_categories, 
			activations,
			output_activations, 
			epochs,
			eta,
			lmbd):


		self.X_data_full = X_data
		self.y_data_full = y_data

		self.activations = activations
		self.output_activations = output_activations
		self.n_categories = n_categories

		self.n_hidden_layers = len(neutron_Length)
		self.n_inputs = X_data.shape[0]
		self.n_features= X_data.shape[1]

		self.final_iterations = 0
		self.n_hidden_neuron = []
		
		for layer in neutron_Length:
			self.n_hidden_neuron.append(layer)

		self.n_hidden_layers = len(self.n_hidden_neuron)

		self.a = np.empty(self.n_hidden_layers + 2, dtype=np.ndarray)
		#self.a = X_data
		self.w = np.empty(self.n_hidden_layers + 1, dtype=np.ndarray)
		self.b = np.empty(self.n_hidden_layers + 1, dtype=np.ndarray)
		self.z = np.empty(self.n_hidden_layers + 2, dtype=np.ndarray)


		self.epochs = epochs
		self.eta = eta
		self.lmbd = lmbd

		self.best_accuracy = -1
		self.best_eta = 0
		self.best_lmbd = 0
		self.best_mse = 1e8

		
		#creates the structure of the activations, weights and bias
		self.create_biases_and_weights()

	def create_biases_and_weights(self):
	
	
		self.a[0] =self.X_data_full #Input layer
		x_n, x_c =self.X_data_full.shape

		#first weigth of hidden layer 
		self.w[0]= np.random.randn(x_c, self.n_hidden_neuron[0])*np.sqrt(2/(self.n_hidden_neuron[0] + x_c)) # input of first hidden layer weights

		#Hidden layers
		for i in range(self.n_hidden_layers):
			self.b[i] = np.zeros(self.n_hidden_neuron[i])
			self.a[i+1] = np.zeros(self.n_hidden_neuron[i])
			self.z[i +1] =np.zeros(self.n_hidden_neuron[i])

		# The weight of hidden layer
		for i in range(1, self.n_hidden_layers):
			self.w[i] = np.random.randn(self.n_hidden_neuron[i-1], self.n_hidden_neuron[i])*np.sqrt(2/(self.n_hidden_neuron[i-1] + self.n_hidden_neuron[i]))

		self.b[-1] =np.zeros(self.n_categories)
		self.w[-1] = np.random.randn(self.n_hidden_neuron[-1], self.n_categories)*np.sqrt(2/(self.n_hidden_neuron[-1] + self.n_categories))
		self.a[-1] = np.zeros(self.n_categories) #output layer
		self.z[-1] =np.zeros(self.n_categories)

	def feed_forward(self):
		"""
		The activation in the hidden layer is by taking the weighted input of the sigmoid function.
		The first activation in neural network is input data. 
		"""

		#iteration of hidden layers
		for i in range(1, self.n_hidden_layers +1):	
			self.z[i] = np.matmul(self.a[i-1], self.w[i-1]) + self.b[i-1]
			
			self.a[i] = self.activations(self.z[i])
			

		#Output layer
		self.z[-1] = np.matmul(self.a[-2], self.w[-1]) + self.b[-1]

		self.a_o = self.output_activations(self.z[-1]) #output
		self.a[-1] = self.a_o
		#print(self.a_o)

		return self.a_o

		#a_o is probabilities

	def gradients(self, error, i):
		"""
		Gradients for the output and hidden layer
		"""
		b_gradient = np.sum(error, axis = 0)
		
		w_gradient = np.matmul(self.a[i].T, error)

		if self.lmbd > 0.0:
			w_gradient += self.lmbd * self.w[i]

		return w_gradient, b_gradient


	def backpropagation(self):
		
		error_layer = self.a_o -self.y_data_full.reshape(self.a_o.shape) #many

		
		#output layer
		w_gradient, b_gradient = self.gradients(error_layer, -1)
		
		self.w[-1] -= self.eta * w_gradient
		self.b[-1] -= self.eta * b_gradient

		error_old = error_layer
		
		# from last hidden later to first hidden layer
		for i in range(self.n_hidden_layers, 0, -1):
			#previous error to propagate the error back to the first hidden layer
			
			error_hidden = np.matmul(error_old, self.w[i].T) * self.activations.derivative(self.a[i])

			w_gradient, b_gradient = self.gradients(error_hidden, i-1)

			#optimize weights and biases
			self.w[i-1] -= self.eta * w_gradient
			self.b[i-1] -= self.eta * b_gradient
			error_old = error_hidden

		#self.final_iterations +=1
		#print(self.total_iterations, self.iterations, total)
	
	
	def predict(self, X_test):
		#Predict for multiple neurons

		self.a[0] = X_test
		self.feed_forward()

		#y_Pred = np.argmax(self.a_o, axis=1)
		y_Pred = np.round(self.a_o)

		return y_Pred

	def predict_regression_franke(self, X_test):
		#Predict for franke regression

		self.a[0] = X_test
		self.feed_forward()
		y_Pred = self.a_o

		return y_Pred


	def train(self, iterations):
		"""
		Train the feed forward and back propagation for a number of iteration.
		"""
		#self.print_values()

		a=[]
		m = []
		cross_entropy = CrossEntropy()
		cross_entropy = MeanSquareError()
		for i in range(iterations):
			y_pred = self.feed_forward()
			self.backpropagation()

			#print(y_pred[0])
			a.append(cross_entropy(y_pred, self.y_data_full))

		# Checking if the model train
		#plt.plot(range(iterations), a)
		#plt.show()



	def GridSearch(self, X_test, X_train, y_test, y_train, lmbd_vals, eta_vals, iterations):
		"""
		Compares the best learning rate and regularization parameter for multiple-cases.
		Calculate AUC-score and accuracy score.
		"""
	
		train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
		test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
		roc_score_test = np.zeros((len(eta_vals), len(lmbd_vals)))
		roc_score_train = np.zeros((len(eta_vals), len(lmbd_vals)))
		NN_numpy= np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)

		#grid search
		for i in range(len(eta_vals)):
			for j in range(len(lmbd_vals)):
				self.create_biases_and_weights()
				#self.print_values()
				self.train(iterations)

				
				test_pred = self.predict(X_test)
				train_pred = self.predict(X_train)
			

				accuracy = accuracy_score(y_test, test_pred)
			
				#train_accuracy[i][j] = metrics.accuracy_score(y_train,train_pred.round(), normalize=False)
				#test_accuracy[i][j] = metrics.accuracy_score(y_test,test_pred)

				train_accuracy[i][j] = accuracy_score(y_train, train_pred)
				test_accuracy[i,j] = accuracy_score(y_test, test_pred)

				roc_score_test[i,j]= metrics.roc_auc_score(y_test, test_pred)
				roc_score_train[i,j]= metrics.roc_auc_score(y_train, train_pred)


				if accuracy > self.best_accuracy:
					self.best_accuracy = accuracy
					self.best_lmbd = lmbd_vals[j]
					self.best_eta = eta_vals[i]


				#print('Accuracy score on test data:', accuracy)
				print('best accuracy:', self.best_accuracy)
				print('lambda:', self.best_lmbd)
				print('Learning rate:', self.best_eta)
				#print('Train Area ratio:', np.mean(roc_score_train))
				#print('Test Area ratio:', np.mean(roc_score_test))
				

		sns.set()
		sns.heatmap(train_accuracy, annot=True, cmap="viridis", fmt='.4g')
		plt.title("Training Accuracy")
		plt.ylabel('Learning rate: $\\eta$')
		plt.xlabel('Regularization Term: $\\lambda$')
		b, t = plt.ylim() 
		b += 0.5 # Add 0.5 to the bottom
		t -= 0.5 # Subtract 0.5 from the top
		plt.ylim(b, t)
		#plt.savefig('traing_accuracy_cc_nn.png')
		plt.show()
		
	
		sns.heatmap(test_accuracy, annot=True, cmap="viridis", fmt='.4g')
		plt.title("Test Accuracy")
		plt.ylabel('Learning rate: $\\eta$')
		plt.xlabel('Regularization Term: $\\lambda$')
		b, t = plt.ylim() 
		b += 0.5 # Add 0.5 to the bottom
		t -= 0.5 # Subtract 0.5 from the top
		plt.ylim(b, t)
		#plt.savefig('test_accuracy_cc_nn.png')
		plt.show()

		sns.heatmap(roc_score_train, annot=True, cmap="viridis", fmt='.4g')
		plt.title("AUC Train")
		plt.ylabel('Learning rate: $\\eta$')
		plt.xlabel('Regularization Term: $\\lambda$')
		b, t = plt.ylim() 
		b += 0.5 # Add 0.5 to the bottom
		t -= 0.5 # Subtract 0.5 from the top
		plt.ylim(b, t)
		#plt.savefig('traing_auc_cc_nn.png')
		plt.show()
		
	
		sns.heatmap(roc_score_test, annot=True, cmap="viridis", fmt='.4g')
		plt.title("AUC Test")
		plt.ylabel('Learning rate: $\\eta$')
		plt.xlabel('Regularization Term: $\\lambda$')
		b, t = plt.ylim() 
		b += 0.5 # Add 0.5 to the bottom
		t -= 0.5 # Subtract 0.5 from the top
		plt.ylim(b, t)
		#plt.savefig('test_auc_cc_nn.png')

		plt.show()
		
		Confusion_Matrix(y_test, test_pred)

		#test_pred = np.argmax(test_pred, axis=1)

		diff = np.concatenate((1- test_pred, test_pred), axis=1)

		plot_cumulative_gain(y_test, diff)

		plot_roc(y_test, diff, plot_micro=False, plot_macro= False)
		plt.show()

		

	def GridSearch_regression(self, X_test, X_train, y_test, y_train, lmbd_vals, eta_vals, iterations):
		"""
		Compares the best learning rate and regularization parameter for regression-cases.
		Calculate MSE and R2-score.
		"""
		#store the model for heatmaps
		train_mse = np.zeros((len(eta_vals), len(lmbd_vals)))
		test_mse = np.zeros((len(eta_vals), len(lmbd_vals)))

		train_R2 = np.zeros((len(eta_vals), len(lmbd_vals)))
		test_R2= np.zeros((len(eta_vals), len(lmbd_vals)))

		#Grid search
		for i in range(len(eta_vals)):
			for j in range(len(lmbd_vals)):
				self.create_biases_and_weights()
				#self.print_values()
				self.train(iterations)

				test_pred = self.predict_regression_franke(X_test)
				train_pred = self.predict_regression_franke(X_train)
				y_Pred = self.predict_regression_franke(X_test)

				#R2_score = metrics.r2_score(y_test,y_Pred)
				MSE = metrics.mean_squared_error(y_test,y_Pred)

				train_mse[i,j] = metrics.mean_squared_error(y_train, train_pred)
				test_mse[i,j]  = metrics.mean_squared_error(y_test,test_pred)

				train_R2[i,j]  = metrics.r2_score(y_train,train_pred)
				test_R2[i,j]  = metrics.r2_score(y_test,test_pred)

				if MSE < self.best_mse:
					self.best_mse = MSE
					self.best_lmbd = lmbd_vals[j]
					self.best_eta = eta_vals[i]


				print('Best MSE:', self.best_mse)
				print('lambda:', self.best_lmbd)
				print('Learning rate:', self.best_eta)
				#print('R2 score:', R2score)
				#print('MSE:', MSE)



		sns.set()
		sns.heatmap(train_mse, annot=True, cmap="viridis")
		b, t = plt.ylim() 
		b += 0.5 # Add 0.5 to the bottom
		t -= 0.5 # Subtract 0.5 from the top
		plt.ylim(b, t)
		plt.title("Training MSE")
		plt.ylabel('Learning rate: $\\eta$')
		plt.xlabel('Regularization Term: $\\lambda$')
		#plt.savefig('train_mse_franke_nn.png')
		plt.show()

		sns.heatmap(test_mse, annot=True,cmap="viridis")
		b, t = plt.ylim() 
		b += 0.5 # Add 0.5 to the bottom
		t -= 0.5 # Subtract 0.5 from the top
		plt.ylim(b, t)
		plt.title("Test MSE")
		plt.ylabel('Learning rate: $\\eta$')
		plt.xlabel('Regularization Term: $\\lambda$')
		#plt.savefig('test_mse_franke_nn.png')
		plt.show()

		
		sns.heatmap(train_R2, annot=True,  cmap="viridis")
		b, t = plt.ylim() 
		b += 0.5 # Add 0.5 to the bottom
		t -= 0.5 # Subtract 0.5 from the top
		plt.ylim(b, t)
		plt.title("Training R2")
		plt.ylabel('Learning rate: $\\eta$')
		plt.xlabel('Regularization Term: $\\lambda$')
		#plt.savefig('train_r2_franke_nn.png')
		plt.show()

		
		sns.heatmap(test_R2, annot=True, cmap="viridis")
		b, t = plt.ylim() 
		b += 0.5 # Add 0.5 to the bottom
		t -= 0.5 # Subtract 0.5 from the top
		plt.ylim(b, t)
		plt.title("TEST R2")
		plt.ylabel('Learning rate: $\\eta$')
		plt.xlabel('Regularization Term: $\\lambda$')
		#plt.savefig('test_r2_franke_nn.png')
		plt.show()



