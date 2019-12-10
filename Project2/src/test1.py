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
	#Defininf learning rate and regularization parameter
	learning_rates = np.logspace(-6,1,8)
	lambda_values = np.logspace(-6,1,8)

	iterations = 10

	#performing logistic regression on credit card data
	LogisticRegression_self(X_train, X_test, y_train, y_test, learning_rates, lambda_values, iterations)


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

	#print ("eta =", eta)
	#print ("lmbd =", lmbd)
	#print ("accuracy =", accuracy_score(y_test, test_predict))

	#Confusion_Matrix(y_test, test_predict)

	#print(test_predict)


	learning_rates = np.logspace(-5,1,7)
	lambda_values = np.logspace(-5,1,7)
	#lambda_values= [0]

	#gridsearch of the model
	NN_GRID = nn.GridSearch(X_test, X_train, y_test, y_train, lambda_values, learning_rates, iterations=200)
	#NN_GRID= nn.GridSearch(X, y, lambda_values, learning_rates, iterations)

	#best_accuracy, best_lmbd, best_eta = NN_GRID.output_values()

	#print('best accuracy:', best_accuracy)
	#print('lambda:', best_lmbd)
	#print('Learning rate:', best_eta)

	#num_splits = 5

	#cv_classification(num_splits, X, y, nn, epochs = 10, eta= 0.1, lmbd =0.0, iterations=100)

	"""
	
	epochs_ = 10
	batch_size_ =100

	accuracy_scores = np.zeros((len(learning_rates), len(lambda_values)))
	auc_scores = np.zeros((len(learning_rates), len(lambda_values)))

	for i, eta in enumerate(learning_rates):
		for j, lmbd in enumerate(lambda_values):
			cv_auc, cv_accuracy = cv_classification(num_splits, X, z, nn, epochs_, batch_size_, eta, lmbd)

			auc_scores[i,j] = cv_auc
			accuracy_scores[i,j] = cv_accuracy
			#print(auc_scores[i,j])

	sns.heatmap(pd.DataFrame(auc_scores),  annot= True, fmt='g')
	plt.title('CV auc scores')
	plt.ylim(top = 0, bottom = learning_rates)
	plt.ylabel('Learning rate: $\\eta$')
	plt.xlabel('Regularization Term: $\\lambda$')
	#plt.xticks(ticks=np.arange(len(learning_rates)) + 0.5, labels=learning_rates)
	#plt.yticks(ticks=np.arange(len(lambda_values)) + 0.5, labels=lambda_values)
	plt.show()

	sns.heatmap(pd.DataFrame(accuracy_scores),  annot= True, fmt='g')
	plt.title('CV auc scores')
	plt.ylim(top = 0, bottom = learning_rates)
	plt.ylabel('Learning rate: $\\eta$')
	plt.xlabel('Regularization Term: $\\lambda$')
	#plt.xticks(ticks=np.arange(len(learning_rates)) + 0.5, labels=learning_rates)
	#plt.yticks(ticks=np.arange(len(lambda_values)) + 0.5, labels=lambda_values)
	plt.show()

	"""

def scikit_learn_NN():
	"""

	"""

	X, y = read_dataset()


	n_inputs = X.shape[0]
	n_features = X.shape[1]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

	lmbd = 0
	eta = 0.01
	epochs = 100

	#scikit_learnNN = dnn = MLPClassifier(hidden_layer_sizes= int(n_features), activation='logistic',
							#alpha=lmbd, learning_rate_init=eta, max_iter=epochs, random_state=1)
	#scikit_learnNN.fit(X,y[:,0])
	#y_predict = scikit_learnNN.predict(X)
	#MSE =metrics.mean_squared_error(y_test, y_predict)
	#R2_Score = regression.score(y_test, y_predict)
	#print('MSE ={}'.format(MSE))
	#print('R2= {}'.format(R2_Score))

	#Confusion_Matrix(y[:,0], y_predict)

	# store models for later use
	eta_vals = np.logspace(-5, 1, 7)
	lmbd_vals = np.logspace(-5, 1, 7)
	# store the models for later use
	DNN_scikit = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
	train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
	
	for i, eta in enumerate(eta_vals):
		for j, lmbd in enumerate(lmbd_vals):
			dnn = MLPClassifier(hidden_layer_sizes= int(n_features), activation='logistic', 
				alpha=lmbd, learning_rate_init=eta, max_iter=epochs, random_state=1)

			dnn.fit(X_train, y_train)
			DNN_scikit[i][j] = dnn
			train_accuracy[i][j] = dnn.score(X_train, y_train)

			test_predict = dnn.predict(X_test)

			MSE =metrics.mean_squared_error(y_test, test_predict)
			R2_Score = metrics.r2_score(y_test, test_predict)
			print('MSE ={}'.format(MSE))
			print('R2= {}'.format(R2_Score))

	sns.set()
	sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
	ax.set_title("Training Accuracy")
	ax.set_ylabel("$\eta$")
	ax.set_xlabel("$\lambda$")
	plt.show()




#logreg()
run_NN()

#scikit_learn_NN()


























