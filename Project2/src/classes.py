import numpy as numpy
import pandas as pd
#import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from scipy.optimize import fmin_tnc

import numpy as np


class Sigmoid():
	def __call__(self,x):
		return 1./(1 + np.exp(-x))

	def derivative(self, x):
		return np.exp(-x)/(1+ np.exp(-x))**2

class Tanh():
	def __call__(self,x):
		return np.tanh(x)

	def derivative(self, x):
		return 1 + np.tanh(x)**2

class Relu():
	def __call__(self,x):
		index = np.where(x<0)
		x[index] = 0.
		t=x
		return t 

	def derivative(self, x):
		index1 = np.where(x > 0)
		index2 = np.where(x <= 0)
		x[index1] = 1.
		x[index2] = 0.

		return x

class SoftMax():
	def __call__(self,x):
		exp_term = np.exp(x)

		return exp_term / np.sum(exp_term, axis=1, keepdims=True)

	def derivative(self, x):
		return self(x)* (1- self(x))

class normal():
	def __call__(self,x):
		return x

	def derivative(self, x):
		return 1

class MeanSquareError():
	def __call__(self, yPred, y):
		return np.mean((yPred -y)**2)

	def derivative(self, yPred, y):
		return yPred-y


class CrossEntropy():
	def __call__(self, yPred, y):
		return -np.mean(y* np.log(yPred) + (1-y)*np.log(1-yPred))

	def derivative(self, yPred, y):
		return (yPred -y)/(yPred* (1-yPred))





































