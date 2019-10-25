from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np 
from random import random, seed
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, KFold
from sklearn.utils import resample
from numpy.random import randint, randn
from sklearn.pipeline import make_pipeline
from random import randrange

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**(2)) - 0.5*((9*y-2)**(2)))
	term2 = 0.75*np.exp(-((9*x+1)**(2))/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**(2) - (9*y-7)**(2))
	term4 = -0.2*np.exp(-(9*x-4)**(2) - (9*y-7)**(2))

	return term1+ term2+ term3+ term4


def Create_DesignMatrix(x,y,n):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = x**(i-k) * y**k

	return X

def MSE(z_test, z_pred):
	mse = np.sum((z_test-z_pred)**2)/(np.size(z_pred))
	return mse


def R2(z_test, z_pred):
	R2 = 1 - np.sum((z_test-z_pred) ** 2) / np.sum((z_test - np.mean(z_pred)) ** 2)
	return R2

def BiasCalc(z_test, z_pred):
	return np.mean((z_test - np.mean(z_pred))**2)

def ConfidenceInterval(beta):
	#sigma = np.var(beta)
	sigma =np.sqrt(np.diag(np.linalg.inv(X.T @ X)))

	betahigh = beta + 1.96*(sigma)
	betalow = beta - 1.96*(sigma)

	diff = betahigh - betalow

	plt.scatter(range(len(beta)),beta)
	plt.errorbar(range(len(beta)), beta, yerr =np.array(diff),capsize=3, lw=1, fmt='r')
	plt.xlabel('$\\beta$ Range')
	plt.ylabel('$\\beta$')
	plt.legend(['Beta', 'Confidence Interval'])
	plt.title('Lasso regression with degree 5, noise 0.05, \n lambda 1e-7 and 100 datapoints')
	#plt.savefig('lassoErrorbar1e-7.png')
	#for i in range(len(beta)):
		#plt.errorbar(i, y= beta[i], yerr= 1.96*sigma[i])
	plt.show()
	
	return betalow, betahigh

datapoints = 100
degree = 5

x = np.linspace(0,1,datapoints)
y = np.linspace(0,1,datapoints)


x,y  = np.meshgrid(x,y)
noise = 0.05*np.random.rand(len(x))
z= FrankeFunction(x,y) + noise


x= np.ravel(x)
y= np.ravel(y)
z = np.ravel(z)
X = Create_DesignMatrix(x, y, degree)

X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.2)


lassoreg =Lasso(alpha= 1e-7)
lassoreg.fit(X_train,z_train)
beta_lassoreg = lassoreg.coef_.T
z_predict_lassoreg = lassoreg.predict(X_test)
z_tilde = lassoreg.predict(X_train)

print('Training MSE: {}'.format(MSE(z_train, z_tilde)))
print('Test MSE: {}'.format(MSE(z_test, z_predict_lassoreg)))

print('Training R2: {}'.format(R2(z_train, z_tilde)))
print('Test R2: {}'.format(R2(z_test, z_predict_lassoreg)))

print('Training Variance: {}'.format(np.var(z_train)))
print('Test Variance: {}'.format(np.var(z_predict_lassoreg)))

print('Confidence Interval: {}'.format(ConfidenceInterval(beta_lassoreg)))


