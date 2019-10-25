from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np 
from imageio import imread
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from random import random, seed
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, KFold
from sklearn.utils import resample
from numpy.random import randint, randn
from sklearn.pipeline import make_pipeline


class Ridge:
	def __init__(self,alpha):
		self.alpha = alpha
		self.beta = None

	def fit(self,X,y):
		p = np.eye(X.shape[1])
		self.beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X) + self.alpha*p), X.T),y)
		#print(self.beta)

	def predict(self,X):
		return X @ self.beta

	def get_params(self, deep= True):
		return {'alpha': self.alpha}


def MSE(z_test, z_pred):
	mse = np.mean((z_test-z_pred)**2)
	return mse


def R2(z_test, z_pred):
	R2 = 1 - np.mean((z_test-z_pred) ** 2) / np.mean((z_test - np.mean(z_test)) ** 2)
	return R2

def VarianceCalc(z_pred):
	#np.mean(np.var(zPred))
	return np.mean((z_pred - np.mean(z_pred))**2 )

def BiasCalc(z_test, z_pred):
	return np.mean((z_test - np.mean(z_pred))**2)

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**(2)) - 0.5*((9*y-2)**(2)))
	term2 = 0.75*np.exp(-((9*x+1)**(2))/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**(2) - (9*y-7)**(2))
	term4 = -0.2*np.exp(-(9*x-4)**(2) - (9*y-7)**(2))

	return term1+ term2+ term3+ term4

def ConfidenceInterval(beta):
	#sigma = np.var(beta)
	sigma =np.sqrt(np.diag(np.linalg.inv(X.T @ X)))

	betahigh = beta + 1.96*sigma
	betalow = beta - 1.96*sigma

	diff = betahigh - betalow

	plt.scatter(range(len(beta)),beta)
	plt.errorbar(range(len(beta)), beta, yerr =np.array(diff),capsize=3, lw=1, fmt='r')
	plt.xlabel('$\\beta$ Range')
	plt.ylabel('$\\beta$')
	plt.legend(['Beta', 'Confidence Interval'])
	plt.title('Ordinary least square with degree 5, noise 0.05 and 100 datapoints')
	#plt.savefig('OLSErrorbarNoise.png')

	#plt.show()
	
	return betalow, betahigh


def Create_DesignMatrix(x,y,degree):
	n = x.shape[0]
	if degree ==1:
		x = np.c_[np.ones(n), x, y]


	elif degree == 2:
		x= np.c_[np.ones(n), x, y, x**2, x*y, y**2]

	elif degree == 3:
		x= np.c_[np.ones(n), x,y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3]

	elif degree == 4:
		x= np.c_[np.ones(n), x,y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3, x**4, x**3*y, x**2*y**2, x*y**3, y**4]

	elif degree == 5:
		x= np.c_[np.ones(n), x,y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3, x**4, x**3*y, x**2*y**2, x*y**3, y**4, \
		x**5, x**4*y, x**3*y**2, x**2*y**3, x*y**4, y**5]

	else:
		raise ValueError('Degree not less than 6! :{}'.format(degree))


	#print(x.shape)
	return x


def BiasVarianceTradeoff(x,y,z, nBoots, degrees, maxdegree, model):

	np.random.seed(2018)

	
	error = np.zeros(maxdegree)
	bias = np.zeros(maxdegree)
	variance = np.zeros(maxdegree)
	trainingerror = np.zeros(maxdegree)
	testerror = np.zeros(maxdegree)
	polydegree = np.arange(maxdegree)

	
	error_bvt =[]
	bias_bvt= []
	variance_bvt = []
	print(maxdegree)

	trials = 100


	for degree in range(1,maxdegree+1):
	
		X = Create_DesignMatrix(x, y, degree)
		X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.2)
		#model = make_pipeline(PolynomialFeatures(degree=degree),LinearRegression(fit_intercept=False))
		zPred = np.empty((z_test.shape[0], nBoots))
		ztilde = np.empty((z_test.shape[0], nBoots))


		for i in range(nBoots):
			X_new, z_new = resample(X_train, z_train)
			zPred[:,i] = model.fit(X_new, z_new).predict(X_test).ravel()
			#print(X_train.shape, z_new.shape, X_new.shape)
			#ztilde[:,i] = model.fit(X_new, z_new).predict(X_train).ravel()


		z_test = z_test.reshape(len(z_test),1)
		error[degree-1] = np.mean( np.mean((z_test- zPred)**2, axis=1, keepdims=True) )
		#error1[degree-1] = np.mean( np.mean((z_test- ztilde)**2, axis=1, keepdims=True) )
		bias[degree-1] = np.mean( (z_test - np.mean(zPred, axis=1, keepdims=True))**2 )
		variance[degree-1] = np.mean((zPred - np.mean(zPred, axis =1, keepdims = True))**2 )
		#variance[degree -1] = np.mean(np.var(zPred, axis=1, keepdims=True))
		
		print('{} >= {} + {} = {}'.format(error, bias, variance, bias+variance))
		

	
	plt.plot(polydegree, error, "*", label='Error')
	plt.plot(polydegree, bias, label='Bias')
	plt.plot(polydegree, variance, label='Variance')
	plt.xlabel('Degree')
	plt.ylabel('Values')
	plt.title('Model complexity of OLS method with degree 5')
	#plt.title('Model complexity of Ridge Regression for Terrain data \n with $\lambda = 1e-7$ and degree 5')
	plt.legend()
	#plt.savefig('BiasVarianceTradeOff_OLS_Terrain.png')
	plt.show()



	return error_bvt, bias_bvt, variance_bvt



def CrossValidation(x,y,z,k, degrees, model, nlambdas):

	np.random.seed(2018)

	kfold = KFold(n_splits=k, shuffle = True, random_state= 5)
	
	
	
	beta_values_cv = []

	average_R2_train = []
	average_MSE_train = []
	average_R2_test = []
	average_MSE_test = []


	KFold_scores =np.zeros((nlambdas,k))

	
	i= 0
	for deg in degrees:

		j=0
		

		R2_values_test = []
		R2_values_train = []

		MSE_values_train = []
		MSE_values_test = []

		Variance_values_test =[]
		Variance_values_train = []

		Bias_values_test =[]
		Bias_values_train = []

		splits = kfold.split(z)
		
		for train_inds, test_inds in splits:
			
			xtrain = x[train_inds]
			ytrain = y[train_inds]
			ztrain = z[train_inds]		

			xtest = x[test_inds]
			ytest = y[test_inds]
			ztest = z[test_inds]


			Xtrain_cv = Create_DesignMatrix(xtrain, ytrain, deg)
			Xtest_cv = Create_DesignMatrix(xtest, ytest, deg)

			
			z_pred_cv_COEF = model.fit(Xtrain_cv, ztrain)
			z_pred_cv = model.predict(Xtest_cv)

			R2_values_test.append(R2(ztest,z_pred_cv))
			MSE_values_test.append(MSE(ztest, z_pred_cv))
			Variance_values_test.append(VarianceCalc(z_pred_cv))
			Bias_values_test.append(BiasCalc(ztest, z_pred_cv))

			z_pred_train_coef = model.fit(Xtrain_cv, ztrain)
			z_pred_train = model.predict(Xtrain_cv)

			R2_values_train.append(R2(ztrain,z_pred_train))
			MSE_values_train.append(MSE(ztrain, z_pred_train))
			Variance_values_train.append(VarianceCalc(z_pred_train))
			Bias_values_train.append(BiasCalc(ztest, z_pred_train))

			#print(R2_values_train)
			#print(R2_values_test)


			j+= 1
		i+=1 
		estimated_mse_Kfold = np.mean(KFold_scores, axis = 1)

		"""
		print(R2_values_train)
		print('----')
		print(R2_values_test)
		print('-----')
		print(MSE_values_train)
		print('-----')
		print(MSE_values_test)
		print('-----')
		print(Variance_values_train)
		print('-----')
		print(Variance_values_test)
		print('-----')
		print(Bias_values_train)
		print('-----')
		print(Bias_values_test)
		"""


		average_R2_train.append(np.mean(R2_values_train))
		average_MSE_train.append(np.mean(MSE_values_train))

		average_R2_test.append(np.mean(R2_values_test))
		average_MSE_test.append(np.mean(MSE_values_test))
		
		print(average_R2_test)
		print('-----')
		print(average_R2_train)	
		print('-----')
		print(average_MSE_test)
		print('-----')
		print(average_MSE_train)
		

		plt.plot(degrees, np.array(R2_values_test), label ='R2 Test data')
		plt.plot(degrees, np.array(R2_values_train), label='R2 Train data')
		#plt.plot(degrees, np.array(MSE_values_test), label ='MSE Test data')
		#plt.plot(degrees, np.array(MSE_values_train), label='MSE Train data')
		#plt.plot(degrees, np.array(Variance_values_test), label ='Variance Test data')
		#plt.plot(degrees, np.array(Variance_values_train), label='Variance Train data')
		#plt.plot(degrees, np.array(Bias_values_test), '*',label ='Bias Test data')
		#plt.plot(degrees, np.array(Bias_values_train), label='Bias Train data')
		plt.xlabel('Degree')
		plt.ylabel('R2 Values')
		plt.title('R2 as a function of polynomial degree %s for Terrain data. \n Lasso regression with $\lambda = 1e-7$' %deg)
		plt.legend()
		#plt.savefig('CVR2_Lasso_Terrain.png')
		plt.legend()
		#plt.show()	
		



	return average_MSE_test, average_R2_test, average_MSE_train, average_R2_train




#load the Terrain data
terrain1 = imread('SRTM_data_Norway_1.tif')

"""
#Show the Terrain
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain1,cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
"""

#print(terrain1.shape)
# Since the shape is (3601, 1801) we define x and y as

#x = np.arange(0, 1801)
#y = np.arange(0, 3601)


row, col = np.shape(terrain1)
x = np.linspace(0, 1,col)
y= np.linspace(0,1,row)




x,y = np.meshgrid(x,y)

"""
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(x,y,terrain1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
"""

degree = 5

z= terrain1

x= np.ravel(x[:100,:100])
y= np.ravel(y[:100,:100])
z = np.ravel(z[:100,:100])
X = Create_DesignMatrix(x, y, degree)


X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.2, shuffle= True)
normaldistributionX = 1/X_train.std(axis=0) #normalizing
print(normaldistributionX.shape)
normaldistributionZ = 1/z_train.std() #normalizing



normaldistributionX[0] = 1


EX = np.mean(X_train,axis=0)
EX[0]= 0
EZ = np.mean(z_train,axis=0)

#print(X_train.shape)
X_train= (X_train- EX)*normaldistributionX
#print(X_train.shape)
z_train= (z_train -EZ)*normaldistributionZ
#print(X_test.shape)
X_test=(X_test - EX)*normaldistributionX


z_test=(z_test-EZ)*normaldistributionZ


"""
beta_OLS = (np.linalg.inv(X_train.T @ X_train)@ X_train.T @ z_train)
z_predict_ols = X_test @ beta_OLS
z_tilde = X_train @ beta_OLS


print('Training MSE: {}'.format(MSE(z_train, z_tilde)))
print('Test MSE: {}'.format(MSE(z_test, z_predict_ols)))

print('Training R2: {}'.format(R2(z_train, z_tilde)))
print('Test R2: {}'.format(R2(z_test, z_predict_ols)))

print('Training Variance: {}'.format(np.var(z_train)))
print('Test Variance: {}'.format(np.var(z_predict_ols)))

print('Confidence Interval: {}'.format(ConfidenceInterval(beta_OLS)))


R_lambda = 0.01

p = np.eye(X.shape[1])
beta_R = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X) + R_lambda*p), X.T),z)	
z_predict_ridreg = X_test.dot(beta_R)
z_tilde = X_train @ beta_R

lassoreg =Lasso(alpha= 1e-4)
lassoreg.fit(X_train,z_train)
beta_lassoreg = lassoreg.coef_.T
z_predict_lassoreg = lassoreg.predict(X_test)
z_tilde = lassoreg.predict(X_train)

"""

"""
k= 5
degrees = np.arange(1,6)

#model = LinearRegression(fit_intercept= False) #OLS
model = Ridge(alpha= 1e-7)
#model = Lasso(precompute = True, alpha= 1e-7)
nlambdas = 500

lambdas = np.logspace(-10,2, nlambdas) 
average_MSE_test, average_R2_test, average_MSE_train, average_R2_train = CrossValidation(x,y,z,k, degrees, model, nlambdas)

"""

maxdegree = 5
nBoots = 100
degrees = np.arange(1,6)

model = LinearRegression(fit_intercept= False)
#model = Ridge(alpha= 1e-7)
#model = Lasso(precompute = True, alpha= 1e-7)

err, bi, var = BiasVarianceTradeoff(x,y,z, nBoots, degrees, maxdegree, model) # husk aa ikke ta med noise her


