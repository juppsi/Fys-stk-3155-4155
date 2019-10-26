from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np 
from random import random, seed
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, KFold
from sklearn.utils import resample
from numpy.random import randint, randn
from sklearn.pipeline import make_pipeline
from random import randrange
from sklearn.metrics import r2_score, mean_squared_error
from imageio import imread


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



def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**(2)) - 0.5*((9*y-2)**(2)))
	term2 = 0.75*np.exp(-((9*x+1)**(2))/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**(2) - (9*y-7)**(2))
	term4 = -0.2*np.exp(-(9*x-4)**(2) - (9*y-7)**(2))

	return term1+ term2+ term3+ term4



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

	trials = 100


	for degree in range(1,maxdegree+1):
	
		X = Create_DesignMatrix(x, y, degree)
		X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.2)
		zPred = np.empty((z_test.shape[0], nBoots))
		ztilde = np.empty((z_test.shape[0], nBoots))


		for i in range(nBoots):
			X_new, z_new = resample(X_train, z_train)
			#model.fit(X_new, z_new)
			zPred[:,i] = model.fit(X_new, z_new).predict(X_test).ravel()
			#print(X_train.shape, z_new.shape, X_new.shape)
			#ztilde[:,i] = model.fit(X_new, z_new).predict(X_train).ravel()



		z_test = z_test.reshape(len(z_test),1)
		error[degree-1] = np.mean( np.mean((z_test- zPred)**2, axis=1, keepdims=True) )
		#error1[degree-1] = np.mean( np.mean((z_test- ztilde)**2, axis=1, keepdims=True) )
		bias[degree-1] = np.mean( (z_test - np.mean(zPred, axis=1, keepdims=True))**2 )
		variance[degree-1] = np.mean((zPred - np.mean(zPred, axis =1, keepdims = True))**2 )
		#variance[degree -1] = np.mean(np.var(zPred, axis=1, keepdims=True))
		
		#print('{} >= {} + {} = {}'.format(error, bias, variance, bias+variance))

		error_bvt.append(np.mean(error))
		bias_bvt.append(np.mean(bias))
		variance_bvt.append(np.mean(variance))

		

	
	plt.plot(polydegree, error, label='Error')
	plt.plot(polydegree, bias, label='Bias')
	plt.plot(polydegree, variance, label='Variance')
	plt.xlabel('Degree')
	plt.ylabel('Values')
	#plt.title('Model complexity of OLS method with degree 5')
	plt.title('Model complexity of Lasso Regression with $\lambda = 1e-7$ and degree 5')
	plt.legend()
	#plt.savefig('BiasVarianceTradeOff_Lasso.png')
	plt.show()
	



		

	return error_bvt, bias_bvt, variance_bvt



def CrossValidation(x,y,z,k, degrees, model, nlambdas):

	np.random.seed(2018)

	kfold = KFold(n_splits=k, shuffle = True, random_state= 5)
	
	
	
	beta_values_cv = []

	average_R2_train = []
	average_MSE_train = []
	average_bias_train = []
	average_variance_train = []
	average_R2_test = []
	average_MSE_test = []
	average_bias_test = []
	average_variance_test = []


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


			#beta_OLS = np.linalg.inv(Xtrain_cv.T @ (Xtrain_cv))@ Xtrain_cv.T @ ztrain
			#z_pred_cv = Xtest_cv @ beta_OLS
			#z_cv = Xtrain_cv @ beta_OLS
			
			z_pred_cv_COEF = model.fit(Xtrain_cv, ztrain)
			z_pred_cv = model.predict(Xtest_cv)

			R2_values_test.append(R2(ztest,z_pred_cv))
			MSE_values_test.append(MSE(ztest, z_pred_cv))
			

			z_pred_train_coef = model.fit(Xtrain_cv, ztrain)
			z_pred_train = model.predict(Xtrain_cv)

			R2_values_train.append(R2(ztrain,z_pred_train))
			MSE_values_train.append(MSE(ztrain, z_pred_train))
			

			#print(R2_values_train)
			#print(R2_values_test)



			#model.fit(xtrain, ztrain)
			#zPred[:,i] = model.predict(xtest)

			j+= 1
		i+=1 
		

		#print('Mean KFold scores using sklearn: {}'.format(np.mean(KFold_scores)))
		

		average_R2_train.append(np.mean(R2_values_train))
		average_MSE_train.append(np.mean(MSE_values_train))
	

		average_R2_test.append(np.mean(R2_values_test))
		average_MSE_test.append(np.mean(MSE_values_test))
		

		
		"""
		print(average_R2_test)
		print('-----')
		print(average_R2_train)	
		print('-----')
		print(average_MSE_test)
		print('-----')
		print(average_MSE_train)
		
		"""

	
		#plt.plot(degrees, np.array(R2_values_test), label ='R2 Test data')
		#plt.plot(degrees, np.array(R2_values_train), label='R2 Train data')
		
		plt.plot(degrees, np.array(MSE_values_test), label ='MSE Test data')
		plt.plot(degrees, np.array(MSE_values_train), label='MSE Train data')
		
		
		plt.xlabel('Degree')
		plt.ylabel('MSE Values')
		plt.title('MSE as a function of polynomial degree %s \n Ridge Regression with $\lambda = 1e-7$' %deg)
		#plt.title('MSE as a function of polynomial degree %s \n OLS method' %deg)
		plt.legend()
		#plt.savefig('CVMSE_OLS.png')
		plt.show()	
		



	return average_MSE_test, average_R2_test, average_MSE_train, average_R2_train



"""
z = np.reshape(z,(iterations, iterations))
fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(x,y,z,cmap=cm.coolwarm,linewidth=0,antialiased = False)

ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
"""
# defining x,y, z

datapoints = 100
degree = 5


x = np.linspace(0,1,datapoints)
y = np.linspace(0,1,datapoints)


x,y  = np.meshgrid(x,y)
noise = 0.05*np.random.rand(len(x))
z= FrankeFunction(x,y)

x= np.ravel(x)
y= np.ravel(y)
z = np.ravel(z)

k= 5 #kfold


#CROSSVALIDATION

degrees = np.arange(1,6)


nlambdas = 500
lambdas = np.logspace(-10,2, nlambdas) 
#model = LinearRegression(fit_intercept= False) #OLS
model = Ridge(alpha= 1e-7) #Ridge 
#model = Lasso(precompute = True, alpha= 1e-7) #Lasso
average_MSE_test, average_R2_test, average_MSE_train, average_R2_train= CrossValidation(x,y,z, k, degrees, model, nlambdas) 




#BIASVARIANCETRADEOFF

#datapoints = 10
maxdegree = 5
nBoots = 100
degrees = np.arange(1,6)

#model = LinearRegression(fit_intercept= False) #OLS
#model = Ridge(alpha= 1e-7) #Ridge 
model = Lasso(precompute = True, alpha= 1e-7) #Lasso

err, bi, var = BiasVarianceTradeoff(x,y,z, nBoots, degrees, maxdegree, model) # husk aa ikke ta med noise her

































