import numpy as np 
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, auc
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from scipy.optimize import fmin_tnc
from sklearn.model_selection import KFold
from scikitplot.metrics import plot_confusion_matrix, plot_roc, plot_cumulative_gain
from scikitplot.helpers import cumulative_gain_curve


def Confusion_Matrix(y_test, yPred):
	"""
	Plot confusion matrix
	"""
	sns.set()
	confusionMatrix = metrics.confusion_matrix(y_test, yPred)
	sns.heatmap(pd.DataFrame(confusionMatrix), annot=True, fmt='g')
	b, t = plt.ylim() 
	b += 0.5 # Add 0.5 to the bottom
	t -= 0.5 # Subtract 0.5 from the top
	plt.ylim(b, t)
	plt.title('Confustion matrix')
	plt.ylabel('True values')
	plt.xlabel('Predicted values')
	#plt.savefig('confusion_matrix_GD_logreg_cc.png')
	plt.show()

def LogisticRegression_sklearn(X_train, X_test, y_train, y_test):
	"""
	Logistic regrssion performed by sklearns LogisticRegression
	"""

	log_reg = LogisticRegression()
	log_reg.fit(X_train, y_train.ravel())
	yPred =log_reg.predict(X_test)

	#Printing metrics of the logistic regression model
	print('Accuracy:', metrics.accuracy_score(y_test, yPred))
	print('Precision:', metrics.precision_score(y_test, yPred))
	print('Recall', metrics.recall_score(y_test, yPred))

	#confusion matrix

	confusionMatrix = matrix.confusion_matrix(y_test, yPred)
	sb.heatmap(pd.DataFrame(confusionMatrix), annot= True, fmt='g')
	plt.title('Confustion matrix with default value 1')
	plt.ylabel('True values')
	plt.xlabel('Predicted values')
	plt.show()
	

def TotalCost(n,y,p,beta):
	total_cost = -np.sum(y*beta - np.log(1+ np.exp(y*beta)))
	return total_cost

def area_ratio(model_x, model_y):
	area_model = np.trapz(model_x, model_y)

	area_optimal = 0.788 + 0.5*0.2212
	area_baseline = 0.5

	area_ratio = (area_model - area_baseline)/(area_optimal - area_baseline)

	return area_ratio

def Probability(x, beta):
	weighted = x @ beta
	prob = 1./(1 + np.exp(-weighted)) #sigmoid
	pred = np.round(prob) # returns 0 or 1 

	#print(pred)
	return prob, pred

def Probability_GD(x, beta):
	weighted = x @ beta
	prob = 1./(1 + np.exp(-weighted)) #sigmoid
	#pred = np.round(prob) # returns 0 or 1 
	pred = np.argmax(prob,axis=1)

	#print(pred)
	return prob, pred

def Gradient(x,y, beta):
	activation_function = 1./(1+ np.exp(-x @ beta)) #Sigmoid

	return np.dot(x.T, (activation_function -y))


def GradientDescent(x, beta, y, iteration, eta):
	#Gradient descent
	#iters = 100
	#eta = 1e-2


	m = x.shape[1]
	eta0 = eta

	for i in range(iteration):
		gradient = Gradient(x,y,beta)
		beta_new = beta - gradient*eta

		norm = np.linalg.norm(beta_new - beta)

		beta = beta_new

		if (norm < 1e-9):
			return beta, norm

	return beta, norm


def learning_schedule(t, t0= 5, t1=20):
	return t0/(t+t1)


def stochastic_gradient_descent(X, beta, y, eta, epochs, iterations, batch_size=100):
	"""
	Stchastic gradient descent
	"""

	#m=X.shape[0]
	m= len(X)

	#batch_size = 1
	beta =np.random.randn(len(X[0]),1)*0.1

	#data_indices = np.arange(m)    # Samples
	for epoch in range(epochs):

		prob_SGD, predict_SGD= Probability(X, beta) 

		#print(accuracy_score(y, predict_SGD))
		#print(beta)
		#print(epoch)
		for i in range(iterations):
			#print(i)
			random_index = np.random.randint(m - batch_size)

			xi = X[random_index * batch_size: random_index*batch_size + batch_size]
			yi = y[random_index * batch_size: random_index*batch_size + batch_size]
			#yi = yi.reshape((batch_size, 1))

			gradient = Gradient(xi, yi, beta)

			#eta = learning_schedule(epoch *m + i)

			beta -= eta * gradient/batch_size


	return beta





def LogisticRegression_self_test(X_train, X_test, y_train, y_test, learning_rates, epochs,  iteration):

	"""
	Logistic regression with stochastic gradient descent
	"""

	# scoping number of training samples

	n_inputs = X_train.shape[0]
	n_features = X_train.shape[1]

	#model_curve = np.trapz(n_inputs, n_features)
	#optimal_curve = 0.788 + 0.5*0.2212
	#baseline = 0.5
	#area_ratio = (model_curve - baseline)/(optimal_curve - baseline)

	#iteration = 20

	eta_ = 1e-12
	beta_opt = np.random.randn(X_train.shape[1], 2)
	calc_beta_GD, norm = GradientDescent(X_train, beta_opt, y_train, iteration, eta_)
	prob_GD, predict_GD= Probability_GD(X_test, calc_beta_GD) #defining values to be between 0 and 1
	#yPred_GD = (predict_GD >= 0.5).astype(int) # converting to just 0 or 1

	#Define Logistic regression
	clf = LogisticRegression(solver='lbfgs', max_iter=1e5)
	clf = clf.fit(X_train, np.ravel(y_train))
	pred_sklearn = clf.predict(X_test)
	prob_sklearn = clf.predict_proba(X_test)
	#print(prob_sklearn)

	#for eta in np.logspace(np.log10(1e-6), np.log10(1e0), 7):
	accuracy = np.zeros(len(learning_rates))
	auc_score = np.zeros(len(learning_rates))

	for i, eta in enumerate(learning_rates):
		beta_SGD = stochastic_gradient_descent(X_train, beta_opt, y_train, eta, epochs, iteration)
		prob_SGD, predict_SGD= Probability(X_test, beta_SGD) #defining values to be between 0 and 1
		#yPred_SGD = (predict_SGD >= 0.5).astype(int) # converting to just 0 or 1
		#accuracy[i,j] = np.mean(yPred == y_test)
		accuracy[i] = metrics.accuracy_score(y_test, predict_SGD)
		auc_score[i] = metrics.roc_auc_score(y_test, predict_SGD)
		difference = y_test - predict_SGD

		#area_ratio_curve[i,j] = area_ratio

		if i> 0 and auc_score[i] > auc_score[i-1]:
			best_pred_SGD= predict_SGD
			best_prob_SGD = prob_SGD
	

		print('Accuracy {}, learning rate= {}, iterations = {}'.format(accuracy[i], eta, iteration))
		#print('Classification equal 1:', np.sum(difference ==1))
		#print('Classification equal 0:', np.sum(difference ==0))
		#print('Classification equal -1:', np.sum(difference == -1))
		print('Auc score: {}'.format(auc_score[i]))


		"""
		plt.plot(yPred, label='predict')
		plt.plot(optimal_beta, label ='optimal beta')
		plt.plot(y_test, label='test')
		plt.show()
		"""

	sns.set()
	sns.heatmap(pd.DataFrame(accuracy),  annot= True, fmt='.4g')
	plt.title('Grid-search for logistic regression')
	plt.ylabel('Learning rate: $\\eta$')
	plt.xlabel('Regularization Term: $\\lambda$')
	#plt.xticks(ticks=np.arange(len(learning_rates)) + 0.5, labels=learning_rates)
	#plt.yticks(ticks=np.arange(len(lambda_values)) + 0.5, labels=lambda_values)
	b, t = plt.ylim() # discover the values for bottom and top
	b += 0.5 # Add 0.5 to the bottom
	t -= 0.5 # Subtract 0.5 from the top
	plt.ylim(b, t) # update the ylim(bottom, top) values
	#plt.savefig('accuracy_logreg.png')
	plt.show()

	sns.heatmap(pd.DataFrame(auc_score),  annot= True, fmt='.4g')
	plt.title('Grid-search for logistic regression')
	plt.ylabel('Learning rate: $\\eta$')
	plt.xlabel('Regularization Term: $\\lambda$')
	#plt.xticks(ticks=np.arange(len(learning_rates)) + 0.5, labels=learning_rates)
	#plt.yticks(ticks=np.arange(len(lambda_values)) + 0.5, labels=lambda_values)
	b, t = plt.ylim() # discover the values for bottom and top
	b += 0.5 # Add 0.5 to the bottom
	t -= 0.5 # Subtract 0.5 from the top
	plt.ylim(b, t) # update the ylim(bottom, top) values
	#plt.savefig('auc_score_logreg.png')
	plt.show()

	#plot confusion matrix
	Confusion_Matrix(y_test, predict_GD)
	#Confusion_Matrix(y_test, best_pred_SGD)
	#Confusion_Matrix(y_test, pred_sklearn)

	#diff = np.concatenate((1- predict, predict), axis=1)

	diff_sklearn =  np.concatenate((1- prob_sklearn, prob_sklearn), axis=1)
	diff_GD =  np.concatenate((1- prob_GD, prob_GD), axis=1)
	diff_SGD =  np.concatenate((1- best_prob_SGD, best_prob_SGD), axis=1)

	#plot roc curves
	plot_roc(y_test, prob_sklearn)
	plot_roc(y_test, diff_SGD)
	plot_roc(y_test, prob_GD)
	plt.show()

	#plot cumulative gain curves
	plot_cumulative_gain(y_test, prob_sklearn)
	ax = plot_cumulative_gain(y_test, diff_SGD)
	plot_cumulative_gain(y_test, prob_GD)
	#plt.show()



	"""
	#plot roc curves
	plot_roc(y_test, diff_sklearn, plot_micro=False, plot_macro= False)
	plot_roc(y_test, diff_GD, plot_micro=False, plot_macro= False)
	plot_roc(y_test, diff_SGD, plot_micro=False, plot_macro= False)
	plt.show()

	#plot cumulative gain curves
	plot_cumulative_gain(y_test, diff_sklearn)
	plot_cumulative_gain(y_test, diff_GD)
	plot_cumulative_gain(y_test, diff_SGD)
	plt.show()	

	"""

	model_curve = auc_score
	area_baseline = 0.5
	area_ratio = (model_curve - area_baseline)/(area_baseline)
	print('Area Ratio:',area_ratio)


	"""
	model = ax.lines[0]
	baseline= ax.lines[1]
	optimal = ax.lines[2]

	model_curve = np.trapz(model.getydata(), model.getxdata())
	optimal_curve = np.trapz(optimal.getydata(), optimal.getxdata())
	area_baseline = np.trapz(baseline.getydata(), baseline.getxdata())

	area_ratio = (model_curve - area_baseline)/(optimal_curve - area_baseline)

	print(area_ratio)
	"""


	return accuracy, learning_rates


def LogisticRegression_self(X_train, X_test, y_train, y_test, learning_rates, iteration):
	"""
	Logistic regression with gradient descent
	"""

	# scoping number of training samples

	n_inputs = X_train.shape[0]
	n_features = X_train.shape[1]


	#iteration = 1000

	#eta_ = 1e-7
	#beta_0 = np.random.uniform(-10000, 10000, m)
	#calc_beta, norm = GradientDescent(X_train, beta_0, y_train, iteration, eta_)


	#for eta in np.logspace(np.log10(1e-6), np.log10(1e0), 7):
	accuracy = np.zeros(len(learning_rates))
	auc_score = np.zeros(len(learning_rates))
	area_ratio_curve = np.zeros(len(learning_rates))

	for i, eta in enumerate(learning_rates):
		
		beta_0 = np.random.uniform(-0.5,0.5,n_features)
		beta_0 = np.reshape(beta_0, (n_features, 1))
		optimal_beta, norm = GradientDescent(X_train, beta_0, y_train, iteration, eta)

		#print('beta zero:', beta_0)
		#print('Optimal beta:', optimal_beta)

		predict= Probability(X_test, optimal_beta) #defining values to be between 0 and 1
		yPred = (predict >= 0.5).astype(int) # converting to just 0 or 1
		#accuracy[i,j] = np.mean(yPred == y_test)
		accuracy[i] = metrics.accuracy_score(y_test, yPred)
		auc_score[i] = metrics.roc_auc_score(y_test, yPred)
		difference = y_test - yPred

		#area_ratio_curve[i,j] = area_ratio(y_test, yPred)

		"""
		if i> 0 and auc_score[i] > auc_score[i-1]:
			best_pred_SGD= yPred
		"""

		print('Accuracy {}, where learning rate= {} and iterations = {}'.format(np.mean(accuracy), eta, iteration))
		#print('Classification equal 1:', np.sum(difference ==1))
		#print('Classification equal 0:', np.sum(difference ==0))
		#print('Classification equal -1:', np.sum(difference == -1))
		print('AUC Score: {}'.format(np.mean(auc_score)))

		"""
		plt.plot(yPred, label='predict')
		plt.plot(optimal_beta, label ='optimal beta')
		plt.plot(y_test, label='test')
		plt.show()
		"""
	sns.set()
	sns.heatmap(pd.DataFrame(accuracy),  annot= True, fmt='.4g')
	plt.title('Grid-search for logistic regression')
	plt.ylabel('Learning rate: $\\eta$')
	plt.xlabel('Regularization Term: $\\lambda$')
	#plt.xticks(ticks=np.arange(len(learning_rates)) + 0.5, labels=learning_rates)
	#plt.yticks(ticks=np.arange(len(lambda_values)) + 0.5, labels=lambda_values)
	b, t = plt.ylim() # discover the values for bottom and top
	b += 0.5 # Add 0.5 to the bottom
	t -= 0.5 # Subtract 0.5 from the top
	plt.ylim(b, t) # update the ylim(bottom, top) values
	plt.show()

	"""
	sns.heatmap(pd.DataFrame(area_ratio_curve),  annot= True, fmt='.4g')
	plt.title('Grid-search for logistic regression')
	plt.ylabel('Learning rate: $\\eta$')
	plt.xlabel('Regularization Term: $\\lambda$')
	#plt.xticks(ticks=np.arange(len(learning_rates)) + 0.5, labels=learning_rates)
	#plt.yticks(ticks=np.arange(len(lambda_values)) + 0.5, labels=lambda_values)
	b, t = plt.ylim() # discover the values for bottom and top
	b += 0.5 # Add 0.5 to the bottom
	t -= 0.5 # Subtract 0.5 from the top
	plt.ylim(b, t) # update the ylim(bottom, top) values
	plt.show()
	"""

	sns.heatmap(pd.DataFrame(auc_score),  annot= True, fmt='.4g')
	plt.title('Grid-search for logistic regression')
	plt.ylabel('Learning rate: $\\eta$')
	plt.xlabel('Regularization Term: $\\lambda$')
	#plt.xticks(ticks=np.arange(len(learning_rates)) + 0.5, labels=learning_rates)
	#plt.yticks(ticks=np.arange(len(lambda_values)) + 0.5, labels=lambda_values)
	b, t = plt.ylim() # discover the values for bottom and top
	b += 0.5 # Add 0.5 to the bottom
	t -= 0.5 # Subtract 0.5 from the top
	plt.ylim(b, t) # update the ylim(bottom, top) values
	plt.show()

	#plot confusion matrix
	Confusion_Matrix(y_test, yPred)
	#Confusion_Matrix(y_test, best_pred_SGD)

	diff = np.concatenate((1- predict, predict), axis=1)

	#plot roc curve
	plot_roc(y_test, diff, plot_micro=False, plot_macro= False)
	plt.show()

	#plot cumulative gain curve
	ax = plot_cumulative_gain(y_test, diff)
	#x_d, y_d = cumulative_gain_curve(y_test, diff[:,1])
	plt.show()

	
	#model_curve = np.trapz(n_inputs, n_features)
	#optimal_curve = 0.788 + 0.5*0.2212
	#area_baseline = 0.5
	#area_ratio = (model_curve - baseline)/(optimal_curve - baseline)

	"""
	model = ax.lines[1]
	baseline= ax.lines[2]
	optimal=ax.lines[3]
	model_curve = np.trapz(model.getydata(), model.getxdata())
	optimal_curve = np.trapz(optimal.getydata(), optimal.getxdata())
	area_baseline = np.trapz(baseline.getydata(), baseline.getxdata())

	area_ratio = (model_curve - area_baseline)/(optimal_curve - area_baseline)

	print(area_ratio)
	"""

	return accuracy, learning_rates


def LogisticReg(X_train, X_test, y_train, y_test, learning_rates, lambda_values, iteration):

	# scoping number of training samples

	n_inputs = X_train.shape[0]
	n_features = X_train.shape[1]

	#model_curve = np.trapz(n_inputs, n_features)
	#optimal_curve = 0.788 + 0.5*0.2212
	#baseline = 0.5
	#area_ratio = (model_curve - baseline)/(optimal_curve - baseline)

	#iteration = 1000

	#eta_ = 1e-7
	#beta_0 = np.random.uniform(-10000, 10000, m)
	#calc_beta, norm = GradientDescent(X_train, beta_0, y_train, iteration, eta_)


	#for eta in np.logspace(np.log10(1e-6), np.log10(1e0), 7):
	accuracy = np.zeros((len(learning_rates), len(lambda_values)))
	auc_score = np.zeros((len(learning_rates), len(lambda_values)))
	area_ratio_curve = np.zeros((len(learning_rates), len(lambda_values)))

	for i, eta in enumerate(learning_rates):
		for j, lmbd in enumerate(lambda_values):
			beta_0 = np.random.uniform(-0.5,0.5,n_features)
			beta_0 = np.reshape(beta_0, (n_features, 1))
			optimal_beta, norm = GradientDescent(X_train, beta_0, y_train, iteration, eta)

			#print('beta zero:', beta_0)
			#print('Optimal beta:', optimal_beta)

			predict= Probability(X_test, optimal_beta) #defining values to be between 0 and 1
			yPred = (predict >= 0.5).astype(int) # converting to just 0 or 1
			#accuracy[i,j] = np.mean(yPred == y_test)
			accuracy[i,j] = metrics.accuracy_score(y_test, yPred)
			auc_score[i,j] = metrics.roc_auc_score(y_test, yPred)
			difference = y_test - yPred

			#area_ratio_curve[i,j] = area_ratio
			"""
			if i> 0 and auc_score[i] > auc_score[i-1]:
				best_pred_SGD= yPred
			"""

			print('Accuracy {}, where learning rate= {} and iterations = {}'.format(np.mean(accuracy), eta, iteration))
			#print('Classification equal 1:', np.sum(difference ==1))
			#print('Classification equal 0:', np.sum(difference ==0))
			#print('Classification equal -1:', np.sum(difference == -1))
			print('Area ratio: {}'.format(np.mean(auc_score)))

		"""
		plt.plot(yPred, label='predict')
		plt.plot(optimal_beta, label ='optimal beta')
		plt.plot(y_test, label='test')
		plt.show()
		"""
	sns.set()
	sns.heatmap(pd.DataFrame(accuracy),  annot= True, fmt='.4g')
	plt.title('Grid-search for logistic regression')
	plt.ylabel('Learning rate: $\\eta$')
	plt.xlabel('Regularization Term: $\\lambda$')
	#plt.xticks(ticks=np.arange(len(learning_rates)) + 0.5, labels=learning_rates)
	#plt.yticks(ticks=np.arange(len(lambda_values)) + 0.5, labels=lambda_values)
	b, t = plt.ylim() # discover the values for bottom and top
	b += 0.5 # Add 0.5 to the bottom
	t -= 0.5 # Subtract 0.5 from the top
	plt.ylim(b, t) # update the ylim(bottom, top) values
	plt.show()

	"""
	sns.heatmap(pd.DataFrame(area_ratio_curve),  annot= True, fmt='.4g')
	plt.title('Grid-search for logistic regression')
	plt.ylabel('Learning rate: $\\eta$')
	plt.xlabel('Regularization Term: $\\lambda$')
	#plt.xticks(ticks=np.arange(len(learning_rates)) + 0.5, labels=learning_rates)
	#plt.yticks(ticks=np.arange(len(lambda_values)) + 0.5, labels=lambda_values)
	b, t = plt.ylim() # discover the values for bottom and top
	b += 0.5 # Add 0.5 to the bottom
	t -= 0.5 # Subtract 0.5 from the top
	plt.ylim(b, t) # update the ylim(bottom, top) values
	plt.show()
	"""

	sns.heatmap(pd.DataFrame(auc_score),  annot= True, fmt='.4g')
	plt.title('Grid-search for logistic regression')
	plt.ylabel('Learning rate: $\\eta$')
	plt.xlabel('Regularization Term: $\\lambda$')
	#plt.xticks(ticks=np.arange(len(learning_rates)) + 0.5, labels=learning_rates)
	#plt.yticks(ticks=np.arange(len(lambda_values)) + 0.5, labels=lambda_values)
	b, t = plt.ylim() # discover the values for bottom and top
	b += 0.5 # Add 0.5 to the bottom
	t -= 0.5 # Subtract 0.5 from the top
	plt.ylim(b, t) # update the ylim(bottom, top) values
	plt.show()

	Confusion_Matrix(y_test, yPred)
	#Confusion_Matrix(y_test, best_pred_SGD)

	diff = np.concatenate((1- predict, predict), axis=1)

	plot_roc(y_test, diff, plot_micro=False, plot_macro= False)
	plt.show()

	ax = plot_cumulative_gain(y_test, diff)
	x_d, y_d = cumulative_gain_curve(y_test, diff[:,1])
	plt.show()


	return accuracy, learning_rates, lambda_values



def read_dataset():
	"""
	Read credit card data set
	"""

	onehotencoder = OneHotEncoder(categories="auto", sparse=False)
	scaler = StandardScaler(with_mean=False)
	cwd = os.getcwd()
	filename = cwd + '/default of credit card clients.xls'
	nanDict = {}
	#read file and create the dataframe
	df = pd.read_excel(filename, header=1, skiprows=0,
						index_col=0, na_values=nanDict)

	df.rename(index=str,
				columns={"default payment next month": "defaultPaymentNextMonth"},
				inplace=True)

	df = df.drop(df[(df.BILL_AMT1 == 0)&
						(df.BILL_AMT2 == 0)&
						(df.BILL_AMT3 == 0)&
						(df.BILL_AMT4 == 0)&
						(df.BILL_AMT5 == 0)&
						(df.BILL_AMT6 == 0)].index)
	df = df.drop(df[(df.PAY_AMT1 == 0)&
						(df.PAY_AMT2 == 0)&
						(df.PAY_AMT3 == 0)&
						(df.PAY_AMT4 == 0)&
						(df.PAY_AMT5 == 0)&
						(df.PAY_AMT6 == 0)].index)

	# Creating matrix X and target variable y
	X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
	y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values

	onehotencoder = OneHotEncoder(categories ='auto', sparse =False)
	scalar = StandardScaler(with_mean = False)

	#make one hots
	X = ColumnTransformer(
				[("", onehotencoder, [1,2,3,5,6,7,8,9]),],
				remainder="passthrough"
			).fit_transform(X)

	X = scalar.fit_transform(X)

	"""
	fig, ax = plt.subplots()
	for i in range(1,7):
		sns.distplot(df['PAY_AMT{}'.format(i)],ax=ax, kde=False,label='PAY_AMT{}'.format(i),hist_kws={'log':True})
	plt.ylabel('Count')
	plt.legend()

	fig, ax = plt.subplots()
	for i in range(1,7):
		sns.distplot(df['BILL_AMT{}'.format(i)],ax=ax, kde=False,label='BILL_AMT{}'.format(i),hist_kws={'log':True})
	plt.ylabel('Count')
	plt.legend()
	"""
	return X, y


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    return term1 + term2 + term3 + term4

	

def CreateFranke_data(x,y,n):

	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X


def MSE(z_test, z_pred):
	mse = np.mean((z_test-z_pred)**2)
	return mse


def R2(z_test, z_pred):
	R2 = 1 - np.mean((z_test-z_pred) ** 2) / np.mean((z_test - np.mean(z_test)) ** 2)
	return R2

def OLS(X_test, X_train, z_train):
	beta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(z_train)
	z_predict = X_test @ beta
	z_tilde = X_train @ beta

	return z_tilde, z_predict

def bestCurve(y):
	defaults = sum(y == 1)
	total = len(y)
	x = np.linspace(0, 1, total)
	y1 = np.linspace(0, 1, defaults)
	y2 = np.ones(total-defaults)
	y3 = np.concatenate([y1,y2])
	return x, y3



def cv_classification(num_splits, X, z, neutralnetwork, epochs, batch_size, eta, lmbd, iterations):
	
	k_fold = KFold(n_splits = num_splits, shuffle=True)

	cv_auc_scores = np.zeros(num_splits)
	cv_accuracy_scores = np.zeros(num_splits)

	i = 0

	for train_inds, test_inds in k_fold.split(X):
		sc = StandardScaler()

		x_train = X[train_inds]
		x_test = X[test_inds]

		x_train = sc.fit_transform(x_train)
		x_test = sc.transform(x_test)

		y_train = z[train_inds]
		y_test = z[test_inds]

		neutralnetwork.train(iterations)

		proba = neutralnetwork.predict(x_test)

		cv_auc_scores[i] = metrics.roc_auc_score(y_test, proba)
		print("AUC-score: ", metrics.roc_auc_score( y_test, proba))

		predicted = np.where(proba >= 0.5, 1, 0)
		cv_accuracy_scores[i] = metrics.accuracy_score(y_test, predicted)

		print("Accuracy: ", metrics.accuracy_score(y_test, predicted))
		print(confusion_matrix(y_test, predicted))

		i+=1

	cv_auc = np.mean(cv_auc_scores)
	cv_accuracy = np.mean(cv_accuracy_scores)

	print("CV-AUC: ", cv_auc, "  CV-accuracy: ", cv_accuracy )
	return cv_auc, cv_accuracy


def cv_regression(num_splits, X, z, neutralnetwork, epochs, eta, lmbd, iterations):

	k_fold = KFold(n_splits = num_splits, shuffle=True)

	cv_mse_scores = np.zeros(num_splits)
	cv_r2 = np.zeros(num_splits)

	i = 0

	for train_inds, test_inds in k_fold.split(X):
		sc = StandardScaler()

		x_train = X[train_inds]
		x_test = X[test_inds]

		x_train = sc.fit_transform(x_train)
		x_test = sc.transform(x_test)

		y_train = z[train_inds]
		y_test = z[test_inds]

		neutralnetwork.train(iterations,eta, lmbd)

		pred = neutralnetwork.predict_regression_franke(x_test)

		print(y_test.shape)
		print(pred.shape)

		cv_mse_scores[i] = metrics.mean_squared_error(y_test, pred)

		cv_r2[i] = metrics.r2_score(y_test, pred)

		#print(i)
		#print("MSE_final: ", cv_mse_scores[i])

		i+=1

	cv_mse = np.mean(cv_mse_scores)

	print("CV-MSE: ", cv_mse)
	return cv_mse
























