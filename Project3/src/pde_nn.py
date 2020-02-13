"""
Computing the diffusion equation with neural network model.

"""

import autograd.numpy as np
import autograd.numpy.random as npr
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import time
from sklearn.metrics import r2_score, mean_squared_error

import tensorflow.compat.v1 as tf
# import tensorflow as tf
tf.disable_eager_execution()
from tensorflow.compat.v1 import keras
tf.logging.set_verbosity(tf.logging.ERROR)
tf.executing_eagerly()



def u(x):
	return tf.sin(np.pi*x)


def u_analytic(x, t):
	#analytical solution
	return tf.exp(-np.pi**2*t)*tf.sin(np.pi*x)


def NN_diffusion(nx, nt, iterations, num_hidden_neurons, learning_rate):
	tf.reset_default_graph()

	#set a seed to get the same resuls from every run
	tf.set_random_seed(4155)

	x_= np.linspace(0, 1, nx)
	t_= np.linspace(0, 1, nt)

	X,T =np.meshgrid(x_,t_)

	x= X.ravel()
	t = T.ravel()

	#Construct Neural network
	zeros= tf.reshape(tf.convert_to_tensor(np.zeros(x.shape)), shape=(-1,1))
	x = tf.reshape(tf.convert_to_tensor(x), shape=(-1,1))
	t = tf.reshape(tf.convert_to_tensor(t), shape=(-1,1))

	total_points = tf.concat([x,t],1) #input layer

	#number of hidden layers
	num_hidden_layers = len(num_hidden_neurons)
	#print('hidden layers:',num_hidden_layers)

	X=tf.convert_to_tensor(X)
	T= tf.convert_to_tensor(T)

	#construct the network
	#layer structures
	with tf.name_scope('dnn'):
		num_hidden_layers = np.size(num_hidden_neurons)
		previous_layer = total_points

		for l in range(num_hidden_layers):
			current_layer = tf.layers.dense(previous_layer, num_hidden_neurons[l], name=('hidden{}'.format(l+1)), activation=tf.nn.sigmoid)

			previous_layer = current_layer

		dnn_output = tf.layers.dense(previous_layer, 1, name='output', activation=None)


	#Define the cost function
	with tf.name_scope('loss'):
		
		g_trial = (1 - t)*u(x) + x*(1 - x)*t*dnn_output
		g_trial_d2x = tf.gradients(tf.gradients(g_trial, x), x)
		g_trial_dt = tf.gradients(g_trial,t)
		
		loss = tf.losses.mean_squared_error(zeros, g_trial_dt[0] - g_trial_d2x[0])


	#Defining optimizer
	with tf.name_scope('train'):
		optimizer = tf.train.AdamOptimizer(learning_rate)
		training_op = optimizer.minimize(loss)

	#Define a node that initializes all of the other nodes in the computational graph
	init = tf.global_variables_initializer()

	#g_analytic = tf.sin(np.pi*x)*tf.exp(-np.pi*np.pi*t)
	g_analytic = u_analytic(x,t)
	g_dnn = None

	start = time.time()

	#The execution phase
	with tf.Session() as sess:
		#intialtize the initial cost
		init.run()

		#training of the network
		for i in range(iterations):
			sess.run(training_op)

		#store the results
		g_analytic = g_analytic.eval() #analytic solution
		g_dnn = g_trial.eval() #Neural network solution
		cost = loss.eval() #cost evaluation
	stop = time.time()
	print('time duration:', stop-start)

	"""
	#compare with analytical solution
	diff = np.abs(g_analytic - g_dnn)
	max_diff =np.max(diff)
	print('max absolute difference between the analytical and the tensorflow: ', max_diff)
	"""

	#statistical computations
	r2 = r2_score(g_analytic, g_dnn)
	mse = mean_squared_error(g_analytic, g_dnn)
	print('R2:',r2)
	print('MSE:', mse)
	
	G_analytic = g_analytic.reshape((nt, nx))
	G_dnn = g_dnn.reshape((nt, nx))

	#compare with analytical solution
	diff = np.abs(G_analytic - G_dnn)
	max_diff =np.max(diff)
	print('max absolute difference between the analytical and the tensorflow: ', max_diff)

	X,T =np.meshgrid(x_, t_)

	
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.set_title('Solution from the deep neural network with %d layer \n and 50 neurons within hidden layer'%len(num_hidden_neurons))
	s = ax.plot_surface(X,T,G_dnn,linewidth=0,antialiased=False,cmap=cm.viridis)
	ax.set_xlabel('Time $t$')
	ax.set_ylabel('Position $x$')
	#plt.savefig('solution_deep_nn_new.png')


	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.set_title('Analytical solution of diffusion equation with 4 hidden layers \n and 50 neurons within hidden layer')
	s = ax.plot_surface(X,T,G_analytic,linewidth=0,antialiased=False,cmap=cm.viridis)
	ax.set_xlabel('Time $t$')
	ax.set_ylabel('Position $x$')
	#plt.savefig('analytical_solution_nn_new.png')


	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.set_title('Difference between the numerical and analytical solution, \n with 4 hidden layers and 50 neurons within hidden layer')
	s = ax.plot_surface(X,T,diff,linewidth=0,antialiased=False,cmap=cm.viridis)
	ax.set_xlabel('Time $t$')
	ax.set_ylabel('Position $x$')
	#plt.savefig('difference_nn_new.png')

	plt.show()
	
	"""
	# Take some slices of the 3D plots just to see the solutions at particular times
	indx1 = 0
	indx2 = int(nt/2)
	indx3 = nt-1

	t1 = t_[indx1]
	t2 = t_[indx2]
	t3 = t_[indx3]

	# Slice the results from the DNN
	res1 = g_dnn[:,indx1]
	res2 = g_dnn[:,indx2]
	res3 = g_dnn[:,indx3]

	# Slice the analytical results
	res_analytical1 = G_analytical[:,indx1]
	res_analytical2 = G_analytical[:,indx2]
	res_analytical3 = G_analytical[:,indx3]

	# Plot the slices
	plt.figure()
	plt.title("Computed solutions at time = %g"%t1)
	plt.plot(x_, res1)
	plt.plot(x_,res_analytical1)
	plt.legend(['dnn','analytical'])
	plt.savefig('computed_solution_nn_t1.png')

	plt.figure()
	plt.title("Computed solutions at time = %g"%t2)
	plt.plot(x_, res2)
	plt.plot(x_,res_analytical2)
	plt.legend(['dnn','analytical'])
	plt.savefig('computed_solution_nn_t2.png')

	plt.figure()
	plt.title("Computed solutions at time = %g"%t3)
	plt.plot(x_, res3)
	plt.plot(x_,res_analytical3)
	plt.legend(['dnn','analytical'])
	plt.savefig('computed_solution_nn_t3.png')

	plt.show()
	"""


	return diff


def main():

	#parameters
	x0 = 0.0
	L = 1.0

	t0 = 0.0
	t1 = 1.0

	nx = 10
	nt = 10
	
	learning_rate = 0.01
	iterations = 500
	num_hidden_neurons = [50,50]

	NN_diff = NN_diffusion(nx,nt, iterations, num_hidden_neurons, learning_rate)


main()
















































