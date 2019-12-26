"""
Solving eigenvalue problem by using neural network.
Worked with a guy from datalab.
"""

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
import time

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.disable_eager_execution()
from tensorflow.compat.v1 import keras
tf.logging.set_verbosity(tf.logging.ERROR)
tf.executing_eagerly()


def f(x,A_e):
	# Function from the article from Yi et al. 2004.

	I = tf.eye(n,dtype= tf.float64)
	x_t = tf.transpose(x)
	a= tf.matmul(x_t,x)*A_e
	b= (1-tf.matmul(tf.matmul(x_t,A_e),x))*I

	func= tf.matmul((a+b),x)

	return func

def calc_eigvalue(v,A, n):
	#compute the eigenvalues

	v = v.reshape(n,1)
	v_t = v.transpose()
	n = np.matmul(np.matmul(v_t,A),v)[0,0]
	d = np.matmul(v_t,v)[0,0]

	calc = n/d 

	return calc

def NN(A_e, v0, t_max, dt, n, epsilon, learning_rate):
	N_T = int(t_max/dt)
	N_X = n

	t= np.linspace(0, (N_T- 1)*dt, N_T)
	x= np.linspace(1, N_X, N_X)


	X,T = np.meshgrid(x,t)
	V, T_new = np.meshgrid(v0,t)

	x_new = (X.ravel()).reshape(-1,1)
	t_new = (T.ravel()).reshape(-1,1)
	v0_new = (V.ravel()).reshape(-1,1)

	x_c = tf.convert_to_tensor(x_new, dtype=tf.float64)
	t_c = tf.convert_to_tensor(t_new, dtype=tf.float64)
	v0_c = tf.convert_to_tensor(v0_new, dtype=tf.float64)

	total_points = tf.concat([x_c, t_c],1)

	num_hidden_neurons = [10, 10, 10]
	num_hidden_layers = np.size(num_hidden_neurons)

	#layer structures
	with tf.name_scope('dnn'):
		#input layer
		previous_layer = total_points
		#hidden layers
		for l in range(num_hidden_layers):
			current_layer = tf.layers.dense(previous_layer, num_hidden_neurons[l], name=('hidden{}'.format(l+1)), activation=tf.nn.sigmoid)

			previous_layer = current_layer
		#output layer
		dnn_output = tf.layers.dense(previous_layer, 1, name='output', activation=None)

	# loss function
	with tf.name_scope('loss'):
		#trial
		trial = dnn_output*t_c +v0_c*k

		#calucation of gradient
		trial_dt = tf.gradients(trial, t_c)

		#step size iterations
		trial_ = tf.reshape(trial,(N_T,N_X))
		trial_dt_ = tf.reshape(trial_dt,(N_T, N_X))

		#cost function
		cost_ = 0
		for i in range(N_T):
			trial_t = tf.reshape(trial_[i],(n,1))
			trial_dt_t = tf.reshape(trial_dt_[i],(n,1))
			rhs = f(trial_t,A_e) - trial_t
			error = tf.square(-trial_dt_t+rhs)
			cost_ += tf.reduce_sum(error)
		cost = tf.reduce_sum(cost_/(N_X*N_T), name='cost')


	#Defining optimizer
	with tf.name_scope('train'):
		optimizer = tf.train.AdamOptimizer(learning_rate)
		training_op = optimizer.minimize(cost)

	init = tf.global_variables_initializer()
	v_dnn = None


	#The execution phase
	with tf.Session() as sess:
		init.run()

		#Network training
		i=0
		while cost.eval()> epsilon:
			sess.run(training_op)
			i+=1

		v_dnn = tf.reshape(trial,(N_T, N_X))
		v_dnn = v_dnn.eval()

		return v_dnn, t, i


# k=1 maximum eigenvalue and k=-1 minimum eigenvalue
k= +1

#generate symmetric and real matrix
n = 6
Q = np.random.rand(n,n)
A= (Q.T +Q)/2

print('A=', A)

A_t = k*A
A_e = tf.convert_to_tensor(A_t,dtype=tf.float64)

#compute the eigenvector and eigenvalue using numpy and CPU-time
start_np = time.time()
w_numpy, v_numpy = np.linalg.eig(A)
stop_np = time.time()

print('time duration numpy:', stop_np - start_np)

idx= np.argsort(w_numpy)
v_numpy = v_numpy[:, idx]
w_min_numpy = np.min(w_numpy)
w_max_numpy = np.max(w_numpy)
v_min_numpy = v_numpy[:,0]
v_max_numpy = v_numpy[:,-1]

eig_value = w_numpy[idx]
eig_vec = v_numpy[:,idx]

print('eig value min:',eig_value[0]) #min
print('eig value max:', eig_value[-1]) #max

print('eig vec min:', eig_vec[0]) #min
print('eig vec max:', eig_vec[-1]) #max

#Define the input of Neural network
epsilon = 0.0001
t_max = 3
learning_rate = 0.001
dt = 0.1
v0 = np.random.rand(n)

#CPU neural network
start = time.time()
v_dnn, t, i = NN(A_e, v0, t_max, dt, n, epsilon, learning_rate)
stop = time.time()

print('time duration nn:', stop-start)

#plot of max eigenvector and min eigenvector
fig, ax = plt.subplots()
ax.plot(t, v_dnn)
ax.set_xlabel('Time $t$')
ax.set_ylabel('Estimated $v_{max}$ elements')
#ax.set_ylabel('Estimated $v_{min}$ elements')
#ax.text(0.7, 0.95, 'dt = {} \n $\epsilon$ = {} \n i  = {}'.format(dt,epsilon,i))
ax.set_title('Estimated eigenvalue with time evolution, \n with dt = {}, $\epsilon$ = {}, and {} iterations'.format(dt,epsilon,i))
#ax.tick_params(axis='both')
plt.legend(['$v_{1}$', '$v_{2}$', '$v_{3}$', '$v_{4}$', '$v_{5}$', '$v_{6}$'])
plt.tight_layout()
#plt.savefig('eigenvalue_max_new.png')
plt.show()


v_last = v_dnn[-1]
w_last = calc_eigvalue(v_last, A, n)

v_first = v_dnn[0]
w_first = calc_eigvalue(v_first, A, n)

# v= eigenvalue, w= eigenvector
print('Eigenvalue:')
print('v min:', v0)
print('v max:', v_last)
print('unit v nn:', v_last/np.linalg.norm(v_last))
print('v_max numpy:', v_max_numpy)
print('v_min numpy:', v_min_numpy)
print('v_first:', v_first)
print('w_first:', w_first)

print('------')
print('Eigenvector:')
print('wnn:', w_last)
print('w numpy', w_numpy)
print('w_max numpy:', w_max_numpy)
print('w_min numpy:', w_min_numpy)
































































