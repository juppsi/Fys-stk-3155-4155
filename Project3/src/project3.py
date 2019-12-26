"""
Numerical and analytical solution of 1 dimentional diffusion equation, using forward euler.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, math

	

def ForwardEuler(alpha,u,nt,nx):
	#Explicit forward euler SCHEME
	for it in range(1,nt+2):
		for ix in range(1,nx+1):
			u[it,ix]= alpha*u[it-1,ix-1] + (1.0-2.0*alpha)*u[it-1,ix] + alpha*u[it-1,ix+1]	
	return u



def AnalyticalOneDim():
	#Analytical solution for the 1D diffustion equation 

	for t in [0.003, 0.03, 0.1,0.3, 0.6]: #time evolution
		x=np.linspace(0,1,400)
		u=np.zeros(len(x))
		n=1000

		for n in range(1,n+1):
			u += np.sin(np.pi*n*x)*np.exp(-np.pi*n**2*t)/float(np.pi*n)

		u= x + 2*u

		ts = "t =%s" % str(t)
		plt.plot(u,x, label= ts)
		plt.xlabel('$x$')
		plt.ylabel('$u(x,t)$')
		plt.axis([0, 1.01,0.0, 1.0])
		#plt.savefig('analytical1Ddd.pdf')
		plt.title('Analytical soluion of one- dimenstional diffusion equation')

	plt.tight_layout()
	plt.legend()
	plt.show()

AnalyticalOneDim()



def numerical1D():
	#Computing numerical diffusion equation 1 dimension

	#define the values
	L = 1.0
	dx = 1./float(10) #h
	dt = (dx**2)/float(2.0)
	alpha = dt/float(dx**2)


	for Tfinal in [0.003, 0.03, 0.09, 0.3]: #time evolution

		nt = np.int(Tfinal/float(dt) -2.) #grid points
		nx = np.int(L/float(dx) - 2.) #grid points

		time = np.linspace(0,Tfinal,nt+2) #grid time

		x = np.linspace(0,L,nx+2)
		u = np.zeros((nt+2,nx+2))

		# boundary condition 
		u[:,-1] = 1
	
		v = u.copy()

		vv = ForwardEuler(alpha,v,nt,nx)
		
		FE_ = " t=%s" % str(Tfinal) #FE= forward Euler

	
		line, = plt.plot(x, v[nt+1], label=FE_)
		plt.xlabel('$x$')
		plt.ylabel('$u(x,t)$')
	

	plt.title('1D diffusion equation computed by finite difference scheme with \n dx= %s, dt= %.3g, alpha= %s '%(dx,dt,alpha))
	plt.legend()
	plt.tight_layout()
	#plt.savefig('numerical1D.pdf')
	plt.show()

numerical1D()




