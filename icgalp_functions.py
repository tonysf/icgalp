#The contents of this file make up the functions used to generate the plots
#for the paper Inexact and Stochastic Generalized Conditional Gradient with 
#Augmented Lagrangian and Proximal Step by Silveti-Falls, Molinari, and Fadili

import numpy as np
import scipy.spatial as spat
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import scipy.stats as spst
import scipy.special as spspec
import sympy as spy
import progressbar as pb

#gradient for a batch sample of nabla E
def calc_bgradE(x, mu, y, A, Ah, Rho, lucky_indices):
	v = np.zeros(x.shape)
	Ahmu = Ah.dot(mu)
	AhAx = Rho * Ah.dot(A.dot(x))
	#compute the grad for the lucky batch
	for l in lucky_indices:
		v[l] = x[l] - y[l] + Ahmu[l] + AhAx[l]
	return v

#gradient for single sample of nabla f, nabla L
def calc_gradL(x, y, lucky_index):
	v = np.zeros(x.shape)
	#compute grad for lucky index
	v[lucky_index] = x[lucky_index] - y[lucky_index]
	return v

def calc_gradLQ(x, y, lucky_index, A, Ah, Rho):
	v = np.zeros(x.shape)
	#compute grad for lucky index
	v[lucky_index] = x[lucky_index] - y[lucky_index] + Rho * Ah.dot(A.dot(x))[lucky_index]
	return v
	
def calc_bgradLQ(x, y, lucky_indices, A, Ah, Rho):
	v = np.zeros(x.shape)
	AhAx = Rho * Ah.dot(A.dot(x))
	#compute the grad for the lucky batch
	for l in lucky_indices:
		v[l] = x[l] - y[l] + AhAx[l]
	return v

#gradient for a batch sample of nabla f
def calc_bgradL(x, y, lucky_indices):
	v = np.zeros(x.shape)
	#compute the grad for the lucky batch
	for l in lucky_indices:
		v[l] = x[l] - y[l]
	return v

#variance reduction using averaging
def calc_vrgrad(oldgrad, newgrad, stochweight):
	return (1 - stochweight) * oldgrad + stochweight * newgrad

#gradient terms coming from augmented lagrangian (not f)
def calc_rest(x, mu, A, Ah, Rho):
	return Ah.dot(mu) + (Rho * Ah.dot(A.dot(x)))

#this function solves the linear minimization oracle over the delta l1 ball
#then it takes a step in the direction given from the LMO
def primal_step(x, grad, stepsize, delta):
	#initialize step direction
	step_direc = np.zeros(x.shape)
	#compute step for l1 vec norm
	#first compute the good index corresponding to largest entry in magnitude
	ind = np.argmax(np.absolute(grad))
	#compute the sign of this entry
	sign = np.sign(grad[ind])
	#update this entry in the step direction according to size of the ball (delta)
	step_direc[ind] = -delta * sign
	#return the new primal variable
	return (1 - stepsize) * x + stepsize * step_direc

#compute the update of the dual variable mu_k
def dual_step(x, mu, stepsize, A):
	return mu + (stepsize * A.dot(x))

#original cgalp, errorless gradient
def cgalp(y, A, delta, b, a, Rho, itera, xstar):
	n = y.shape[0]
	x = np.zeros(y.shape)
	erg = np.zeros(y.shape)
	mu = np.zeros(A.shape[0])
	feas = np.zeros(itera)
	dist = np.zeros(itera)
	Gamma = 0
	Ah = A.transpose()
	for i in pb.progressbar(range(itera - 1)):
		#the stepsize gamma_k = (log(k+2)^a)/((k+1)^(1-b))
		stepsize = (np.log(2 + i) ** a)/((1+i) ** (1 - b))
		#the ergodic variable is the cesaro average w.r.t gamma_k
		erg = ((stepsize * x) + (Gamma * erg))/(Gamma + stepsize)
		#record feasibility and optimality values for ergodic variable
		feas[i] = np.linalg.norm(A.dot(erg)) ** 2
		dist[i] = np.linalg.norm(erg - xstar) ** 2
		#compute the gradient
		grad = (x - y)/float(x.shape[0]) + calc_rest(x, mu, A, Ah, Rho)
		#update primal and dual variables
		x = primal_step(x, grad, stepsize, delta)
		mu = dual_step(x, mu, stepsize, A)
		#update sum of gammas for cesaro average
		Gamma += stepsize
	stepsize = (np.log(1 + itera) ** a)/((itera) ** (1 - b))
	erg = ((stepsize * x) + (Gamma * erg))/(Gamma + stepsize)
	feas[itera - 1] = np.linalg.norm(A.dot(erg)) **2
	dist[itera - 1] = np.linalg.norm(erg - xstar) ** 2
	return x, mu, erg, feas, dist

#icgalp with random sweeping on gradf_i
def icgalp_rswf(y, A, batchsize, delta, b, a, Rho, itera, xstar):
	n = y.shape[0]
	x = np.zeros(y.shape)
	erg = np.zeros(y.shape)
	mu = np.zeros(A.shape[0])
	feas = np.zeros(itera)
	dist = np.zeros(itera)
	Gamma = 0
	Ah = A.transpose()
	#initialize the gradient
	#old
	#grad = (x - y)/float(n)
	grad = np.zeros(y.shape)
	for i in pb.progressbar(range(itera - 1)):
		#the stepsize gamma_k = (log(k+2)^a)/((k+1)^(1-b))
		stepsize = (np.log(2 + i) ** a)/((1+i) ** (1 - b))
		#the ergodic variable is the cesaro average w.r.t gamma_k
		erg = ((stepsize * x) + (Gamma * erg))/(Gamma + stepsize)
		#record the feasibility and optimality for ergodic variable
		feas[i] = np.linalg.norm(A.dot(erg)) ** 2
		dist[i] = np.linalg.norm(erg - xstar) ** 2
		#calculate the current index to be updated
		l = np.random.choice(n, batchsize, replace=False)
		#replace the lucky_index'th component with a newly calculated grad
		grad[l] = (x[l] - y[l])/float(n)
		#calculate the remaining grad terms from aug lagrangian
		restgrad = calc_rest(x, mu, A, Ah, Rho)
		fullgrad = grad + restgrad
		#update primal and dual variabels
		x = primal_step(x, fullgrad, stepsize, delta)
		mu = dual_step(x, mu, stepsize, A)
		#update sum of gammas for cesaro averaging
		Gamma += stepsize
	stepsize = (np.log(1 + itera) ** a)/((itera) ** (1 - b))
	erg = ((stepsize * x) + (Gamma * erg))/(Gamma + stepsize)
	feas[itera - 1] = np.linalg.norm(A.dot(erg)) **2
	dist[itera - 1] = np.linalg.norm(erg - xstar) ** 2
	return x, mu, erg, feas, dist

#icgalp with random sweeping on gradf_i and quadratic term
def icgalp_rswfQ(y, A, batchsize, delta, b, a, Rho, itera, xstar):
	n = y.shape[0]
	x = np.zeros(y.shape)
	erg = np.zeros(y.shape)
	mu = np.zeros(A.shape[0])
	feas = np.zeros(itera)
	dist = np.zeros(itera)
	Gamma = 0
	Ah = A.transpose()
	#initialize the full gradient of f at x0
	#old
	#grad = (x - y)/float(n)
	grad = np.zeros(y.shape)
	for i in pb.progressbar(range(itera - 1)):
		#the stepsize gamma_k = (log(k+2)^a)/((k+1)^(1-b))
		stepsize = (np.log(2 + i) ** a)/((1+i) ** (1 - b))
		#the ergodic variable is the cesaro average w.r.t gamma_k
		erg = ((stepsize * x) + (Gamma * erg))/(Gamma + stepsize)
		#record the feasibility and optimality for ergodic variable
		feas[i] = np.linalg.norm(A.dot(erg)) ** 2
		dist[i] = np.linalg.norm(erg - xstar) ** 2
		#calculate the current index to be updated
		l = np.random.choice(n, batchsize, replace=False)
		#replace the lucky_index'th component with a newly calculated grad
		grad[l] = (x[l] - y[l])/float(n) + Rho * Ah.dot(A.dot(x))[l]
		#calculate the remaining grad terms from aug lagrangian
		restgrad = Ah.dot(mu)
		fullgrad = grad + restgrad
		#update primal and dual variables
		x = primal_step(x, fullgrad, stepsize, delta)
		mu = dual_step(x, mu, stepsize, A)
		#update sum of gammas for cesaro averaging
		Gamma += stepsize
	stepsize = (np.log(1 + itera) ** a)/((itera) ** (1 - b))
	erg = ((stepsize * x) + (Gamma * erg))/(Gamma + stepsize)
	feas[itera - 1] = np.linalg.norm(A.dot(erg)) **2
	dist[itera - 1] = np.linalg.norm(erg - xstar) ** 2
	return x, mu, erg, feas, dist

#icgalp with sweeping on gradf_i
def icgalp_swf(y, A, delta, b, a, Rho, itera, xstar):
	n = y.shape[0]
	x = np.zeros(y.shape)
	erg = np.zeros(y.shape)
	mu = np.zeros(A.shape[0])
	feas = np.zeros(itera)
	dist = np.zeros(itera)
	Gamma = 0
	Ah = A.transpose()
	#initialize the gradient
	#old
	#grad = (x - y)/float(n)
	grad = np.zeros(y.shape)
	for i in pb.progressbar(range(itera - 1)):
		#the stepsize gamma_k = (log(k+2)^a)/((k+1)^(1-b))
		stepsize = (np.log(2 + i) ** a)/((1+i) ** (1 - b))
		#the ergodic variable is the cesaro average w.r.t gamma_k
		erg = ((stepsize * x) + (Gamma * erg))/(Gamma + stepsize)
		#record the feasibility and optimality for ergodic variable
		feas[i] = np.linalg.norm(A.dot(erg)) ** 2
		dist[i] = np.linalg.norm(erg - xstar) ** 2
		#calculate the current index to be updated
		l = i % n
		#replace the lucky_index'th component with a newly calculated grad
		grad[l] = (x[l] - y[l])/float(n)
		#calculate the remaining grad terms from aug lagrangian
		restgrad = calc_rest(x, mu, A, Ah, Rho)
		fullgrad = grad + restgrad
		#update primal and dual variabels
		x = primal_step(x, fullgrad, stepsize, delta)
		mu = dual_step(x, mu, stepsize, A)
		#update sum of gammas for cesaro averaging
		Gamma += stepsize
	stepsize = (np.log(1 + itera) ** a)/((itera) ** (1 - b))
	erg = ((stepsize * x) + (Gamma * erg))/(Gamma + stepsize)
	feas[itera - 1] = np.linalg.norm(A.dot(erg)) **2
	dist[itera - 1] = np.linalg.norm(erg - xstar) ** 2
	return x, mu, erg, feas, dist

#icgalp with sweeping on gradf_i and quadratic term
def icgalp_swfQ(y, A, delta, b, a, Rho, itera, xstar):
	n = y.shape[0]
	x = np.zeros(y.shape)
	erg = np.zeros(y.shape)
	mu = np.zeros(A.shape[0])
	feas = np.zeros(itera)
	dist = np.zeros(itera)
	Gamma = 0
	Ah = A.transpose()
	#initialize the full gradient of f at x0
	#old
	#grad = (x - y)/float(n)
	grad = np.zeros(y.shape)
	for i in pb.progressbar(range(itera - 1)):
		#the stepsize gamma_k = (log(k+2)^a)/((k+1)^(1-b))
		stepsize = (np.log(2 + i) ** a)/((1+i) ** (1 - b))
		#the ergodic variable is the cesaro average w.r.t gamma_k
		erg = ((stepsize * x) + (Gamma * erg))/(Gamma + stepsize)
		#record the feasibility and optimality for ergodic variable
		feas[i] = np.linalg.norm(A.dot(erg)) ** 2
		dist[i] = np.linalg.norm(erg - xstar) ** 2
		#calculate the current index to be updated
		l = i % n
		#replace the lucky_index'th component with a newly calculated grad
		grad[l] = (x[l] - y[l])/float(n) + Rho * Ah.dot(A.dot(x))[l]
		#calculate the remaining grad terms from aug lagrangian
		restgrad = Ah.dot(mu)
		fullgrad = grad + restgrad
		#update primal and dual variables
		x = primal_step(x, fullgrad, stepsize, delta)
		mu = dual_step(x, mu, stepsize, A)
		#update sum of gammas for cesaro averaging
		Gamma += stepsize
	stepsize = (np.log(1 + itera) ** a)/((itera) ** (1 - b))
	erg = ((stepsize * x) + (Gamma * erg))/(Gamma + stepsize)
	feas[itera - 1] = np.linalg.norm(A.dot(erg)) **2
	dist[itera - 1] = np.linalg.norm(erg - xstar) ** 2
	return x, mu, erg, feas, dist

#icgalp with sampling on gradf_i and variance reduction
def icgalp_vrf(y, A, delta, s, b, a, Rho, itera, xstar):
	n = y.shape[0]
	x = np.zeros(y.shape)
	erg = np.zeros(y.shape)
	mu = np.zeros(A.shape[0])
	feas = np.zeros(itera)
	dist = np.zeros(itera)
	Gamma = 0
	Ah = A.transpose()
	#initialize grad as 0 for the recursion
	vrgrad = np.zeros(n)
	for i in pb.progressbar(range(itera - 1)):
		#set the step size and the stochastic weight accordingly
		stepsize = (np.log(2 + i) ** a)/((1+i) ** (1 - b))
		stochweight = (np.log(2 + i) ** (a * s))/((1+i) ** (s * (1 - b)))
		erg = ((stepsize * x) + (Gamma * erg))/(Gamma + stepsize)
		feas[i] = np.linalg.norm(A.dot(erg)) ** 2
		dist[i] = np.linalg.norm(erg - xstar) ** 2
		#draw an index randomly
		lucky_index = np.random.randint(n, size=1)[0]
		#compute a gradient according to lucky index
		newgrad = calc_gradL(x, y, lucky_index)
		#do the variance reduction recursion
		vrgrad = calc_vrgrad(vrgrad, newgrad, stochweight)
		#compute the other grad terms of aug lagrangian
		restgrad = calc_rest(x, mu, A, Ah, Rho)
		grad = vrgrad + restgrad
		#update primal var
		x = primal_step(x, grad, stepsize, delta)
		#update dual var
		mu = dual_step(x, mu, stepsize, A)
		#update sum of gammas
		Gamma += stepsize
	stepsize = (np.log(1 + itera) ** a)/((itera) ** (1 - b))
	erg = ((stepsize * x) + (Gamma * erg))/(Gamma + stepsize)
	feas[itera - 1] = np.linalg.norm(A.dot(erg)) **2
	dist[itera - 1] = np.linalg.norm(erg - xstar) ** 2
	return x, mu, erg, feas, dist
	
#icgalp with sampling on gradf_i, quadratic term, and variance reduction
def icgalp_vrfQ(y, A, delta, s, b, a, Rho, itera, xstar):
	n = y.shape[0]
	x = np.zeros(y.shape)
	erg = np.zeros(y.shape)
	mu = np.zeros(A.shape[0])
	feas = np.zeros(itera)
	dist = np.zeros(itera)
	Gamma = 0
	Ah = A.transpose()
	#initialize grad as 0 for the recursion
	vrgrad = np.zeros(n)
	for i in pb.progressbar(range(itera - 1)):
		#set the step size and the stochastic weight accordingly
		stepsize = (np.log(2 + i) ** a)/((1+i) ** (1 - b))
		stochweight = (np.log(2 + i) ** (a * s))/((1+i) ** (s * (1 - b)))
		erg = ((stepsize * x) + (Gamma * erg))/(Gamma + stepsize)
		feas[i] = np.linalg.norm(A.dot(erg)) ** 2
		dist[i] = np.linalg.norm(erg - xstar) ** 2
		#draw an index randomly
		lucky_index = np.random.randint(n, size=1)[0]
		#compute a gradient according to lucky index
		newgrad = calc_gradLQ(x, y, lucky_index, A, Ah, Rho)
		#do the variance reduction recursion
		vrgrad = calc_vrgrad(vrgrad, newgrad, stochweight)
		#compute the linear grad term of aug lagrangian
		restgrad = Ah.dot(mu)
		grad = vrgrad + restgrad
		#update primal var
		x = primal_step(x, grad, stepsize, delta)
		#update dual var
		mu = dual_step(x, mu, stepsize, A)
		#update sum of gammas
		Gamma += stepsize
	stepsize = (np.log(1 + itera) ** a)/((itera) ** (1 - b))
	erg = ((stepsize * x) + (Gamma * erg))/(Gamma + stepsize)
	feas[itera - 1] = np.linalg.norm(A.dot(erg)) **2
	dist[itera - 1] = np.linalg.norm(erg - xstar) ** 2
	return x, mu, erg, feas, dist

#icgalp with batch sampling of gradf_i and variance reduction
def icgalp_vrfb(y, A, batchsize, delta, s, b, a, Rho, itera, xstar):
	n = y.shape[0]
	x = np.zeros(y.shape)
	erg = np.zeros(y.shape)
	mu = np.zeros(A.shape[0])
	feas = np.zeros(itera)
	dist = np.zeros(itera)
	Gamma = 0
	Ah = A.transpose()
	#initialize the gradient as 0 for the recursion
	vrgrad = np.zeros(n)
	for i in pb.progressbar(range(itera - 1)):
		#set the step size and the stochastic weight accordingly
		stepsize = (np.log(2 + i) ** a)/((1+i) ** (1 - b))
		stochweight = (np.log(2 + i) ** (a * s))/((1+i) ** (s * (1 - b)))
		erg = ((stepsize * x) + (Gamma * erg))/(Gamma + stepsize)
		feas[i] = np.linalg.norm(A.dot(erg)) ** 2
		dist[i] = np.linalg.norm(erg - xstar) ** 2
		#draw a batch randomly
		lucky_indices = np.random.choice(n, batchsize, replace=False)
		#compute a gradient according to lucky index
		newgrad = calc_bgradL(x, y, lucky_indices)
		#do the variance reduction recursion
		vrgrad = calc_vrgrad(vrgrad, newgrad, stochweight)
		#compute the other grad terms of aug lagrangian
		restgrad = calc_rest(x, mu, A, Ah, Rho)
		grad = vrgrad + restgrad
		#update primal var
		x = primal_step(x, grad, stepsize, delta)
		#update dual var
		mu = dual_step(x, mu, stepsize, A)
		#update sum of gammas
		Gamma += stepsize
	stepsize = (np.log(1 + itera) ** a)/((itera) ** (1 - b))
	erg = ((stepsize * x) + (Gamma * erg))/(Gamma + stepsize)
	feas[itera - 1] = np.linalg.norm(A.dot(erg)) **2
	dist[itera - 1] = np.linalg.norm(erg - xstar) ** 2
	return x, mu, erg, feas, dist
	
#icgalp with batch sampling of gradf_i, quadratic term, and variance reduction
def icgalp_vrfQb(y, A, batchsize, delta, s, b, a, Rho, itera, xstar):
	n = y.shape[0]
	x = np.zeros(y.shape)
	erg = np.zeros(y.shape)
	mu = np.zeros(A.shape[0])
	feas = np.zeros(itera)
	dist = np.zeros(itera)
	Gamma = 0
	Ah = A.transpose()
	#initialize the gradient as 0 for the recursion
	vrgrad = np.zeros(n)
	for i in pb.progressbar(range(itera - 1)):
		#set the step size and the stochastic weight accordingly
		stepsize = (np.log(2 + i) ** a)/((1+i) ** (1 - b))
		stochweight = (np.log(2 + i) ** (a * s))/((1+i) ** (s * (1 - b)))
		erg = ((stepsize * x) + (Gamma * erg))/(Gamma + stepsize)
		feas[i] = np.linalg.norm(A.dot(erg)) ** 2
		dist[i] = np.linalg.norm(erg - xstar) ** 2
		#draw a batch randomly
		lucky_indices = np.random.choice(n, batchsize, replace=False)
		#compute a gradient according to lucky index
		newgrad = calc_bgradLQ(x, y, lucky_indices, A, Ah, Rho)
		#do the variance reduction recursion
		vrgrad = calc_vrgrad(vrgrad, newgrad, stochweight)
		#compute the linear grad terms of aug lagrangian
		restgrad = Ah.dot(mu)
		grad = vrgrad + restgrad
		#update primal var
		x = primal_step(x, grad, stepsize, delta)
		#update dual var
		mu = dual_step(x, mu, stepsize, A)
		#update sum of gammas
		Gamma += stepsize
	stepsize = (np.log(1 + itera) ** a)/((itera) ** (1 - b))
	erg = ((stepsize * x) + (Gamma * erg))/(Gamma + stepsize)
	feas[itera - 1] = np.linalg.norm(A.dot(erg)) **2
	dist[itera - 1] = np.linalg.norm(erg - xstar) ** 2
	return x, mu, erg, feas, dist

#icgalp with batch sampling on gradE_i
def icgalp_vrb(y, A, batchsize, delta, s, b, a, Rho, itera, xstar):
	n = y.shape[0]
	x = np.zeros(y.shape)
	erg = np.zeros(y.shape)
	mu = np.zeros(A.shape[0])
	feas = np.zeros(itera)
	dist = np.zeros(itera)
	Gamma = 0
	Ah = A.transpose()
	#initialize grad at 0 for recursion
	vrgrad = np.zeros(n)
	for i in pb.progressbar(range(itera - 1)):
		#set the step size and the stochastic weight accordingly
		stepsize = (np.log(2 + i) ** a)/((1+i) ** (1 - b))
		stochweight = (np.log(2 + i) ** (a * s))/((1+i) ** (s * (1 - b)))
		erg = ((stepsize * x) + (Gamma * erg))/(Gamma + stepsize)
		feas[i] = np.linalg.norm(A.dot(erg)) ** 2
		dist[i] = np.linalg.norm(erg - xstar) ** 2
		#draw a batch randomly
		lucky_indices = np.random.choice(n, batchsize, replace=False)
		newgrad = calc_bgradE(x, mu, y, A, Ah, Rho, lucky_indices)
		#do the variance reduction recursion
		vrgrad = calc_vrgrad(vrgrad, newgrad, stochweight)
		#update primal var
		x = primal_step(x, vrgrad, stepsize, delta)
		#update dual var
		mu = dual_step(x, mu, stepsize, A)
		#update sum of gammas
		Gamma += stepsize
	stepsize = (np.log(1 + itera) ** a)/((itera) ** (1 - b))
	erg = ((stepsize * x) + (Gamma * erg))/(Gamma + stepsize)
	feas[itera - 1] = np.linalg.norm(A.dot(erg)) **2
	dist[itera - 1] = np.linalg.norm(erg - xstar) ** 2
	return x, mu, erg, feas, dist

#icgalp with sweeping on gradE_i
def icgalp_sw(y, A, delta, b, a, Rho, itera, xstar):
	n = y.shape[0]
	x = np.zeros(y.shape)
	erg = np.zeros(y.shape)
	mu = np.zeros(A.shape[0])
	feas = np.zeros(itera)
	dist = np.zeros(itera)
	Gamma = 0
	Ah = A.transpose()
	#initialize the gradient of E at x0, mu0 (it's just nabla_x f(x0))
	#old
	#grad = (x - y)/float(n)
	grad = np.zeros(y.shape)
	for i in pb.progressbar(range(itera - 1)):
		stepsize = (np.log(2 + i) ** a)/((1+i) ** (1 - b))
		erg = ((stepsize * x) + (Gamma * erg))/(Gamma + stepsize)
		feas[i] = np.linalg.norm(A.dot(erg)) ** 2
		dist[i] = np.linalg.norm(erg - xstar) ** 2
		#calculate the current index to be updated
		l = i % n
		#replace the l'th component of the grad with a newly calculated grad
		grad[l] = (x[l] - y[l])/float(n) + Ah.dot(mu)[l] + (Rho * Ah.dot(A.dot(x))[l])
		#update primal and dual variables
		x = primal_step(x, grad, stepsize, delta)
		mu = dual_step(x, mu, stepsize, A)
		Gamma += stepsize
	stepsize = (np.log(1 + itera) ** a)/((itera) ** (1 - b))
	erg = ((stepsize * x) + (Gamma * erg))/(Gamma + stepsize)
	feas[itera - 1] = np.linalg.norm(A.dot(erg)) **2
	dist[itera - 1] = np.linalg.norm(erg - xstar) ** 2
	return x, mu, erg, feas, dist

#the following two functions are from google
#they are used for l1 projection in the gfb implentation
def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = float(cssv[rho] - s) / (rho + 1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def euclidean_proj_l1ball(v, s=1):
    """ Compute the Euclidean projection on a L1-ball
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the L1-ball
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s
    Notes
    -----
    Solves the problem by a reduction to the positive simplex case
    See also
    --------
    euclidean_proj_simplex
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    u = np.abs(v)
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        return v
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = euclidean_proj_simplex(u, s=s)
    # compute the solution to the original problem on v
    w *= np.sign(v)
    return w
   
#generalized forward-backward, used to find the solution to high accuracy
def gfb(y, A, delta, itera):
	x = np.zeros(y.shape)
	u = np.zeros(y.shape)
	z = np.zeros(y.shape)
	gam = 1.9
	lam = 1
	pA = np.linalg.pinv(A).dot(A)
	for i in pb.progressbar(range(itera)):
		inneru = (2 * x) - z - (gam * (x - y))
		u = euclidean_proj_l1ball(inneru, s=delta)
		z = z + (lam * (u - x))
		x = z - pA.dot(z)
	return x
