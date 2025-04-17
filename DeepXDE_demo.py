"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
import deepxde as dde
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import sympy as sp
import math
import sys



def rectangle_sheet_de():
	# Parameters
	q_bar = 50
	# Defining PDE
	def dy(x, y):
		return dde.grad.jacobian(y, x, i=0, j=1)
	def dx(x, y):
		return dde.grad.jacobian(y, x, i=0, j=0)
	def pde(x, w):
		dw_xx = dde.grad.hessian(w, x, i=0, j=0)
		dw_xxxx = dde.grad.hessian(dw_xx, x, i=0, j=0)
		dw_yy = dde.grad.hessian(w, x, i=1, j=1)
		dw_yyyy = dde.grad.hessian(dw_yy, x, i=1, j=1)
		dw_xxyy = dde.grad.hessian(dw_xx, x, i=1, j=1)
		T0 = 1
		T1 = 2
		T2 = 1
		return ( T0 * dw_xxxx + T1 * dw_xxyy + T2 * dw_yyyy) + (q_bar)
	# Boundary conditions
	def boundary_left(x, on_boundary):
		return on_boundary and np.isclose(x[0], 0)
	def boundary_right(x, on_boundary):
		return on_boundary and np.isclose(x[0], 1)
	def boundary_up(x, on_boundary):
		return on_boundary and np.isclose(x[1], 0)
	def boundary_down(x, on_boundary):
		return on_boundary and np.isclose(x[1], 1)


	geom = dde.geometry.Rectangle([0, 0], [1, 1])
	# clamped supported boundaries
	#bc1 = dde.DirichletBC(geom, lambda x: 0, boundary_left)
	bc2 = dde.OperatorBC(geom, lambda x, y, _: dx(x, y), boundary_left)
	#bc3 = dde.DirichletBC(geom, lambda x: 0, boundary_right)
	bc4 = dde.OperatorBC(geom, lambda x, y, _: dx(x, y), boundary_right)
	#bc5 = dde.DirichletBC(geom, lambda x: 0, boundary_up)
	bc6 = dde.OperatorBC(geom, lambda x, y, _: dy(x,y), boundary_up)
	#bc7 = dde.DirichletBC(geom, lambda x: 0, boundary_down)
	bc8 = dde.OperatorBC(geom, lambda x, y, _: dy(x,y), boundary_down)
	# Data
	data = dde.data.PDE(
	geom, pde, [ bc2, bc4, bc6, bc8],
	num_domain=500, num_boundary=180, train_distribution="uniform",
	num_test=500,
	)
	# Neural network
	layer_size = [2] + [20] * 4 + [1]
	activation = "tanh"
	initializer = "Glorot uniform"
	net = dde.maps.FNN(layer_size, activation, initializer)
	def transform(x, y):
		res = x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * (1 - x[:, 1:2])
		return res * y
	net.apply_output_transform(transform)
	model = dde.Model(data, net)
	model.compile("adam", lr=0.001)
	# Training
	losshistory, train_state = model.train(epochs=1000)
	# Save and plot
	dde.saveplot(losshistory, train_state, issave=True, isplot=True,
	loss_fname='loss.dat', train_fname='train.dat', test_fname='test.dat',output_dir='outputs')



def tan():
	def func(x):
		"""
		x: array_like, N x D_in
		y: array_like, N x D_out
		"""
		return np.tan(x)



	# Define variables
	x, t = sp.symbols('x t')
	w = sp.Function('w')(x, t)
	q = sp.Function('q')(x, t)

	# Define constants
	E, I, rho, A, e0, a, KG, KAG = sp.symbols('E I rho A e0 a KG KAG')
	EI = E * I
	rhoI = rho * I

	# Define derivatives
	d4w_dx4 = sp.diff(w, x, 4)
	d2w_dt2 = sp.diff(w, t, 2)
	d2_dx2 = sp.diff(sp.diff(1, x), x)  # placeholder for second derivative operator
	d2w_dx2 = sp.diff(w, x, 2)
	d4w_dx2dt2 = sp.diff(w, x, 2, t, 2)
	d4w_dt4 = sp.diff(w, t, 4)
	d2q_dx2 = sp.diff(q, x, 2)
	d2q_dt2 = sp.diff(q, t, 2)
	d4q_dx2dt2 = sp.diff(q, x, 2, t, 2)

	# Equation construction
	term1 = EI * d4w_dx4
	term2 = rho * A * (1 - (e0 * a)**2 * sp.diff(sp.diff(1, x), x)) * d2w_dt2
	term3 = -(1 - (e0 * a)**2 * sp.diff(sp.diff(1, x), x)) * (rho * I + (rho * EI / (KAG))) * d4w_dx2dt2
	term4 = (rho**2 * I / KG) * ((1 - (e0 * a)**2 * sp.diff(sp.diff(1, x), x))**2) * d4w_dt4
	term5 = (1 - (e0 * a)**2 * sp.diff(sp.diff(1, x), x)) * q
	term6 = (rho * I / KAG) * (1 - (e0 * a)**2 * sp.diff(sp.diff(1, x), x))**2 * d2q_dt2
	term7 = (EI / (KAG)) * (1 - (e0 * a)**2 * sp.diff(sp.diff(1, x), x)) * d2q_dx2

	# Final equation
	lhs = term1 + term2 + term3 + term4
	rhs = term5 + term6 - term7

	equation = sp.Eq(lhs, rhs)
	sp.pprint(equation)

	# sys.exit('53')



	geom = dde.geometry.Interval(0,2*math.pi)
	num_train = 16
	num_test = 100
	data = dde.data.Function(geom, equation, num_train, num_test)
	print(data)

	activation = "tanh"
	initializer = "Glorot uniform"
	net = dde.nn.FNN([1] + [20] * 3 + [1], activation, initializer)

	model = dde.Model(data, net)
	model.compile("adam", lr=0.001, metrics=["l2 relative error"])
	losshistory, train_state = model.train(iterations=10000)

	dde.saveplot(losshistory, train_state, issave=True, isplot=True,output_dir='outputs')



rectangle_sheet_de()