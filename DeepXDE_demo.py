"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
import deepxde as dde
import numpy as np
import sympy as sp
import math
import sys



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
