"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
import deepxde as dde
import numpy as np
import math


def func(x):
    """
    x: array_like, N x D_in
    y: array_like, N x D_out
    """
    return np.tan(x)


geom = dde.geometry.Interval(0,2*math.pi)
num_train = 16
num_test = 100
data = dde.data.Function(geom, func, num_train, num_test)
print(data)

activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN([1] + [20] * 3 + [1], activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=10000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True,output_dir='outputs')
