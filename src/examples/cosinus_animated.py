import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.layers.Dense import Dense
from src.losses.Losses import mse, mse_prime
from src.activation.Activations import Sigmoid, Tanh
import matplotlib
matplotlib.use('TkAgg')

X = np.linspace(-2 * np.pi, 2 * np.pi, 500).reshape(-1, 1)
Y = np.cos(X)


network = [
	Dense(1, 30),
	Sigmoid(),
	Dense(30, 20),
	Tanh(),
	Dense(20, 10),
	Sigmoid(),
	Dense(10, 1)
]

epochs = 1000
learning_rate = 0.02
errors = []
X_test = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)


def get_approximation():
	cosx = []
	for x in X_test:
		x = x.reshape(1, -1)
		output = x
		for layer in network:
			output = layer.forward(output)
		cosx.append(output)
	return np.array(cosx).reshape(-1)



fig, ax = plt.subplots()
ax.set_xlim(-2 * np.pi, 2 * np.pi)
ax.set_ylim(-1.5, 1.5)
ax.plot(X, Y, label='cos(x)', linestyle='dashed', color='gray')
(line,) = ax.plot(X_test, np.zeros_like(X_test), label='NN Approximation', color='red')
epoch_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=12)
ax.legend()


def update(epoch):
	global network
	error = 0
	for x, y in zip(X, Y):
		x = x.reshape(1, -1)
		output = x
		for layer in network:
			output = layer.forward(output)
		error += mse(output, y)
		grad = mse_prime(y, output)
		for layer in reversed(network):
			grad = layer.backward(grad, learning_rate)

	error /= len(X)
	errors.append(error)

	line.set_data(X_test, get_approximation())
	epoch_text.set_text(f'Epoch: {epoch + 1}/{epochs}')
	return line,


ani = animation.FuncAnimation(fig, update, frames=epochs, interval=10, blit=False)
plt.ion()
plt.show()
plt.show(block=True)