from src.activation.Activation import Activation
import numpy as np


class Tanh(Activation):
	def __init__(self):
		tanh = lambda x: np.tanh(x)
		tanh_prime = lambda x: 1 - np.tanh(x)**2
		super().__init__(tanh, tanh_prime)


class Sigmoid(Activation):
	def __init__(self):
		def sigmoid(x):
			return 1/(1+np.exp(-x))
		def sigmoid_prime(x):
			s = sigmoid(x)
			return s * (1-s)
		super().__init__(sigmoid, sigmoid_prime)


class ReLU(Activation):
	def __init__(self):
		def relu(x):
			return x * (x > 0)
		def relu_prime(x):
			return np.where(x > 0, 1.0, 0.0)
		super().__init__(relu, relu_prime)
