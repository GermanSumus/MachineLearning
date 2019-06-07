import matplotlib.pyplot as plt
import numpy as np

# Activator function
def sigmoid(x):
	return 1/(1 + np.exp(-x))

# Derivitive of the activator function
# Instantaneous rate of change at a given point
def sigmoid_derivative(x):
	return x * (1 - x)


training_set = np.array([[0,0,1],
						 [1,1,1],
						 [1,0,1],
						 [0,1,1]])

# Transpose target_set to rows instead of columns
target_set = np.array([[0,1,1,0]]).T

# Seed the random generator
np.random.seed(1)

weights = 2 * np.random.random((3,1)) - 1

print('Random weight starting point:' )
print(weights)

for iteration in range(20000):
	input_layer = training_set

	output_layer = sigmoid(np.dot(input_layer, weights))

	cost = target_set - output_layer

	adj = cost * sigmoid_derivative(output_layer)

	weights += np.dot(input_layer.T, adj)

print('Output choice is:')
print(output_layer)
