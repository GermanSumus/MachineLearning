import seaborn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
seaborn.set()

iris = load_iris()
print(iris.keys())
print(iris.data[:5])
print(iris.feature_names)
n_samples, n_features = iris.data.shape
print((n_samples, n_features))
print(iris.target.shape)
print(iris.target)
print(iris.target_names)

x_index = 2
y_index = 3
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
plt.scatter(iris.data[:, x_index], iris.data[:,y_index], c=iris.target, cmap=plt.get_cmap('RdYlBu', 3))
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.clim(-0.5, 2.5)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
plt.show()


from sklearn.linear_model import LinearRegression
