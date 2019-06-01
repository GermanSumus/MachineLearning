import seaborn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
seaborn.set()

# Iris Dataset
iris = load_iris()
print(iris.keys())
print(iris.data[:5])
print(iris.feature_names)
n_samples, n_features = iris.data.shape
print((n_samples, n_features))
print(iris.target.shape)
print(iris.target)
print(iris.target_names)

# Iris Dataset Visualized In Matplotlib
# adjust the two indexes to see different results
x_index = 2
y_index = 3
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
plt.scatter(iris.data[:, x_index], iris.data[:,y_index], c=iris.target, cmap=plt.get_cmap('RdYlBu', 3))
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.clim(-0.5, 2.5)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
plt.show()


# Machine Learning Example
# selecting a learning problem
from sklearn.linear_model import LinearRegression

# estimator paramaters are set up when it is instantiated
model = LinearRegression(normalize=True)

# some example data quickly made
x = np.arange(10)
y = 2 * x + 1
print(x)
print(y)

# visualize in matplotlib
plt.plot(x, y, 'o')
plt.yticks(np.arange(0, (y.max() / 5 + 1) * 5, 5))
plt.xticks(np.arange(0, 11, 1))
plt.grid(False)
plt.show()

# Format Data For Sklearn
# np.newaxis will increase the dimension of the array by 1
X = x[:, np.newaxis]
print(X)
print(y)
print(model.fit(X, y))

# Attributes ending with a '_' indicate a fit parameter
print(model.coef_)
print(model.intercept_)
print(model._residues)
