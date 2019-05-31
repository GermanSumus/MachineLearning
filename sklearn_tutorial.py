# This line is for the interactive python console using jupyter: %matplotlib inline
# Helps to ensure our models[graphs, diagrams, visuals overall] to show
import seaborn # Helps prettify our models
seaborn.set() # Set seaborn plot defaults
from sklearn.datasets import load_iris
iris = load_iris()
print('Our dataset is represented in a Dictionary like object. As such it contains keys and values.\nPrint out the keys to see what kind of information our dataset contains.\n ')
print('>>> iris.keys()')
print(iris.keys(), '\n')
print('You can print out the information of the keys by using Dot(.) notation.')
print('See what kind of data we have by printing it out to the console.\n')
print('>>> iris.data[:5]')
print(iris.data[:5],'\n')
print('We see data contains a list of lists each with 4 floats.\nTo better understand what these floats mean lets look at the feature_name.\n')
print('>>> iris.feature_names')
print(iris.feature_names, '\n')
print('Each one of our floats from our data represent a value of a feature names.')
print('Below we have the shape of our dataset. A collection of 150 diffrent samples each with 4 unique features.\n>>> iris.data.shape')
n_samples, n_features = iris.data.shape
print((n_samples, n_features),'\n')
print('As you can see our data is in a format of a 2-D array. 150 sample each with 4 features.\nWe label our data with a 1-D array with a value for each sample.\n>>> iris.target.shape')
print(iris.target.shape)
print('Looking at our target for each sample in our dataset we get an array containing a value from 0-2.\nTo turn this numerical representation of labels to something more readable print out its target_names. .\n>>> iris.target')
print(iris.target,'\n>>> iris.target_names')
print(iris.target_names)
import numpy as np
import matplotlib.pyplot as plt

x_index = 2
y_index = 3

# This formatter will label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
# x, y, color, colormap
plt.scatter(iris.data[:, x_index], iris.data[:,y_index], c=iris.target, cmap=plt.get_cmap('RdYlBu', 3))
plt.colorbar(ticks=[0, 1, 2], format=formatter) # Set up the colorbar
plt.clim(-0.5, 2.5) # color limit range, min and max. Play with that to prefrence
plt.xlabel(iris.feature_names[x_index]) # label our x axis
plt.ylabel(iris.feature_names[y_index]) # and our y axis label
plt.show()

'''This is a diffrent learning problem below'''
from sklearn.linear_model import LinearRegression
