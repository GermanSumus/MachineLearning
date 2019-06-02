from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

digits = datasets.load_digits()
print(digits.keys())

# the images themselves
print(digits.images.shape)
print(digits.images[0])

# the data for use in our algorithms
print(digits.data.shape)
print(digits.data[0])

# the targets labels
print(digits.target)

# plot a portion of the dataset in matplotlib
fig, axes = plt.subplots(10, 10, figsize=(8, 8))
fig.subplots_adjust(hspace=0.2, wspace=0.2)

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary')
    ax.text(0.05, 0.05, str(digits.target[i]), transform=ax.transAxes, color='green')
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

# Dimensionality Reduction
from sklearn.manifold import Isomap

iso = Isomap(n_components=2)
data_projected = iso.fit_transform(digits.data)
plt.show()
print(data_projected.shape)

# plot our new data in matplotlib
plt.scatter(data_projected[:, 0], data_projected[:,1], c=digits.target, edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.clim(-0.05, 9.5)
plt.colorbar().set_ticks(np.arange(0,10,1))
plt.show()

# Classification on Digits
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target, random_state=2)

# two diffrent sets one for training and one for testing
print(Xtrain.shape, Xtest.shape)

# train a learning problem model
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver='lbfgs', multi_class='auto')
clf.fit(Xtrain, ytrain)
yPrediction = clf.predict(Xtest)

# test accuracy of model
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest, yPrediction))

# more insight
from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest, yPrediction))

# a visual of what it predicted
plt.imshow(np.log(confusion_matrix(ytest, yPrediction)), cmap='binary', interpolation='nearest')
plt.grid(False)
plt.ylabel('true')
plt.xlabel('predicted')
plt.show()

# plot a portion of the dataset in matplotlib
fig, axes = plt.subplots(10, 10, figsize=(8, 8))
fig.subplots_adjust(hspace=0.2, wspace=0.2)

# show the image with the predicted label
for i, ax in enumerate(axes.flat):
    ax.imshow(Xtest[i].reshape(8, 8), cmap='binary')
    ax.text(0.05, 0.05, str(yPrediction[i]), transform=ax.transAxes, color='green' if (ytest[i] == yPrediction[i]) else 'red')
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
