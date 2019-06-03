import numpy as np
import matplotlib.pyplot as plt

# Sample data from sklearn generator
from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=50, centers=2,
                  random_state=0, cluster_std=0.60)

# Plot onto matplotlib
plt.scatter(X[:, 0],  X[:, 1],
            c=y, s=25,  cmap='spring')
plt.show()

from sklearn.svm import SVC  # "Support Vector Classifier"
clf = SVC(kernel='linear')
clf.fit(X, y) # fit the model

# Makes a visual of the seperation choice somehow. Not often used after 3 vectors
def plot_svc_decision_function(clf, ax=None):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca() # assign graph axis to a variable

    x = np.linspace(start=plt.xlim()[0], stop=plt.xlim()[1], num=30)
    y = np.linspace(start=plt.ylim()[0], stop=plt.ylim()[1], num=30)

    Y, X = np.meshgrid(y, x)
    P = np.zeros_like(X)

    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            P[i, j] = clf.decision_function([[xi, yj]])
    # plot the margins
    ax.contour(X, Y, P, colors='k', linestyles=['--', '-', '--'],
               levels=[-1, 0, 1], alpha=.5)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring')
plot_svc_decision_function(clf)
plt.show()

"""
Non linear separable data.
"""
from sklearn.datasets.samples_generator import make_circles
X, y = make_circles(100, factor=.1, noise=.1)

clf = SVC(kernel='linear').fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring')
plot_svc_decision_function(clf)
plt.show()

clf = SVC(kernel='rbf', gamma='scale').fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring')
plot_svc_decision_function(clf)
plt.show()
