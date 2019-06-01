import seaborn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
seaborn.set()

# More Complex Data
np.random.seed(0)
X = np.random.random(size=(20, 1))

# reduce X by one dimesion with squeeze() to keep target as a single dimesion
y = 3 * X.squeeze() + 2 + np.random.randn(20)

plt.plot(X.squeeze(), y, 'o')
plt.show()
"""Pop-up window"""

# LinearRegression Model
model = LinearRegression()
model.fit(X, y)

# plot the data and the model prediction
x_fit = np.linspace(0, 1, 100)[:, np.newaxis]
y_fit = model.predict(x_fit)
plt.plot(X.squeeze(), y, 'o')
plt.plot(x_fit.squeeze(), y_fit)
plt.show()
"""Pop-up window"""

# Copy paste the code from above and change the model type
model = RandomForestRegressor()
model.fit(X, y)

# plot the data and the model prediction
x_fit = np.linspace(0, 1, 100)[:, np.newaxis]
y_fit = model.predict(x_fit)
plt.plot(X.squeeze(), y, 'o')
plt.plot(x_fit.squeeze(), y_fit)
plt.show()
"""Pop-up window"""
