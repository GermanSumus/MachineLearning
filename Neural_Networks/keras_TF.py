import tensorflow as tf
import numpy as np
print(f'\nTensorFlow version: {tf.VERSION}')
print(f'Keras version: {tf.keras.__version__}\n')

# 28x28 images of hand written digits 0-9
digits = tf.keras.datasets.mnist

# Unpack the data
(x_train, y_train), (x_test, y_test) = digits.load_data()

# Scale data: Normalization(values are converted to be between 0 & 1)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Sequential type model
model = tf.keras.models.Sequential()

# Input Layer
model.add(tf.keras.layers.Flatten(input_shape = (28, 28)))

# Hidden Layers(2):128 neuron each with Rectified Linear("Default")
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

# Output Layer: 0-9 as our choices(10) with Probability Distribution
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# Paramaters for the training of the model
model.compile(optimizer='adam', # "Default"
			  loss='sparse_categorical_crossentropy',# Cost(degree of erroe)
			  metrics=['accuracy'] # Metrics to track
			  )

# Fit model
print('Training the neural network...\n')
model.fit(x_train, y_train, epochs=3)
print('\nTraining Complete!\n')

# Evaluate the model with validation tests
print('Evaluation testing...\n')
validation_loss, validation_accuracy = model.evaluate(x_test, y_test)
print(f'Degree of error: {validation_loss}\n')
print(f'Accuracy of prediction: {validation_accuracy}\n')

# Save the learning model
print('Example of a saved then loaded model...\n')
model.save('number_rocognizer.model')

# Load a saved model
new_model = tf.keras.models.load_model('number_rocognizer.model')

# Example with new_model
import matplotlib.pyplot as plt
import numpy as np

# Predicted values of new_model
predictions = new_model.predict([x_test])

# Show a prediction with argmax(return the highest digit index; 0-9)
print('This is what the predictions look like:\n')
print(predictions[0])
print(f'\nNot very readable, but you can see the seventh index value is .999 in decimal notation.\nYou can easily retrieve the index of the highest number in a array with argmax.\n')
print(f'Predicted out come of the first value: {np.argmax(predictions[0])}\n')

# Show digit in matplotlib
plt.imshow(x_test[0])
print('Refer to image to see if the model predicted correctly.')
plt.show()
