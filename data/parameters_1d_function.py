
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """ function to approximate by polynomial interpolation"""
    return x * np.sin(x)

seed = 1234 # set a random seed to replicate results
np.random.seed(seed)

#
n_train_points = 8 # number of training points (use at least 6)
n_test_points = 100 # number of test points

# generate points used to plot
x_plot = np.linspace(0, 10, 1000)

# generate points and keep a subset of them
x = np.linspace(0, 10, 1000)
rng = np.random.RandomState(seed)
rng.shuffle(x)

x_train = np.sort(x[:n_train_points])
x_test = np.sort(x[n_train_points+1:n_train_points+n_test_points+1])

# Calculate the y values for train and test sets:
y_train = f(x_train) # training data
y_test = f(x_test) # test data (unseen)

# create matrix versions of these arrays
X_train = x_train[:, np.newaxis]
X_test = x_test[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]

fig, axs = plt.subplots(1)
# Plot training points
plt.plot(x_train, y_train, 'ro', markersize=6, label="training points")
plt.plot(x_plot, f(x_plot),  color='black', linewidth=2, linestyle='-', label="signal")
# Plot test points
plt.plot(x_test, y_test, 'bx', markersize=5, label="testing points")
plt.legend()

