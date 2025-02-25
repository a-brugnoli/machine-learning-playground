import smt
import numpy as np
import matplotlib.pyplot as plt

from data.parameters_1d_function import *

from smt.surrogate_models import KRG

#sm = KRG(theta0=[1e-2], poly='constant', corr='squar_exp')
sm = KRG(theta0=[1e-2], poly='constant', corr='matern52')
sm.set_training_values(x_train, y_train)
sm.train()


y_pred = sm.predict_values(x_plot)
# estimated variance
s2 = sm.predict_variances(x_plot)

fig, axs = plt.subplots(1)

# add a plot with variance
axs.plot(x_train, y_train, "ro")
axs.plot(x_plot, y_pred)
axs.plot(x_plot, f(x_plot))
axs.fill_between(
    np.ravel(x_plot),
    np.ravel(y_pred - 3 * np.sqrt(s2)),
    np.ravel(y_pred + 3 * np.sqrt(s2)),
    color="lightgrey",
)
axs.set_xlabel("x")
axs.set_ylabel("y")
axs.legend(
    ["Training data", "Prediction","ground truth: $f(x) = x\,\sin(x)$","Confidence Interval 99%"],
    loc="lower right",
)


# Prediction of the validation points
y = sm.predict_values(x_test)
#print('LS,  err: '+str(compute_rms_error(y,x_test,y_test)))

# Plot prediction/true values

fig = plt.figure()
plt.plot(y_test, y_test, '-', label='$y_{true}$')
plt.plot(y_test, y, 'r.', label='$\hat{y}$')
       
plt.xlabel('$y_{true}$')
plt.ylabel('$\hat{y}$')
        
plt.legend(loc='upper left')
#plt.title('LS model: validation of the prediction model')

#Q1: vizualise the residual, is the Kriging interpolant or regressant?
#play with all available Kernels


from sklearn.metrics import mean_squared_error, r2_score


# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y))
# Explained variance score: 1 is perfect prediction
print('Variance score (R2): %.2f' % r2_score(y_test, y))

plt.show()


from smt.surrogate_models import LS


sm = LS()
sm.set_training_values(x_train, y_train)
sm.train()


y_pred = sm.predict_values(x_plot)
fig, axs = plt.subplots(1)
# add a plot with variance
axs.plot(x_train, y_train, "ro")
axs.plot(x_plot, y_pred)
axs.plot(x_plot, f(x_plot))

axs.set_xlabel("x")
axs.set_ylabel("y")
axs.legend(
    ["Training data", "Prediction","ground truth: $f(x) = x\,\sin(x)$"],
    loc="lower right",
)

# Prediction of the validation points
y = sm.predict_values(x_test)
#print('LS,  err: '+str(compute_rms_error(y,x_test,y_test)))

# Plot prediction/true values

fig = plt.figure()
plt.plot(y_test, y_test, '-', label='$y_{true}$')
plt.plot(y_test, y, 'r.', label='$\hat{y}$')
       
plt.xlabel('$y_{true}$')
plt.ylabel('$\hat{y}$')
        
plt.legend(loc='upper left')
#plt.title('LS model: validation of the prediction model')

from sklearn.metrics import mean_squared_error, r2_score


# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y))
# Explained variance score: 1 is perfect prediction
print('Variance score (R2): %.2f' % r2_score(y_test, y))

plt.show()


from smt.surrogate_models import QP


sm = QP()
sm.set_training_values(x_train, y_train)
sm.train()


y_pred = sm.predict_values(x_plot)
fig, axs = plt.subplots(1)
# add a plot with variance
axs.plot(x_train, y_train, "ro")
axs.plot(x_plot, y_pred)
axs.plot(x_plot, f(x_plot))

axs.set_xlabel("x")
axs.set_ylabel("y")
axs.legend(
    ["Training data", "Prediction","ground truth: $f(x) = x\,\sin(x)$"],
    loc="lower right",
)

# Prediction of the validation points
y = sm.predict_values(x_test)
#print('LS,  err: '+str(compute_rms_error(y,x_test,y_test)))

# Plot prediction/true values

fig = plt.figure()
plt.plot(y_test, y_test, '-', label='$y_{true}$')
plt.plot(y_test, y, 'r.', label='$\hat{y}$')
       
plt.xlabel('$y_{true}$')
plt.ylabel('$\hat{y}$')
        
plt.legend(loc='upper left')
#plt.title('LS model: validation of the prediction model')

from sklearn.metrics import mean_squared_error, r2_score


# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y))
# Explained variance score: 1 is perfect prediction
print('Variance score (R2): %.2f' % r2_score(y_test, y))

#Q2: vizualise the residual, is something missing?

plt.show()