import numpy as np
import matplotlib.pyplot as plt
from gaussian_processes.utilities.posterior import compute_posterior_mean_covariance

# Inputs
x_data = np.array([-2.0, -1.5, -1, -0.5, 0, 0.5]).reshape(-1, 1)
n = x_data.shape[0]

# Outputs
y_data = np.array([-1.8, -1.6, -1.1, -0.5, 0.2, 0.7]).reshape(-1, 1)

# Plotting data
plt.figure()
plt.plot(x_data, y_data, 'k.')
plt.xlabel('x')
plt.ylabel('y')


# Interpolation and extrapolation points
# xstar > 0.5 # Extrapolation

# New data points
x_star = np.arange(-2.0, 1.01, 0.1).reshape(-1, 1)  # Adjust the range and step for better visualization
N = x_star.shape[0]

# Computing covariance natrix block structure (before Kernel) 
XX_ind1, XX_ind2 = np.meshgrid(x_data, x_data)
XXs_ind1, XXs_ind2 = np.meshgrid(x_data, x_star)
XsXs_ind1, XsXs_ind2 = np.meshgrid(x_star, x_star)

# Judicious Hyperparameters for the Kernel Function
l = 1 # Lengthscale
sig_f = np.sqrt(3) # Signal variance

# Computing covariance matrices through SE Kernel
cov_XX = sig_f**2 * np.exp(-0.5 * (XX_ind1 - XX_ind2)**2 / l**2)
cov_XsXs = sig_f**2 * np.exp(-0.5 * (XsXs_ind1 - XsXs_ind2)**2 / l**2)
cov_XXs = sig_f**2 * np.exp(-0.5* (XXs_ind1 - XXs_ind2)**2 / l**2)

# Adding noise to covariance matrix, let's try less?
sig_n = 0.8
cov_XX_noisy = cov_XX + sig_n ** 2 * np.eye(n)

# Plotting covariance matrix with and without noise
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left subplot
im0 = axes[0].imshow(cov_XX_noisy, cmap='viridis')  # You can choose a colormap that suits your data
axes[0].set_title('Noisy Covariance Matrix')
axes[0].set_axis_off()  # Optional: Turn off axis labels and ticks

# Plot the second image on the right subplot
im1 = axes[1].imshow(cov_XX, cmap='viridis')  # You can choose a colormap that suits your data
axes[1].set_title('Non Noisy Covariance Matrix')
axes[1].set_axis_off()  # Optional: Turn off axis labels and ticks

# Display the colorbars if needed
cbar0 = fig.colorbar(im0, ax=axes[0])
cbar1 = fig.colorbar(im1, ax=axes[1])

# Adjust layout to prevent clipping of titles
plt.tight_layout()

# Computing posterior mean and covariance without noise
posterior_mean, posterior_cov  = compute_posterior_mean_covariance(cov_XX, cov_XXs, cov_XsXs, y_data)
# Computing posterior mean and covariance with noise
posterior_mean_noisy, posterior_cov_noisy  = compute_posterior_mean_covariance(cov_XX_noisy, cov_XXs, cov_XsXs, y_data)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the mean and variance of the posterior without noise
upper_bound = posterior_mean.flatten() + 2 * np.sqrt(np.diag(posterior_cov))
lower_bound = posterior_mean.flatten() - 2 * np.sqrt(np.diag(posterior_cov))

axes[0].fill_between(x_star.flatten(), upper_bound, lower_bound, color=[7/8, 7/8, 7/8])
axes[0].plot(x_star, posterior_mean, 'b-', linewidth=2)
axes[0].plot(x_data, y_data, 'k.')
axes[0].set_title('Without noise')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')


# Plot the mean and variance of the posterior with noise
upper_bound_noisy = posterior_mean_noisy.flatten() + 2 * np.sqrt(np.diag(posterior_cov_noisy))
lower_bound_noisy = posterior_mean_noisy.flatten() - 2 * np.sqrt(np.diag(posterior_cov_noisy))

axes[1].fill_between(x_star.flatten(), upper_bound_noisy, lower_bound_noisy, color=[7/8, 7/8, 7/8])
axes[1].plot(x_star, posterior_mean_noisy, 'b-', linewidth=2)
axes[1].plot(x_data, y_data, 'k.')
axes[1].set_title('With noise')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')

# Generating random function samples from posterior

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

jitter = 10 ** (-6)

# Without noise
L = np.linalg.cholesky(posterior_cov + jitter * np.eye(N))
random_functions = posterior_mean + L.T @ np.random.randn(N, 5)
axes[0].fill_between(x_star.flatten(), upper_bound, lower_bound, color=[7/8, 7/8, 7/8])
axes[0].plot(x_star, random_functions,'.--')
axes[0].set_title('Without noise')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')


# Wiht noise
L = np.linalg.cholesky(posterior_cov_noisy + jitter * np.eye(N))
random_functions = posterior_mean_noisy + L.T @ np.random.randn(N, 5)

axes[1].fill_between(x_star.flatten(), upper_bound_noisy, lower_bound_noisy, color=[7/8, 7/8, 7/8])
axes[1].plot(x_star, random_functions,'.--')
axes[1].set_title('With noise')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')

plt.show()