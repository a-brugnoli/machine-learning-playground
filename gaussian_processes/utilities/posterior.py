# Computing posterior mean and covariance
import numpy as np

def compute_posterior_mean_covariance(cov_XX, cov_XXs, cov_XsXs, y_data):
    # inv_cov_XX = np.linalg.inv(cov_XX)
    posterior_mean = cov_XXs @ np.linalg.solve(cov_XX, y_data)
    posterior_cov  = cov_XsXs - cov_XXs @ np.linalg.solve(cov_XX, cov_XXs.T) 
    return posterior_mean, posterior_cov

