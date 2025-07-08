import numpy as np
from scipy.linalg import eigvals

def extract_features(window: np.ndarray) -> np.ndarray:
    """
        a helper function to extract features from the given window
        Features are: (X, Y, R, G, B)
    """
    H, W, _ = window.shape
    Y, X = np.mgrid[0:H, 0:W]
    X = X.flatten()
    Y = Y.flatten()
    R = window[:, :, 0].flatten()
    G = window[:, :, 1].flatten()
    B = window[:, :, 2].flatten()
    return np.stack([X, Y, R, G, B], axis=1)

def riemannian_distance(C1, C2):
    """
        Compute the Riemannian distance between two covariance matrices C1 and C2
    """
    eigenvalues = eigvals(C1, C2)
    
    # filter real, non-positive eigenvalues
    eigenvalues = np.real(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 0]

    # finally compute the distance
    EPSILON = 1e-6
    log_eigs = np.log(eigenvalues + EPSILON)
    return np.sqrt(np.sum(log_eigs**2))
