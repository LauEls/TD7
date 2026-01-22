import numpy as np

def fit_ellipsoid_pca(points):
    # Center the data
    offset = np.mean(points, axis=0)
    xc = points - offset

    # Calculate the covariance matrix
    cov_matrix = np.cov(xc, rowvar=False)

    # Perform Singular Value Decomposition (SVD) on the covariance matrix
    # The eigenvectors (Vh.T) give the orientation
    # The eigenvalues relate to the axis lengths
    U, S, Vh = np.linalg.svd(cov_matrix)
    
    # S contains the eigenvalues. The square roots of these are proportional to 
    # the standard deviations along the principal axes. For a bounding ellipsoid, 
    # you can use a scaling factor (e.g., 2 or 3 for 2 or 3 standard deviations).
    # The axis lengths (radii) are derived from S.
    # Note: A simple heuristic for axis lengths might be needed depending on data distribution.
    radii = np.sqrt(S) * 2. # Example scaling

    # The orientation matrix is the transpose of Vh (or just Vh if you use the U, S, Vh output of numpy's svd)
    orientation = Vh.T

    return offset, radii, orientation

# Example usage with dummy points
# Generate some points from a known ellipsoid for testing
# You would replace this with your actual data
# from sklearn.datasets import make_spd_matrix
np.random.seed(42)
center_true = np.array([10, 20, 30])
radii_true = np.array([5, 2, 3])
# Generate data: a more involved process to generate points *on* or *within* an ellipsoid
# For demonstration, use a generic point cloud:
points = np.random.rand(100, 3) * 5 + center_true

center, radii, orientation = fit_ellipsoid_pca(points)

print(f"Center: {center}")
print(f"Radii: {radii}")
print(f"Orientation Matrix:\n{orientation}")
