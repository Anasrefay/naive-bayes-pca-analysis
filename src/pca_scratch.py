import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.std = None
        self.explained_variance_ratio_ = None
    def fit(self, X):

        # Step 1: standardize (calculate mean and std for each feature and standardize the data)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        std_safe = np.where(self.std == 0, 1.0, self.std)
        X_stand = (X - self.mean) / std_safe

        # Step 2: calculate the covariance matrix
        cov_matrix = np.cov(X_stand, rowvar=False)

        # Step 3: get the eigenvectors and eigenvalues
        eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)

        # Step 4: sort the eigen vectors by the eigen values in a descending order
        sorted_index = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[sorted_index]
        eigen_vectors = eigen_vectors[:, sorted_index]

        # Step 5: select the top k components
        self.components = eigen_vectors[:, :self.n_components]

        # Step 6: calculate the explained variance ratio for scree plot
        self.explained_variance_ratio_ = eigen_values / np.sum(eigen_values)

    def transform(self, X):
        std_safe = np.where(self.std == 0, 1.0, self.std)
        X_stand = (X - self.mean) / std_safe
        return np.dot(X_stand,self.components)
