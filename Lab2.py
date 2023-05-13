import numpy as np
from sklearn.decomposition import TruncatedSVD, PCA

# create a matrix
matrix = np.array([[1, 2], [3, 4], [5, 6]])

# perform SVD
svd = TruncatedSVD(n_components=1)
svd.fit(matrix)
svd_matrix = svd.transform(matrix)

# perform PCA
pca = PCA(n_components=1)
pca.fit(matrix)
pca_matrix = pca.transform(matrix)

print("Original matrix:\n", matrix)
print("SVD matrix:\n", svd_matrix)
print("PCA matrix:\n", pca_matrix)

