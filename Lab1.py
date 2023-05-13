import numpy as np

# Define the data points
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Define the relation matrix
relation = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])

# Perform matrix multiplication to compute the result
result = np.dot(relation, data)

# Display the result
print(result)