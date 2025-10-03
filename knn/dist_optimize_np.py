'''
Optimized distance matrix finding for KNN
'''
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import timeit
iris = load_iris()
X = iris.data 
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,
    random_state=42,
)
def no_loops_dist(X_train, X_test):
    dists = np.sqrt(
            np.sum(X_test**2, axis=1, keepdims=True) +
            np.sum(X_train**2, axis=1) -
            2 * X_test @ X_train.T
        )
    return dists
execution_time_no_loops = timeit.timeit(
    lambda: no_loops_dist(X_train, X_test), 
    number=100
)
def two_loops(X_train, X_test):
  num_test = X_test.shape[0]
  num_train = X_train.shape[0] 
  dists = np.zeros((num_test, num_train))
  for i in range(num_test):
    for j in range(num_train):
      dists[i, j] = np.sqrt(np.sum((X_test[i] - X_train[j])**2))
  return dists

execution_time_two_loops = timeit.timeit(
    lambda: two_loops(X_train, X_test), 
    number=100
)
print(f"Program execution time without cycles {execution_time_no_loops}", f"Program execution time with cycles {execution_time_two_loops}")

