# from mdlp.discretization import MDLP
# from sklearn.datasets import load_iris
# transformer = MDLP()
# iris = load_iris()
#
# print(type(iris))
# X, y = iris.data, iris.target
#
# print(type(X))
#
# print(X)
# print(type(y))
# print(y)
# X_disc = transformer.fit_transform(X, y)

# print(X_disc)

# import numpy as np
#
# X_t = [[1,2,3],[4,5,6]]
# X = np.array([[1,2,3], [4,5,6]])
#
# print(type(X))
#
# print(X)
#
# np.random.shuffle(X[:, 2])
#
# print(X)

# a = [15, 2, 3]
# a.remove(15)
#
# print(a)

import pandas as pd

df = pd.DataFrame(data={'a': [1,2,3],'b': [5,6,7]})

print(df.dtypes)