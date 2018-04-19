from mdlp.discretization import MDLP
from sklearn.datasets import load_iris
transformer = MDLP()
iris = load_iris()

print(type(iris))
X, y = iris.data, iris.target

print(type(X))

print(X)
print(type(y))
print(y)
X_disc = transformer.fit_transform(X, y)

# print(X_disc)