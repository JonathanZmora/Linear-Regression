import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from LinearRegression import LinearRegression

np.set_printoptions(suppress=True, precision=4)

X, y = fetch_california_housing(return_X_y=True)

X_scaled = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

print(f'score: {model.score(X_test, y_test):.3f}')
print(f'weights: {model.weights}')





