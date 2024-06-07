import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from LinearRegression import LinearRegression


df = pd.read_csv("..\Students_on_Mars.csv")
X = df.drop(['y'], axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

degrees = [1, 2, 3, 4]
errors = list()
model = LinearRegression()

for degree in degrees:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    model.fit(X_train_poly, y_train)
    errors.append(1 - model.score(X_test_poly, y_test))

min_error = min(errors)
print(f'The optimal polynomial degree is {degrees[errors.index(min_error)]}')

plt.plot(degrees, errors)
plt.xlabel('degree')
plt.ylabel('error')
plt.xticks(degrees)
plt.show()
