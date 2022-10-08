import pickle

import pandas as pd
import numpy as np
from sklearn import datasets, linear_model, metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle

Data = pd.read_csv("Housing_Modified.csv")

Y = Data.iloc[:,0].values // 1000
X = Data.iloc[:,1:4].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1,random_state=1)

reg = linear_model.LinearRegression()
reg.fit(X_train , y_train)

print('Coefficients: ', reg.coef_)
print('Variance score: {}'.format(reg.score(X_test, y_test)))

train_error = np.mean((reg.predict(X_train) - y_train)**2)
test_error = np.mean((reg.predict(X_test) - y_test)**2)

print("Train Cost is : {}".format(round(train_error , 3)))
print("Test Cost is : {}".format(round(test_error , 3)))

score = round(reg.score(X_test , y_test) * 100 , 3)
print("Accuracy of Model is {}".format(score))

pickle.dump(reg , open("model.pkl" , "wb"))

model = pickle.load(open("model.pkl" , "rb"))