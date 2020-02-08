import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#dataset
train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

#X and y
X_train = train_dataset.iloc[:,0:1]
y_train = train_dataset.iloc[:,1]
X_test = test_dataset.iloc[:,0:1]
y_test = test_dataset.iloc[:,1]

#Using Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting values of test set
y_pred = regressor.predict(X_test)

#Graph for training set
plt.scatter(X_train,y_train,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.title('Training set')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#Graph for testing set
plt.scatter(X_test,y_test,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.title('Testing set')
plt.xlabel('x')
plt.ylabel('y')
plt.show()