# Multiple Linear Regression

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset

dataset = pd.read_csv("c:\\Users\\costa\\Codes\\Learning\\Machine Learning\\Continuous\\Multiple Linear Regression\\data\\50_Startups.csv")
print(dataset)
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y,test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)
print(y_test)
print("==============")
print(y_pred)

"""
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Profit')
plt.show()
"""