# Naive bayes


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('pulsar_stars.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# Data visualization
men = X[:, 0]
plt.scatter(men, y)
plt.title('Mean of the integrated profile vs target_class')
plt.xlabel('Mean of the integrated profile')
plt.ylabel('target_class')
plt.show()

glu = X[:, 1]
plt.scatter(glu, y)
plt.title('Standard deviation of the integrated profile vs target_class')
plt.xlabel('Standard deviation of the integrated profile')
plt.ylabel('target_class')
plt.show()

bld = X[:, 2]
plt.scatter(bld, y)
plt.title('Excess kurtosis of the integrated profile vs target_class')
plt.xlabel('Excess kurtosis of the integrated profile')
plt.ylabel('target_class')
plt.show()

skn = X[:, 3]
plt.scatter(skn, y)
plt.title('Skewness of the integrated profile vs target_class')
plt.xlabel('Skewness of the integrated profile')
plt.ylabel('target_class')
plt.show()

ins = X[:, 4]
plt.scatter(ins, y)
plt.title('Mean of the DM-SNR curve vs target_class')
plt.xlabel( 'Mean of the DM-SNR curve')
plt.ylabel('target_class')
plt.show()

bmi = X[:, 5]
plt.scatter(bmi, y)
plt.title('Standard deviation of the DM-SNR curve vs target_class')
plt.xlabel('Standard deviation of the DM-SNR curve')
plt.ylabel('target_class')
plt.show()

dpf = X[:, 6]
plt.scatter(dpf, y)
plt.title('Excess kurtosis of the DM-SNR curve vs target_class')
plt.xlabel('Excess kurtosis of the DM-SNR curve')
plt.ylabel('target_class')
plt.show()

age = X[:, 7]
plt.scatter(age, y)
plt.title('Skewness of the DM-SNR curve vs target_class')
plt.xlabel('Skewness of the DM-SNR curve')
plt.ylabel('target_class')
plt.show()


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# R-Squared and Adjusted R-Squared values
from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_pred)
print("R-square value: "+str(R2))
n = 2685
p = 8
AR2 = 1-(1-R2)*(n-1)/(n-p-1)
print("Adjusted R-square value: "+str(AR2))

