import csv
import pandas as pd
import matplotlib.pyplot as plt
from decimal import Decimal
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.impute import SimpleImputer

##load in the data
world=pd.read_csv('data/world.csv',encoding = 'ISO-8859-1')
life=pd.read_csv('data/life.csv',encoding = 'ISO-8859-1')

world = world.drop(columns=['Country Name', 'Time'])
world = life.merge(world, on='Country Code', how='inner').drop(columns=['Year'])
world.sort_values(by='Country Code', inplace = True)
world.replace('..', np.nan, inplace = True)

# get the required data
data = world.drop(columns=['Country', 'Country Code', 'Life expectancy at birth (years)']).astype(float)
# get the class labels
classlabel = world['Life expectancy at birth (years)']


# randomly select 70% of the instances to be training and the rest to be testing
X_train, X_test, y_train, y_test = train_test_split(data,classlabel, train_size=0.7, test_size=0.3, random_state=200)

# impute median by fitting with X_train and the transforming both sets
imp_median = SimpleImputer(missing_values=np.nan, strategy='median').fit(X_train)
X_train = imp_median.transform(X_train)
X_test = imp_median.transform(X_test)
X_train = pd.DataFrame(X_train, columns=data.columns)
X_test = pd.DataFrame(X_test, columns=data.columns)

# Standardize the data to have 0 mean and unit variance.
scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

# Get medians, means and variances for each feature
medians = [round(Decimal(imp_median.statistics_[i])) for i in range(len(imp_median.statistics_))]
means = [round(Decimal(scaler.mean_[i])) for i in range(len(scaler.mean_))]
variances = [round(Decimal(scaler.var_[i])) for i in range(len(scaler.var_))]

# Produce task2a file with features and their median, mean and variance
with open('output/task2a.csv', 'w') as task2a:
    writer = csv.writer(task2a)
    writer.writerow(['feature' , 'median', 'mean', 'variance'])
    for i in range(len(data.columns)):
        writer.writerow([data.columns[i], medians[i], means[i], variances[i]])


# Fit the decision tree and predict X_test
dt = DecisionTreeClassifier(random_state=200, max_depth=3)
dt.fit(X_train, y_train)
y_pred_dt=dt.predict(X_test)
accuracy_dt = round(Decimal(accuracy_score(y_test, y_pred_dt)*100), 3)

# fit the 7-nn model and predict X_test
knn = neighbors.KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred_k7=knn.predict(X_test)
accuracy_k7=round(Decimal(accuracy_score(y_test, y_pred_k7)*100),3)

# fit the 3-nn model and predict X_test
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_k3=knn.predict(X_test)
accuracy_k3=round(Decimal(accuracy_score(y_test, y_pred_k3)*100),3)

print("Accuracy of decision tree: " + str(accuracy_dt) + '%')
print("Accuracy of k-nn (k=3): " + str(accuracy_k3) + '%')
print("Accuracy of k-nn (k=7): " + str(accuracy_k7) + '%')


