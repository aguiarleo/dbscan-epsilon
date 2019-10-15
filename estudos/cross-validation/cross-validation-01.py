# Ref: https://medium.com/datadriveninvestor/k-fold-cross-validation-6b8518070833

import pandas
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import numpy as np

dataset = pandas.read_csv('housing.csv',sep='\s+')
X = dataset.iloc[:, [0, 12]]
y = dataset.iloc[:, 13]


scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

scores = []
best_svr = SVR(kernel='rbf',gamma='auto')

# the number of folds we want our data set to be split into. Here, we have used 10-Fold CV (n_splits=10), where the data will be split into 10 folds.
cv = KFold(n_splits=10, random_state=42, shuffle=False)
for train_index, test_index in cv.split(X):
    
    #We are printing out the indexes of the training and the testing sets in each iteration to clearly see the process of K-Fold CV where the training and testing set changes in each iteration.
    print("Train Index: ", train_index, "\n")
    print("Test Index: ", test_index)

    #we specify the training and testing sets to be used in each iteration. For this, we use the indexes(train_index, test_index) specified in the K-Fold CV process. 
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    
    # we train the model in each iteration using the train_index of each iteration of the K-Fold process and append the error metric value to a list(scores ).
    best_svr.fit(X_train, y_train)
    scores.append(best_svr.score(X_test, y_test))


#get the mean value in order to determine the overall accuracy of the model.
print("\nnp.mean(scores):\n",np.mean(scores))
print("\ncross_val_score:\n",cross_val_score(best_svr, X, y, cv=10))
print("\ncross_val_predict",cross_val_predict(best_svr, X, y, cv=10))
