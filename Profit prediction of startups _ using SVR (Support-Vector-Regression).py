# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# print(X)
# print(y)
y = y.reshape(len(y),1)
# print(y)

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)
X_test = sc_X.transform(X_test)

# Training the Support Vector Regression model on the Training set
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf', C=100, degree=2, epsilon=0.01, gamma=0.001)
regressor.fit(X_train, y_train.ravel())

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{    'C': [0.01, 0.1, 1, 10, 100],
                    'kernel': ['poly', 'rbf', 'sigmoid'],
                    'degree': [2, 3, 4],
                    'gamma': ['scale', 'auto'] + list(np.arange(0.001, 1.001, 0.1)),
                    'epsilon': [0.01, 0.1, 1]}]
grid_search = GridSearchCV(estimator = regressor,
                            param_grid = parameters,
                            scoring = 'neg_mean_squared_error',
                            cv = 10,
                            n_jobs = -1)
grid_search.fit(X_train, y_train.ravel())
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Mean Squared Error: {:.2f} %".format(-best_accuracy*100))
print("Best Parameters:", best_parameters)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10, scoring='neg_mean_squared_error')
print("Mean Squared Error: {:.2f} %".format(-scores.mean()*100))
print("Standard Deviation: {:.2f} %".format(scores.std()*100))

# Predicting the Test set results
y_pred = sc_y.inverse_transform(regressor.predict(X_test).reshape(-1,1))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Average of Differences
differences= []
for i in range(len(y_test)):
    differences.append(abs(y_test[i]-y_pred[i]))
average_differences = sum(differences)/len(differences)
print('\n Average of Differences :', average_differences[0])

# Single Prediction
single_prediction_ = [130298.13, 147198.87, 256512.92, 'California']
single_prediction = np.array(sc_X.transform(ct.transform([single_prediction_])))
y_single_prediction = sc_y.inverse_transform(regressor.predict(single_prediction).reshape(-1,1))
print(f''' Single Prediction:
      Startup with these features:
      {str(single_prediction_)}
      will be {y_single_prediction[0][0]}$ profit.''')

# Visualising Difference between Test and Predicted results
plt.scatter(np.arange(len(y_pred)), y_test, color = 'green', s=110, label='Real Values')
plt.scatter(np.arange(len(y_pred)), y_pred, color = 'red', marker='*', s=70, label='Prediction')
plt.title('Difference between Test and Predicted results')
plt.xlabel('Test Set')
plt.ylabel('Profit')
plt.legend()
plt.show()

print('____________________Visualizing Train set results____________________')

# Visualising Difference between Train and Regressor results (R&D Spend)
X_train_sorted = X_train[X_train[:,3].argsort()]
y_train_sorted = y_train[X_train[:,3].argsort()]
plt.scatter(sc_X.inverse_transform(X_train_sorted)[:,3].reshape(-1,1), sc_y.inverse_transform(y_train_sorted).reshape(-1,1), color = 'green', s=110, label='R&D Spend , y')
plt.plot(sc_X.inverse_transform(X_train_sorted)[:,3].reshape(-1,1), sc_y.inverse_transform(regressor.predict(X_train_sorted).reshape(-1,1)), color = 'red', marker='*', label='Regressor')
plt.scatter(sc_X.inverse_transform(single_prediction)[0][3], y_single_prediction, color='blue', s=200, marker='*', label='Single Prediction') # this has been added to show the single prediction result
plt.title('Difference between Train and Regressor results (R&D Spend)')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.legend()
plt.show()

# Visualising Difference between Train and Regressor results (Administration)
X_train_sorted = X_train[X_train[:,4].argsort()]
y_train_sorted = y_train[X_train[:,4].argsort()]
plt.scatter(sc_X.inverse_transform(X_train_sorted)[:,4].reshape(-1,1), sc_y.inverse_transform(y_train_sorted).reshape(-1,1), color = 'green', s=110, label='Administration , y')
plt.plot(sc_X.inverse_transform(X_train_sorted)[:,4].reshape(-1,1), sc_y.inverse_transform(regressor.predict(X_train_sorted).reshape(-1,1)), color = 'red', marker='*', label='Regressor')
plt.scatter(sc_X.inverse_transform(single_prediction)[0][4], y_single_prediction, color='blue', s=200, marker='*', label='Single Prediction') # this has been added to show the single prediction result
plt.title('Difference between Train and Regressor results (Administration)')
plt.xlabel('Administration')
plt.ylabel('Profit')
plt.legend()
plt.show()

# Visualising Difference between Train and Regressor results (Marketing Spend)
X_train_sorted = X_train[X_train[:,5].argsort()]
y_train_sorted = y_train[X_train[:,5].argsort()]
plt.scatter(sc_X.inverse_transform(X_train_sorted)[:,5].reshape(-1,1), sc_y.inverse_transform(y_train_sorted).reshape(-1,1), color = 'green', s=110, label='Marketing Spend , y')
plt.plot(sc_X.inverse_transform(X_train_sorted)[:,5].reshape(-1,1), sc_y.inverse_transform(regressor.predict(X_train_sorted).reshape(-1,1)), color = 'red', marker='*', label='Regressor')
plt.scatter(sc_X.inverse_transform(single_prediction)[0][5], y_single_prediction, color='blue', s=200, marker='*', label='Single Prediction') # this has been added to show the single prediction result
plt.title('Difference between Train and Regressor results (Marketing Spend)')
plt.xlabel('Marketing Spend')
plt.ylabel('Profit')
plt.legend()
plt.show()

print('____________________Visualizing Test set results____________________')

# Visualising Difference between Test and Regressor results (R&D Spend)
X_test_sorted = X_test[X_test[:,3].argsort()]
y_test_sorted = y_test[X_test[:,3].argsort()]
plt.scatter(sc_X.inverse_transform(X_test_sorted)[:,3].reshape(-1,1), y_test_sorted.reshape(-1,1), color = 'green', s=110, label='R&D Spend , y')
plt.plot(sc_X.inverse_transform(X_test_sorted)[:,3].reshape(-1,1), sc_y.inverse_transform(regressor.predict(X_test_sorted).reshape(-1,1)), color = 'red', marker='*', label='Regressor')
plt.scatter(sc_X.inverse_transform(single_prediction)[0][3], y_single_prediction, color='blue', s=200, marker='*', label='Single Prediction') # this has been added to show the single prediction result
plt.title('Difference between Test and Regressor results (R&D Spend)')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.legend()
plt.show()

# Visualising Difference between Test and Regressor results (Administration)
X_test_sorted = X_test[X_test[:,4].argsort()]
y_test_sorted = y_test[X_test[:,4].argsort()]
plt.scatter(sc_X.inverse_transform(X_test_sorted)[:,4].reshape(-1,1), y_test_sorted.reshape(-1,1), color = 'green', s=110, label='Administration , y')
plt.plot(sc_X.inverse_transform(X_test_sorted)[:,4].reshape(-1,1), sc_y.inverse_transform(regressor.predict(X_test_sorted).reshape(-1,1)), color = 'red', marker='*', label='Regressor')
plt.scatter(sc_X.inverse_transform(single_prediction)[0][4], y_single_prediction, color='blue', s=200, marker='*', label='Single Prediction') # this has been added to show the single prediction result
plt.title('Difference between Test and Regressor results (Administration)')
plt.xlabel('Administration')
plt.ylabel('Profit')
plt.legend()
plt.show()

# Visualising Difference between Test and Regressor results (Marketing Spend)
X_test_sorted = X_test[X_test[:,5].argsort()]
y_test_sorted = y_test[X_test[:,5].argsort()]
plt.scatter(sc_X.inverse_transform(X_test_sorted)[:,5].reshape(-1,1), y_test_sorted.reshape(-1,1), color = 'green', s=110, label='Marketing Spend , y')
plt.plot(sc_X.inverse_transform(X_test_sorted)[:,5].reshape(-1,1), sc_y.inverse_transform(regressor.predict(X_test_sorted).reshape(-1,1)), color = 'red', marker='*', label='Regressor')
plt.scatter(sc_X.inverse_transform(single_prediction)[0][5], y_single_prediction, color='blue', s=200, marker='*', label='Single Prediction') # this has been added to show the single prediction result
plt.title('Difference between Test and Regressor results (Marketing Spend)')
plt.xlabel('Marketing Spend')
plt.ylabel('Profit')
plt.legend()
plt.show()

# Save the output (result) in CSV file :
Header = ['Module_Name', 'Mean_Squared_Error', 'Standard_Deviation', 'Average of Differences', 'Predicted_Startup_Price']
output= [['SVR', -scores.mean()*100, scores.std()*100, average_differences[0], y_single_prediction[0][0]]]
output = np.array(output)
Output = pd.DataFrame(output)
try :
    pd.read_csv('C:/Users/Miramin_LPC/Desktop/PythonS/Projects/Profit prediction of startups (with 50 startups dataset)/Outputs.csv')
    Output.to_csv('C:/Users/Miramin_LPC/Desktop/PythonS/Projects/Profit prediction of startups (with 50 startups dataset)/Outputs.csv', mode='a', index=False, header=False)
except:
    Output.to_csv('C:/Users/Miramin_LPC/Desktop/PythonS/Projects/Profit prediction of startups (with 50 startups dataset)/Outputs.csv', mode='w', sep=',', index=False, header=Header)
