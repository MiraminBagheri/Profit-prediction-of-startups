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

# Training the Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(X_train, y_train)

# Training the Polynomial Regression model on the Training set
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_train)
regressor2 = LinearRegression()
regressor2.fit(X_poly, y_train)

print('______________________Here starts Linear results______________________')
#______________________________________________________________________________

# Predicting the Test set results
y_pred1 = sc_y.inverse_transform(regressor1.predict(X_test).reshape(-1,1))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred1.reshape(len(y_pred1),1), y_test.reshape(len(y_test),1)),1))

# Average of Differences
differences1= []
for i in range(len(y_test)):
    differences1.append(abs(y_test[i]-y_pred1[i]))
average_differences1 = sum(differences1)/len(differences1)
print('\n Average of Differences :', average_differences1)

# Single Prediction
single_prediction_ = [130298.13, 147198.87, 256512.92, 'California']
single_prediction = np.array(sc_X.transform(ct.transform([single_prediction_])))
y_single_prediction1 = sc_y.inverse_transform(regressor1.predict(single_prediction).reshape(-1,1))
print(f''' Single Prediction:
      Startup with these features:
      {str(single_prediction_)}
      will be {y_single_prediction1[0][0]}$ profit.''')

# Visualising Difference between Test and Predicted results
plt.scatter(np.arange(len(y_pred1)), y_test, color = 'green', s=110, label='Real Values')
plt.scatter(np.arange(len(y_pred1)), y_pred1, color = 'red', marker='*', s=70, label='Prediction')
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
plt.plot(sc_X.inverse_transform(X_train_sorted)[:,3].reshape(-1,1), sc_y.inverse_transform(regressor1.predict(X_train_sorted).reshape(-1,1)), color = 'red', marker='*', label='Regressor')
plt.scatter(sc_X.inverse_transform(single_prediction)[0][3], y_single_prediction1, color='blue', s=200, marker='*', label='Single Prediction') # this has been added to show the single prediction result
plt.title('Difference between Train and Regressor results (R&D Spend)')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.legend()
plt.show()

# Visualising Difference between Train and Regressor results (Administration)
X_train_sorted = X_train[X_train[:,4].argsort()]
y_train_sorted = y_train[X_train[:,4].argsort()]
plt.scatter(sc_X.inverse_transform(X_train_sorted)[:,4].reshape(-1,1), sc_y.inverse_transform(y_train_sorted).reshape(-1,1), color = 'green', s=110, label='Administration , y')
plt.plot(sc_X.inverse_transform(X_train_sorted)[:,4].reshape(-1,1), sc_y.inverse_transform(regressor1.predict(X_train_sorted).reshape(-1,1)), color = 'red', marker='*', label='Regressor')
plt.scatter(sc_X.inverse_transform(single_prediction)[0][4], y_single_prediction1, color='blue', s=200, marker='*', label='Single Prediction') # this has been added to show the single prediction result
plt.title('Difference between Train and Regressor results (Administration)')
plt.xlabel('Administration')
plt.ylabel('Profit')
plt.legend()
plt.show()

# Visualising Difference between Train and Regressor results (Marketing Spend)
X_train_sorted = X_train[X_train[:,5].argsort()]
y_train_sorted = y_train[X_train[:,5].argsort()]
plt.scatter(sc_X.inverse_transform(X_train_sorted)[:,5].reshape(-1,1), sc_y.inverse_transform(y_train_sorted).reshape(-1,1), color = 'green', s=110, label='Marketing Spend , y')
plt.plot(sc_X.inverse_transform(X_train_sorted)[:,5].reshape(-1,1), sc_y.inverse_transform(regressor1.predict(X_train_sorted).reshape(-1,1)), color = 'red', marker='*', label='Regressor')
plt.scatter(sc_X.inverse_transform(single_prediction)[0][5], y_single_prediction1, color='blue', s=200, marker='*', label='Single Prediction') # this has been added to show the single prediction result
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
plt.plot(sc_X.inverse_transform(X_test_sorted)[:,3].reshape(-1,1), sc_y.inverse_transform(regressor1.predict(X_test_sorted).reshape(-1,1)), color = 'red', marker='*', label='Regressor')
plt.scatter(sc_X.inverse_transform(single_prediction)[0][3], y_single_prediction1, color='blue', s=200, marker='*', label='Single Prediction') # this has been added to show the single prediction result
plt.title('Difference between Test and Regressor results (R&D Spend)')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.legend()
plt.show()

# Visualising Difference between Test and Regressor results (Administration)
X_test_sorted = X_test[X_test[:,4].argsort()]
y_test_sorted = y_test[X_test[:,4].argsort()]
plt.scatter(sc_X.inverse_transform(X_test_sorted)[:,4].reshape(-1,1), y_test_sorted.reshape(-1,1), color = 'green', s=110, label='Administration , y')
plt.plot(sc_X.inverse_transform(X_test_sorted)[:,4].reshape(-1,1), sc_y.inverse_transform(regressor1.predict(X_test_sorted).reshape(-1,1)), color = 'red', marker='*', label='Regressor')
plt.scatter(sc_X.inverse_transform(single_prediction)[0][4], y_single_prediction1, color='blue', s=200, marker='*', label='Single Prediction') # this has been added to show the single prediction result
plt.title('Difference between Test and Regressor results (Administration)')
plt.xlabel('Administration')
plt.ylabel('Profit')
plt.legend()
plt.show()

# Visualising Difference between Test and Regressor results (Marketing Spend)
X_test_sorted = X_test[X_test[:,5].argsort()]
y_test_sorted = y_test[X_test[:,5].argsort()]
plt.scatter(sc_X.inverse_transform(X_test_sorted)[:,5].reshape(-1,1), y_test_sorted.reshape(-1,1), color = 'green', s=110, label='Marketing Spend , y')
plt.plot(sc_X.inverse_transform(X_test_sorted)[:,5].reshape(-1,1), sc_y.inverse_transform(regressor1.predict(X_test_sorted).reshape(-1,1)), color = 'red', marker='*', label='Regressor')
plt.scatter(sc_X.inverse_transform(single_prediction)[0][5], y_single_prediction1, color='blue', s=200, marker='*', label='Single Prediction') # this has been added to show the single prediction result
plt.title('Difference between Test and Regressor results (Marketing Spend)')
plt.xlabel('Marketing Spend')
plt.ylabel('Profit')
plt.legend()
plt.show()

print('____________________Here starts Polynominal results____________________')
#______________________________________________________________________________

# Predicting the Test set results
y_pred2 = sc_y.inverse_transform(regressor2.predict(poly_reg.fit_transform(X_test)).reshape(-1,1))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred2.reshape(len(y_pred2),1), y_test.reshape(len(y_test),1)),1))

# Average of Differences
differences2= []
for i in range(len(y_test)):
    differences2.append(abs(y_test[i]-y_pred2[i]))
average_differences2 = sum(differences2)/len(differences2)
print('\n Average of Differences :', average_differences2)

# Single Prediction
single_prediction_ = [130298.13, 147198.87, 256512.92, 'California']
single_prediction = np.array(sc_X.transform(ct.transform([single_prediction_])))
y_single_prediction2 = sc_y.inverse_transform(regressor2.predict(poly_reg.fit_transform(single_prediction)).reshape(-1,1))
print(f''' Single Prediction:
      Startup with these features:
      {str(single_prediction_)} 
      will be {y_single_prediction2[0][0]}$ profit.''')

# Visualising Difference between Test and Predicted results
plt.scatter(np.arange(len(y_pred2)), y_test, color = 'green', s=110, label='Real Values')
plt.scatter(np.arange(len(y_pred2)), y_pred2, color = 'red', marker='*', s=70, label='Prediction')
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
plt.plot(sc_X.inverse_transform(X_train_sorted)[:,3].reshape(-1,1), sc_y.inverse_transform(regressor2.predict(poly_reg.fit_transform(X_train_sorted)).reshape(-1, 1)), color = 'red', marker='*', label='Regressor')
plt.scatter(sc_X.inverse_transform(single_prediction)[0][3], y_single_prediction2, color='blue', s=200, marker='*', label='Single Prediction') # this has been added to show the single prediction result
plt.title('Difference between Train and Regressor results (R&D Spend)')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.legend()
plt.show()

# Visualising Difference between Train and Regressor results (Administration)
X_train_sorted = X_train[X_train[:,4].argsort()]
y_train_sorted = y_train[X_train[:,4].argsort()]
plt.scatter(sc_X.inverse_transform(X_train_sorted)[:,4].reshape(-1,1), sc_y.inverse_transform(y_train_sorted).reshape(-1,1), color = 'green', s=110, label='Administration , y')
plt.plot(sc_X.inverse_transform(X_train_sorted)[:,4].reshape(-1,1), sc_y.inverse_transform(regressor2.predict(poly_reg.fit_transform(X_train_sorted)).reshape(-1, 1)), color = 'red', marker='*', label='Regressor')
plt.scatter(sc_X.inverse_transform(single_prediction)[0][4], y_single_prediction2, color='blue', s=200, marker='*', label='Single Prediction') # this has been added to show the single prediction result
plt.title('Difference between Train and Regressor results (Administration)')
plt.xlabel('Administration')
plt.ylabel('Profit')
plt.legend()
plt.show()

# Visualising Difference between Train and Regressor results (Marketing Spend)
X_train_sorted = X_train[X_train[:,5].argsort()]
y_train_sorted = y_train[X_train[:,5].argsort()]
plt.scatter(sc_X.inverse_transform(X_train_sorted)[:,5].reshape(-1,1), sc_y.inverse_transform(y_train_sorted).reshape(-1,1), color = 'green', s=110, label='Marketing Spend , y')
plt.plot(sc_X.inverse_transform(X_train_sorted)[:,5].reshape(-1,1), sc_y.inverse_transform(regressor2.predict(poly_reg.fit_transform(X_train_sorted)).reshape(-1, 1)), color = 'red', marker='*', label='Regressor')
plt.scatter(sc_X.inverse_transform(single_prediction)[0][5], y_single_prediction2, color='blue', s=200, marker='*', label='Single Prediction') # this has been added to show the single prediction result
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
plt.plot(sc_X.inverse_transform(X_test_sorted)[:,3].reshape(-1,1), sc_y.inverse_transform(regressor2.predict(poly_reg.fit_transform(X_test_sorted)).reshape(-1, 1)), color = 'red', marker='*', label='Regressor')
plt.scatter(sc_X.inverse_transform(single_prediction)[0][3], y_single_prediction2, color='blue', s=200, marker='*', label='Single Prediction') # this has been added to show the single prediction result
plt.title('Difference between Test and Regressor results (R&D Spend)')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.legend()
plt.show()

# Visualising Difference between Test and Regressor results (Administration)
X_test_sorted = X_test[X_test[:,4].argsort()]
y_test_sorted = y_test[X_test[:,4].argsort()]
plt.scatter(sc_X.inverse_transform(X_test_sorted)[:,4].reshape(-1,1), y_test_sorted.reshape(-1,1), color = 'green', s=110, label='Administration , y')
plt.plot(sc_X.inverse_transform(X_test_sorted)[:,4].reshape(-1,1), sc_y.inverse_transform(regressor2.predict(poly_reg.fit_transform(X_test_sorted)).reshape(-1, 1)), color = 'red', marker='*', label='Regressor')
plt.scatter(sc_X.inverse_transform(single_prediction)[0][4], y_single_prediction2, color='blue', s=200, marker='*', label='Single Prediction') # this has been added to show the single prediction result
plt.title('Difference between Test and Regressor results (Administration)')
plt.xlabel('Administration')
plt.ylabel('Profit')
plt.legend()
plt.show()

# Visualising Difference between Test and Regressor results (Marketing Spend)
X_test_sorted = X_test[X_test[:,5].argsort()]
y_test_sorted = y_test[X_test[:,5].argsort()]
plt.scatter(sc_X.inverse_transform(X_test_sorted)[:,5].reshape(-1,1), y_test_sorted.reshape(-1,1), color = 'green', s=110, label='Marketing Spend , y')
plt.plot(sc_X.inverse_transform(X_test_sorted)[:,5].reshape(-1,1), sc_y.inverse_transform(regressor2.predict(poly_reg.fit_transform(X_test_sorted)).reshape(-1, 1)), color = 'red', marker='*', label='Regressor')
plt.scatter(sc_X.inverse_transform(single_prediction)[0][5], y_single_prediction2, color='blue', s=200, marker='*', label='Single Prediction') # this has been added to show the single prediction result
plt.title('Difference between Test and Regressor results (Marketing Spend)')
plt.xlabel('Marketing Spend')
plt.ylabel('Profit')
plt.legend()
plt.show()

# Save the output (result) in CSV file :
Header = ['Module_Name', 'Mean_Squared_Error', 'Standard_Deviation', 'Average of Differences', 'Predicted_Startup_Price']
output= [['Linear Regression', '', '', average_differences1[0], y_single_prediction1[0][0]],
         ['Polynominal Regression', '', '', average_differences2[0], y_single_prediction2[0][0]]]
output = np.array(output)
Output = pd.DataFrame(output)
try :
    pd.read_csv('C:/Users/Miramin_LPC/Desktop/PythonS/Projects/Profit prediction of startups (with 50 startups dataset)/Outputs.csv')
    Output.to_csv('C:/Users/Miramin_LPC/Desktop/PythonS/Projects/Profit prediction of startups (with 50 startups dataset)/Outputs.csv', mode='a', index=False, header=False)
except:
    Output.to_csv('C:/Users/Miramin_LPC/Desktop/PythonS/Projects/Profit prediction of startups (with 50 startups dataset)/Outputs.csv', mode='w', sep=',', index=False, header=Header)
