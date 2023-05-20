# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('Outputs.csv')
print(dataset)
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1:5].values

# Visualising the Outputs (results)
sns.lineplot(X, y[:,0])
plt.scatter(X, y[:,0])
plt.title('Mean Squared Error')
plt.xlabel('Modules')
plt.ylabel('Mean Squared Error')
plt.xticks(rotation=60)
plt.show()

# Visualising the Outputs (results)
sns.lineplot(X, y[:,1])
plt.scatter(X, y[:,1])
plt.title('Standard Deviation')
plt.xlabel('Modules')
plt.ylabel('Standard Deviation')
plt.xticks(rotation=60)
plt.show()

sns.lineplot(X, y[:,2])
plt.scatter(X, y[:,2])
plt.title('Average of Differences')
plt.xlabel('Modules')
plt.ylabel('Average of Differences')
plt.xticks(rotation=60)
plt.show()

# Visualising the Outputs (results)
sns.lineplot(X, y[:,3])
plt.scatter(X, y[:,3])
plt.title('Predicted Startup Price')
plt.xlabel('Modules')
plt.ylabel('Predicted Startup Price')
plt.xticks(rotation=60)
plt.show()

# Print Performances and Summary 
print('\n Performances:\n',dataset)
summary = {'':['Minimum','Maximum','Average'],
             'Estimated_Position_Salary':[y[:,3].min(), y[:,3].max(), y[:,3].mean()],
             'Module_Name':[X[list(y[:,3]).index(y[:,3].min())], X[list(y[:,3]).index(y[:,3].max())], '']}
summary = pd.DataFrame(summary, index=None)
print('\n Summary:\n',summary)