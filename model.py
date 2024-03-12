#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv('data.csv')

# Preprocess the data
# This is a simplified example, in a real-world scenario you'd need to handle missing values, normalize variables, etc.
X = data.drop('INCOME', axis=1)
y = data['INCOME']

# Preprocess the data
# Keep only numeric columns
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
X = X[numeric_cols]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Choose a model
model = LinearRegression()




# In[26]:


from sklearn.impute import SimpleImputer

# Create an instance of SimpleImputer with strategy to fill missing values with the 'mean'
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fit the imputer and transform the data
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Continue with the model training
model.fit(X_train, y_train)


# In[27]:


from sklearn.metrics import mean_squared_error

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")


# In[28]:


# save the model to a file
import joblib
MODEL_FILE = "predictions_model.pkl"
joblib.dump(model, MODEL_FILE)


# In[29]:


from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)


# In[30]:


# Calculate the score of the model
loaded_model = joblib.load(MODEL_FILE)
score = loaded_model.score(X_test, y_test)

print(f"Model Score: {score}")

