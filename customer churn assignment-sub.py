#!/usr/bin/env python
# coding: utf-8

# 1. Problem Statement:
# You are the data scientist at a telecom company named “Neo” whose customers
# are churning out to its competitors. You have to analyze the data of your
# company and find insights and stop your customers from churning out to other
# telecom companies.
# ##Customer_churn Dataset:
# The details regarding this ‘customer_churn’ dataset are present in the data
# dictionary

# Lab Environment: Anaconda
# Domain: Telecom
# Tasks To Be Performed:
# 1. Data Manipulation:
# 
# ● Extract the 5th column and store it in ‘customer_5’
# ● Extract the 15th column and store it in ‘customer_15’
# ● Extract all the male senior citizens whose payment method is electronic
# check and store the result in ‘senior_male_electronic’
# ● Extract all those customers whose tenure is greater than 70 months or
# their monthly charges is more than $100 and store the result in
# ‘customer_total_tenure’
# ● Extract all the customers whose contract is of two years, payment method
# is mailed check and the value of churn is ‘Yes’ and store the result in
# ‘two_mail_yes’
# ● Extract 333 random records from the customer_churndataframe and store
# the result in ‘customer_333’
# ● Get the count of different levels from the ‘Churn’ column
# 

# 2. Data Visualization:
# ● Build a bar-plot for the ’InternetService’ column:
# a. Set x-axis label to ‘Categories of Internet Service’
# b. Set y-axis label to ‘Count of Categories’
# c. Set the title of plot to be ‘Distribution of Internet Service’
# d. Set the color of the bars to be ‘orange’
# ● Build a histogram for the ‘tenure’ column:
# a. Set the number of bins to be 30
# b. Set the color of the bins to be ‘green’
# c. Assign the title ‘Distribution of tenure’
# ● Build a scatter-plot between ‘MonthlyCharges’ and ‘tenure’. Map
# ‘MonthlyCharges’ to the y-axis and ‘tenure’ to the ‘x-axis’:
# a. Assign the points a color of ‘brown’
# b. Set the x-axis label to ‘Tenure of customer’
# c. Set the y-axis label to ‘Monthly Charges of customer’
# d. Set the title to ‘Tenure vs Monthly Charges’
# e. Build a box-plot between ‘tenure’ & ‘Contract’. Map ‘tenure’ on the
# y-axis &
# f. ‘Contract’ on the x-axis.
# 
# 
# 
# 

# 3. Linear Regression:
# ● Build a simple linear model where dependent variable is ‘MonthlyCharges’
# and independent variable is ‘tenure’:
# a. Divide the dataset into train and test sets in 70:30 ratio.
# b. Build the model on train set and predict the values on test set
# c. After predicting the values, find the root mean square error
# d. Find out the error in prediction & store the result in ‘error’
# e. Find the root mean square error
# 

# 4. Logistic Regression:
# ● Build a simple logistic regression model where dependent variable is
# ‘Churn’ and independent variable is ‘MonthlyCharges’:
# a. Divide the dataset in 65:35 ratio
# b. Build the model on train set and predict the values on test set
# c. Build the confusion matrix and get the accuracy score
# d. Build a multiple logistic regression model where dependent variable
# is ‘Churn’ and independent variables are ‘tenure’ and
# ‘MonthlyCharges’
# e. Divide the dataset in 80:20 ratio
# f. Build the model on train set and predict the values on test set
# g. Build the confusion matrix and get the accuracy score

# 5. Decision Tree:
# ● Build a decision tree model where dependent variable is ‘Churn’ and
# independent variable is ‘tenure’:
# a. Divide the dataset in 80:20 ratio
# b. Build the model on train set and predict the values on test set
# c. Build the confusion matrix and calculate the accuracy

# 6. Random Forest:
# ● Build a Random Forest model where dependent variable is ‘Churn’ and
# independent variables are ‘tenure’ and ‘MonthlyCharges’:
# a. Divide the dataset in 70:30 ratio
# b. Build the model on train set and predict the values on test set
# c. Build the confusion matrix and calculate the accuracy

# In[43]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv(r"S:\INTELLIPAAT\Statistics\machinelearning\assignquizpro\customer_churn.csv")


# In[32]:


df


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.info()


# In[8]:


df.isna().sum().sum()


# In[1]:


customer_5 = df.iloc[:, 5]
customer_5_list = customer_5.tolist()


# In[11]:


customer_5_list


# In[12]:


customer_15 = df.iloc[: , 15]


# In[13]:


customer_15


# In[33]:


senior_male_electronic = df[(df['gender'] == 'Male') & (df['SeniorCitizen'] == 1) & (df['PaymentMethod'] == 'Electronic check')]


# In[15]:


senior_male_electronic


# In[34]:


customer_total_tenure = df[(df['tenure'] > 70) | (df['MonthlyCharges'] > 100)]


# In[21]:


customer_total_tenure


# In[35]:


two_mail_yes = df[(df['Contract'] =='Two year') & (df['PaymentMethod'] == 'Mailed check') & (df['Churn'] == 'Yes')]


# In[29]:


two_mail_yes


# In[36]:


import pandas as pd
contract_mailed_check_churn_yes = df[(df['Contract'] == 'Two year') & (df['PaymentMethod'] == 'Mailed check') & (df['Churn'] == 'Yes')]


# In[31]:


contract_mailed_check_churn_yes


# In[39]:


customer_333 = df.sample(n=333, random_state=42)


# In[40]:


customer_333


# In[41]:


churn_counts = df['Churn'].value_counts()

print(churn_counts)


# Data Visualization

#  ● Build a bar-plot for the ’InternetService’ column: 
#         a. Set x-axis label to ‘Categories of Internet Service’ 
#         b. Set y-axis label to ‘Count of Categories’ 
#         c. Set the title of plot to be ‘Distribution of Internet Service’ 
#         d. Set the color of the bars to be ‘orange’ 

# In[45]:


plt.figure(figsize=(8, 6))
ax = sns.countplot(x='InternetService', data=df, color='orange')
plt.xlabel('Categories of Internet Service')
plt.ylabel('Count of Categories')
plt.title('Distribution of Internet Service')
plt.show()


# ● Build a histogram for the ‘tenure’ column: 
#     a. Set the number of bins to be 30 
#     b. Set the color of the bins to be ‘green’ 
#     c. Assign the title ‘Distribution of tenure’

# In[49]:


plt.figure(figsize=(8, 6))
plt.hist(df['tenure'], bins=30, color='green', edgecolor='black')
plt.xlabel('Tenure')
plt.ylabel('Frequency')
plt.title('Distribution of tenure')
plt.show()


# Build a scatter-plot between ‘MonthlyCharges’ and ‘tenure’. Map ‘MonthlyCharges’ to the y-axis and ‘tenure’ to the ‘x-axis’: 
#     a. Assign the points a color of ‘brown’ 
#     b. Set the x-axis label to ‘Tenure of customer’ 
#     c. Set the y-axis label to ‘Monthly Charges of customer’ 
#     d. Set the title to ‘Tenure vs Monthly Charges’ 
#     e. Build a box-plot between ‘tenure’ & ‘Contract’. Map ‘tenure’ on the y-axis & 
#     f. ‘Contract’ on the x-axis.

# In[52]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x='tenure', y='MonthlyCharges', data=df, color='brown')
plt.xlabel('Tenure of customer')
plt.ylabel('Monthly Charges of customer')
plt.title('Tenure vs Monthly Charges')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='Contract', y='tenure', data=df)
plt.xlabel('Contract')
plt.ylabel('Tenure of customer')
plt.show()


# Linear Regression

# ● Build a simple linear model where dependent variable is ‘MonthlyCharges’
# and independent variable is ‘tenure’:
# a. Divide the dataset into train and test sets in 70:30 ratio.
# b. Build the model on train set and predict the values on test set
# c. After predicting the values, find the root mean square error
# d. Find out the error in prediction & store the result in ‘error’
# e. Find the root mean square error

# In[54]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[55]:


X = df[['tenure']]
y = df['MonthlyCharges']


# In[56]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[57]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[58]:


y_pred = model.predict(X_test)


# In[59]:


rmse = np.sqrt(mean_squared_error(y_test, y_pred))


# In[60]:


error = y_test - y_pred


# In[61]:


print("Root Mean Square Error:", rmse)


# Logistic Regression: 

# Logistic Regression: ● Build a simple logistic regression model where dependent variable is ‘Churn’ and independent variable is ‘MonthlyCharges’: 
#             a. Divide the dataset in 65:35 ratio 
#             b. Build the model on train set and predict the values on test set 
#             c. Build the confusion matrix and get the accuracy score 
#             d. Build a multiple logistic regression model where dependent variable is ‘Churn’ and independent variables are                ‘tenure’ and ‘MonthlyCharges’ 
#             e. Divide the dataset in 80:20 ratio 
#             f. Build the model on train set and predict the values on test set 
#             g. Build the confusion matrix and get the accuracy score

# In[62]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


# In[64]:


X_simple = df[['MonthlyCharges']]
y_simple = df['Churn']


# In[65]:


X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(X_simple, y_simple, test_size=0.35, random_state=42)


# In[66]:


model_simple = LogisticRegression()
model_simple.fit(X_train_simple, y_train_simple)


# In[67]:


y_pred_simple = model_simple.predict(X_test_simple)


# In[68]:


cm_simple = confusion_matrix(y_test_simple, y_pred_simple)
accuracy_simple = accuracy_score(y_test_simple, y_pred_simple)

print("Confusion Matrix (Simple Logistic Regression):")
print(cm_simple)
print("Accuracy Score (Simple Logistic Regression):", accuracy_simple)


# In[70]:


X_multiple = df[['tenure', 'MonthlyCharges']]
y_multiple = df['Churn']


# In[71]:


X_train_multiple, X_test_multiple, y_train_multiple, y_test_multiple = train_test_split(X_multiple, y_multiple, test_size=0.20, random_state=42)


# In[72]:


model_multiple = LogisticRegression()
model_multiple.fit(X_train_multiple, y_train_multiple)


# In[73]:


y_pred_multiple = model_multiple.predict(X_test_multiple)


# In[74]:


cm_multiple = confusion_matrix(y_test_multiple, y_pred_multiple)
accuracy_multiple = accuracy_score(y_test_multiple, y_pred_multiple)

print("Confusion Matrix (Multiple Logistic Regression):")
print(cm_multiple)
print("Accuracy Score (Multiple Logistic Regression):", accuracy_multiple)


# Decision Tree:  
#     ● Build a decision tree model where dependent variable is ‘Churn’ and independent variable is ‘tenure’:  
#         a. Divide the dataset in 80:20 ratio 
#         b. Build the model on train set and predict the values on test set 
#         c. Build the confusion matrix and calculate the accuracy

# In[75]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# In[76]:


X = df[['tenure']]
y = df['Churn']


# In[77]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[78]:


model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)


# In[79]:


y_pred = model.predict(X_test)


# In[80]:


cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("Accuracy:", accuracy)


# In[ ]:


Random Forest: 
    ● Build a Random Forest model where dependent variable is ‘Churn’ and independent variables are ‘tenure’ and ‘MonthlyCharges’: 
        a. Divide the dataset in 70:30 ratio 
        b. Build the model on train set and predict the values on test set 
        c. Build the confusion matrix and calculate the accuracy


# In[81]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# In[83]:


X = df[['tenure', 'MonthlyCharges']]
y = df['Churn']


# In[84]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[85]:


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# In[86]:


y_pred = model.predict(X_test)


# In[87]:


cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print("Accuracy:", accuracy)


# In[ ]:





# In[ ]:




