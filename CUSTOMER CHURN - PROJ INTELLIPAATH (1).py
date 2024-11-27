#!/usr/bin/env python
# coding: utf-8

#  Problem Statement:
#  You are the data scientist at a telecom company named “Neo” whose customers
#  are churning out to its competitors. You have to analyze the data of your
#  company and find insights and stop your customers from churning out to other
#  telecom companies

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv(r"C:\Users\DELL\Downloads\customer_churn.csv")


# In[3]:


df


# In[4]:


customer_5 = df.iloc[:, 4]  # Using iloc to select the 5th column (index 4)


# In[5]:


customer_5


# In[6]:


customer_15 = df.iloc[:, 14]


# In[7]:


customer_15


# In[8]:


senior_male_electronic = df[(df['gender'] == 'Male') & (df['SeniorCitizen'] == 1) & (df['PaymentMethod'] == 'Electronic check')]


# In[9]:


senior_male_electronic


# In[10]:


customer_total_tenure = df[(df['tenure'] > 70) | (df['MonthlyCharges'] > 100)]


# In[11]:


customer_total_tenure


# In[12]:


two_mail_yes = df[(df['Contract']== 'Two year') & (df['PaymentMethod'] == 'Mailed check') & (df['Churn']== 'Yes')]


# In[13]:


two_mail_yes


# In[14]:


customer_333 = df.sample(333)


# In[15]:


customer_333


# In[16]:


churn = df['Churn'].value_counts()


# In[17]:


churn


# In[18]:


counts = df['InternetService'].value_counts()


# In[19]:


counts.plot(kind='bar', color= 'orange')
plt.title('Distribution of Internet Service')
plt.xlabel('‘Categories of Internet Service')
plt.ylabel('‘Count of Categories')
plt.show()


# 
# The bar plot shows that Fiber Optics has a higher count compared to DSL, indicating a greater prevalence of Fiber Optic internet service in the dataset

# In[20]:


# Create the histogram
plt.hist(df['tenure'], bins=30, color='green')
# Set the title and labels
plt.title('Distribution of Tenure')
plt.xlabel('Tenure')
plt.ylabel('Frequency')


# In[21]:


# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['tenure'], df['MonthlyCharges'], color='brown')

# Set the x and y axis labels and the title
plt.xlabel('Tenure of Customer')
plt.ylabel('Monthly Charges of Customer')
plt.title('Tenure vs Monthly Charges')

# Show the plot
plt.show()


# # Linear Regression:
#  ● Build a simple linear model where dependent variable is ‘MonthlyCharges’
#  and independent variable is ‘tenure’:

# In[22]:


#Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error


# In[23]:


X = df[['tenure']]  # Independent variable
y = df['MonthlyCharges']  # Dependent variable


# In[24]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=42)


# In[25]:


lr_model = LinearRegression()


# In[26]:


# Fit the model to the training data
lr_model.fit(X_train, y_train)


# In[27]:


y_pred = lr_model.predict(X_test)


# In[28]:


# Step 5: Calculate the errors
error = y_test - y_pred  # Error in predictions

# Step 6: Calculate Root Mean Square Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Output results
print("Error in Prediction:\n", error)
print("Root Mean Square Error (RMSE):", rmse)


# # Logistic Regression:
#  ● Build a simple logistic regression model where dependent variable is
#  ‘Churn’ and independent variable is ‘MonthlyCharges’:

# In[29]:


X = df[['tenure', 'MonthlyCharges']]      #independent variable
y = df['Churn']                           #dependent variable


# In[30]:


churn = df


# In[31]:


from sklearn.linear_model import LogisticRegression


# In[32]:


log_model =LogisticRegression()


# In[33]:


log_model


# In[34]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state=42)


# In[35]:


log_model.fit(X_train,y_train)


# In[36]:


y_pred = log_model.predict(X_test)


# In[37]:


conf_matrix = confusion_matrix(y_test,y_pred)
accuracy_score = accuracy_score(y_test,y_pred)


# In[38]:


# Output results
print("Confusion Matrix for Simple Logistic Regression:\n", conf_matrix)
print("Accuracy Score for Simple Logistic Regression:",accuracy_score)


# # Decision Tree:
#  ● Build a decision tree model where dependent variable is ‘Churn’ and
#  independent variable is ‘tenure’:

# In[39]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# In[40]:


# Convert 'Churn' to numerical values (1 for 'Yes', 0 for 'No')
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})


# In[41]:


# Check for NaN values in 'Churn' and handle them
if df['Churn'].isnull().sum() > 0:
    df = df.dropna(subset=['Churn'])  # Drop NaN values if any


# In[42]:


# Verify there are no NaN values in 'Churn'
print("NaN values in 'Churn':", df['Churn'].isnull().sum())


# In[43]:


# Check for other NaN values in the dataset
print("NaN values in each column:\n", df.isnull().sum())


# In[44]:


# Step 3: Split the data into train and test sets (80:20 ratio)
X = df[['tenure']]  # Independent variable
y = df['Churn']  # Dependent variable


# In[45]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[46]:


# Step 4: Build the decision tree model on the train set
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)


# In[47]:


# Step 5: Predict the values on the test set
y_pred = dt_model.predict(X_test)


# In[48]:


# Step 6: Build the confusion matrix and calculate the accuracy score
confusion = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)


# In[49]:


# Output results
print("Confusion Matrix:\n", confusion)
print("Accuracy Score:", accuracy)


# #  Random Forest:
#  ● Build a Random Forest model where dependent variable is ‘Churn’ and
#  independent variables are ‘tenure’ and ‘MonthlyCharges’:

# In[50]:


X = df[['tenure', 'MonthlyCharges' ]]  # Independent variable
y = df['Churn']  # Dependent variable


# In[51]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score


# In[52]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[53]:


rt_model =RandomForestClassifier()


# In[54]:


rt_model.fit(X_train,y_train)


# In[55]:


y_pred = rt_model.predict(X_test)


# In[56]:


conf_matrix = confusion_matrix(y_pred,y_test)
acc_score = accuracy_score(y_test,y_pred)


# In[57]:


print("Confusion Matrix:")
print(conf_matrix)


# In[59]:


print(f"Accuracy Score: {acc_score:.2f}")


# Based on the performance metrics, Logistic Regression with an accuracy of 79.77% outperforms both Decision Tree (75.73%) and Random Forest (76%), making it the most effective model for prediction in this project.

# In[ ]:




