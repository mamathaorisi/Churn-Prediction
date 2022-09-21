#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("Telco Churn.csv")
df.sample(5)


# In[ ]:


df.drop('customerID', axis='columns',inplace=True)
df.dtypes


# In[ ]:


df.TotalCharges.values


# In[ ]:


df.MonthlyCharges.values 


# In[ ]:


pd.to_numeric(df.TotalCharges)


# In[ ]:


df[pd.to_numeric(df.TotalCharges,errors='coerce').isnull()]


# In[ ]:


df.shape


# In[ ]:


df.iloc[488]['TotalCharges']


# In[ ]:


df1 = df[df.TotalCharges!=' ']
df1.shape


# In[ ]:


df1.dtypes


# In[ ]:


pd.to_numeric(df1.TotalCharges)


# In[ ]:


df1.TotalCharges = pd.to_numeric(df1.TotalCharges)


# In[ ]:


df1.TotalCharges.dtypes


# In[ ]:


tenure_churn_no = df1[df1.Churn=='No'].tenure
tenure_churn_yes = df1[df1.Churn=='Yes'].tenure

plt.xlabel("tenure")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visulaization")

plt.hist([tenure_churn_no, tenure_churn_yes], color=['black','skyblue'],label=['Churn=No','Churn=Yes'])
plt.legend()


# In[ ]:


mc_churn_no = df1[df1.Churn=='No'].MonthlyCharges
mc_churn_yes = df1[df1.Churn=='Yes'].MonthlyCharges

plt.xlabel("Monthly Charges")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visulaization")

blood_sugar_men = [113, 85, 90, 150, 149, 88, 93, 115, 135, 80, 77, 82, 129]
blood_sugar_women = [67, 98, 89, 120, 133, 150, 84, 69, 79, 120, 112, 100]

plt.hist([mc_churn_yes, mc_churn_no], rwidth=0.95, color=['pink','grey'],label=['Churn=Yes','Churn=No'])
plt.legend()


# In[ ]:


def print_unique_col_values(df):
    for column in df:
        if df[column].dtypes=='object':
            print(f'{column} : {df[column].unique()}')
 


# In[ ]:


print_unique_col_values(df1)


# In[ ]:


df1.replace('No internet service','No', inplace=True)
df1.replace('No phone service','No', inplace=True)

print_unique_col_values(df1)


# In[ ]:


yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection',
                 'TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
for col in yes_no_columns:
    df1[col].replace({"Yes": 1,"No": 0},inplace=True)


# In[ ]:


for col in df1:
    print(f'{col}: {df1[col].unique()}')


# as seen from the output all the yes and no are replaced with o and 1, now coming to female and male column encoding with 0 and 1

# In[ ]:


df1['gender'].replace({'Female':1,'Male':0},inplace=True)


# In[ ]:


df1['gender'].unique()


# In[ ]:


pd.get_dummies(df1,columns=['InternetService','Contract','PaymentMethod'])


# In[ ]:


df2 = pd.get_dummies(df1,columns=['InternetService','Contract','PaymentMethod'])
df2.columns


#  And now skilling of the dataset is required because tenure, monthly charges and total charges as they are in still in their values other than 0 and 1 range and for scaling them iam using minmax scaler.
#  so here preprocessing is done, for storing those columns in the data frame iam using fit in transform function.
# 
# 
# 

# In[ ]:


cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])


# In[ ]:


df2.head()


# In[ ]:


X = df2.drop('Churn',axis='columns')
y = df2['Churn']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


X_train[:10]


# In[ ]:


len(X_train.columns) 


# In[ ]:


import tensorflow as tf
from tensorflow import keras
model = keras.Sequential(
    [
        keras.layers.Dense(20, input_shape=(26,), activation='relu'),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy", 
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs = 10)


# In[ ]:


model.evaluate(X_test, y_test)


# In[ ]:


yp = model.predict(X_test)
yp[:10]


# In[ ]:


y_pred = []
for element in yp:
  if element > 0.5:
      y_pred.append(1)
  else:
      y_pred.append


# In[ ]:


y_pred[:5]


# In[ ]:


from sklearn.linear_model import LogisticRegression
lm = LogisticRegression(random_state=0, max_iter=1000, solver='lbfgs', class_weight='balanced')
lm.fit(X_train, y_train)


# In[ ]:


y_pred = lm.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report 
print(classification_report(y_test, y_pred))


# In[ ]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))


# In[ ]:


import seaborn as sn
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred)

plt.figure(figsize = (10,7))
sn.heatmap(cm,annot=True, fmt='d')
plt.xlabel('predicted')
plt.ylabel('Truth')


# In[ ]:


(714+335)/(714+285+73+335)


# In[ ]:


(714/714+73)


# In[ ]:


(334/285+73)

