#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt 
import matplotlib 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("/Users/deeksha/Desktop/loan_data_set.csv")


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


df.info()


# Pre-Processing

# In[7]:


df.isnull().sum()


# In[8]:


df['LoanAmount']=df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Credit_History']=df['Credit_History'].fillna(df['Credit_History'].mean())
df['LoanAmount_Term']=df['LoanAmount_Term'].fillna(df['LoanAmount_Term'].mean())


# In[9]:


df['Gender']=df['Gender'].fillna(df['Gender'].mode()[0])
df['Married']=df['Married'].fillna(df['Married'].mode()[0])
df['Self_Employed']=df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['Dependent']=df['Dependent'].fillna(df['Dependent'].mode()[0])


# In[10]:


df.isnull().sum()


# In[11]:


sns.countplot(df['Gender']) 


# In[12]:


sns.countplot(df['Married'])


# In[13]:


sns.countplot(df['Dependent'])


# In[14]:


sns.countplot(df['Education'])


# In[15]:


sns.countplot(df['Self_Employed'])


# In[16]:


sns.countplot(df['Property_Area'])


# In[17]:


sns.distplot(df['ApplicantIncome'])


# In[18]:


sns.distplot(df['CoapplicantIncome'])


# In[19]:


corr=df.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr,annot=True,cmap="BuPu")


# In[20]:


df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df.head()


# In[21]:


df['ApplicantIncomelog']=np.log(df['ApplicantIncome'])


# In[22]:


df['LoanAmountlog']=np.log(df['LoanAmount'])


# In[23]:


df['LoanAmount_Term_log']=np.log(df['LoanAmount_Term'])


# In[24]:


df['Total_Incomelog']=np.log(df['Total_Income'])


# In[25]:


sns.distplot(df['LoanAmountlog'])


# In[26]:


corr=df.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr,annot=True,cmap="BuPu")


# In[27]:


cols = ['ApplicantIncome', 'CoapplicantIncome', "LoanAmount", "LoanAmount_Term"]
df = df.drop(columns=cols, axis=1)
df.head()


# In[28]:


df['CS_Income']=[200 if ((x>=9.8)&(x<11.4)) else 150 if((x>=8.4)&(x<9.8)) else 100 for x in df['Total_Incomelog']]
df['CS_Education']=[150 if (x==1) else 90 for x in df['Education']]
df['CS_Employment']=[150 if (x==1) else 100 for x in df['Self_Employed']]
df['CS_Loan']=[200 if (x<4.5) else 140  if ((x<5)&(x>=4.5)) else 100 for x in df['LoanAmountlog']]


# In[29]:


df['Credit_Score']=df['CS_Income']+df['CS_Employment']+df['CS_Education']+df['CS_Loan']


# In[30]:


del df['CS_Income']
del df['CS_Education']
del df['CS_Employment']
del df['CS_Loan']


# In[31]:


df[['Credit_Score']]


# In[32]:


corr=df.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr,annot=True,cmap="BuPu")


# In[33]:


X = df.drop(columns=['LoanStatus'], axis=1)
y = df['LoanStatus']


# In[34]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=77)


# In[35]:


# classify function
from sklearn.model_selection import cross_val_score
def classify(model, x, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=77)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print("Accuracy is", model.score(x_test, y_test)*100)
    score = cross_val_score(model, x, y,cv=5)
    print("Cross validation is",np.mean(score)*100)
    print(f'Confusion Matrix:\n{confusion_matrix(y_test,pred)}')
    print(f'\nClassification Report:\n{classification_report(y_test,pred)}')


# In[36]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
classify(model, X, y)


# In[37]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
classify(model,X,y)


# In[38]:


from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
model=RandomForestClassifier()
classify(model,X,y)


# In[39]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model=LinearDiscriminantAnalysis()
classify(model,X,y)


# In[40]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=30)
classify(model,X,y)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




