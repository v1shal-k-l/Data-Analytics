#!/usr/bin/env python
# coding: utf-8

# # Black Friday Dataset

# ## Cleaning and Preparing the Dataset for Model Training

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Problem Statement

# A retail company “ABC Private Limited” wants to understand the customer purchase behaviour (specifically, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month. The data set also contains customer demographics (age, gender, marital status, city_type, stay_in_current_city), product details (product_id and product category) and Total purchase_amount from last month.
# 
# Now, they want to build a model to predict the purchase amount of customer against various products which will help them to create personalized offer for customers against different products.

# In[ ]:


## importing the train dataset
df_train=pd.read_csv("E:/Downloads/archive (1)/Black_Friday_train.csv")
df_train.head()


# In[ ]:


## importing the test dataset
df_test=pd.read_csv("E:/Downloads/archive (1)/Black_Friday_test.csv")
df_test.head()


# In[ ]:


## Merging the datasets
df=pd.merge(df_train,df_test,how="left")
df.head()


# In[ ]:


## Observing 
df.info()


# In[ ]:


## Removing unnecessary columns 
df.drop(['User_ID'],axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


## Handling Categorical Features i.e (Gender)
d1={"F":0,"M":1}
df['Gender']=df['Gender'].map(d1)
df.head()


# In[ ]:


## Handling Categorical Features i.e (Age)
df['Age'].unique()


# In[ ]:


d2={"0-17":1,"18-25":2,"26-35":3,"36-45":4,"46-50":5,"51-55":6,"55+":7}
df['Age']=df['Age'].map(d2)
df.head()  


# In[ ]:


## Handling Categorical Features i.e (City_Category)
df['City_Category'].unique()


# In[ ]:


d3={"A":1,"B":2,"C":3}
df["City_Category"]=df["City_Category"].map(d3)
df.head()


# In[ ]:


## Drop the "City_Category"
df.drop("City_Category",axis=1,inplace=True)
df.head()


# In[ ]:


## Missing Values
df.isnull().sum()


# In[ ]:


## Focusing on missing values
df['Product_Category_2'].unique()


# In[ ]:


(df['Product_Category_2']).mode()[0]


# In[ ]:


## Replacing the missing values using the mode
df['Product_Category_2']=df['Product_Category_2'].fillna(df['Product_Category_2']).mode()[0]


# In[ ]:


## Checking replaced Missing values
df['Product_Category_2'].isnull().sum()


# In[ ]:


## Focusing on missing values
df['Product_Category_3'].unique()


# In[ ]:


df['Product_Category_3'].mode()[0]


# In[ ]:


## Replacing the missing values using the mode
df['Product_Category_3']=df['Product_Category_3'].fillna(df['Product_Category_3']).mode()[0]


# In[ ]:


## Checking replaced Missing values
df['Product_Category_3'].isnull().sum()


# In[ ]:


df["Stay_In_Current_City_Years"].unique()


# In[ ]:


df["Stay_In_Current_City_Years"].str.replace("+","")


# In[ ]:


df.info()


# In[ ]:


## Convert Object into Integer
df["Stay_In_Current_City_Years"]=df["Stay_In_Current_City_Years"].astype(int)


# In[ ]:


## Visualization
sns.pairplot(df)


# In[ ]:


sns.barplot("Age","Purchase",hue="Gender",data=df)


# Obsevation : Purchase of Men is Higher than Women

# In[ ]:


df.head()


# In[ ]:


## Visualisation of purchase with respect to the occupation
sns.barplot("Purchase","Occupation",hue="Gender",data=df)


# In[ ]:





# In[ ]:


## Visualisation of Product_Category_1 with respect to the Purchase
sns.barplot("Product_Category_1","Purchase",hue="Gender",data=df)


# In[ ]:


sns.barplot("Product_Category_2","Purchase",hue="Gender",data=df)


# In[ ]:


sns.barplot("Product_Category_3","Purchase",hue="Gender",data=df)


# In[ ]:


##Feature Scaling 
df_test=df[df['Purchase'].isnull()]


# In[ ]:


df_train=df[~df['Purchase'].isnull()]


# In[ ]:


X=df_train.drop('Purchase',axis=1)


# In[ ]:


X.head()


# In[ ]:


X.shape


# In[ ]:


y=df_train['Purchase']


# In[ ]:


y.shape


# In[ ]:


from sklearn.model_selection import train_test_split
#split the data qet into 75% training and 25% testing
X_train, X_test, y_train, y_test = train_test_split (X, y,test_size=0.2, random_state=0)


# In[ ]:


X_train.drop('Product_ID',axis=1,inplace=True)
X_test.drop('Product_ID',axis=1,inplace=True)


# In[ ]:


#scale the data (feature scaling)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_train = sc.fit_transform(X_test)


# In[ ]:


## Creating Model
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


# In[ ]:


import numpy as np
X_final=(embedded_docs).astype(np.float32)
y_final=(y).astype(np.float32)
print(y_final)


# In[ ]:


## final touching
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)


# In[ ]:




