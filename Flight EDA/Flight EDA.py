#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


train_df=pd.read_excel('Data_Train.xlsx')
train_df.head()


# In[6]:


test_df=pd.read_excel('Test_set.xlsx')
test_df.head()


# In[7]:


final_df=train_df.append(test_df)
final_df.head()


# In[8]:


final_df.tail()


# In[9]:


final_df["Date_of_Journey"].str.split("/").str[0]


# In[10]:


final_df['Date']=final_df["Date_of_Journey"].str.split("/").str[0]
final_df['Month']=final_df["Date_of_Journey"].str.split("/").str[1]
final_df['Year']=final_df["Date_of_Journey"].str.split("/").str[2]
final_df=final_df.drop("Date_of_Journey",axis=1)


# In[11]:


final_df.head(3)


# In[12]:


final_df["Date"]=final_df["Date"].astype(int)
final_df["Month"]=final_df["Month"].astype(int)
final_df["Year"]=final_df["Year"].astype(int)


# In[13]:


final_df.info()


# In[14]:


final_df["Arrival_Time"]=final_df["Arrival_Time"].str.split(" ").str[0]


# In[15]:


final_df.isnull().sum()


# In[16]:


final_df["Arrival_Hours"]=final_df["Arrival_Time"].str.split(":").str[0]
final_df["Arrival_Minutes"]=final_df["Arrival_Time"].str.split(":").str[1]
final_df["Arrival_Hours"]=final_df["Arrival_Hours"].astype(int)
final_df["Arrival_Minutes"]=final_df["Arrival_Minutes"].astype(int)
final_df=final_df.drop("Arrival_Time",axis=1)


# In[17]:


final_df.head()


# In[18]:


final_df["Dep_Time"]=final_df["Dep_Time"].str.split(":").str[0]


# In[19]:


final_df["Dept_Hours"]=final_df["Dep_Time"].str.split(":").str[0]
final_df["Dept_Minutes"]=final_df["Dep_Time"].str.split(":").str[1]
final_df["Dept_Hours"]=final_df["Dept_Hours"].astype("Int64")
final_df["Dept_Minutes"]=final_df["Dept_Minutes"].astype("Int64")
final_df=final_df.drop("Dep_Time",axis=1)


# In[20]:


final_df.info()


# In[21]:


final_df["Total_Stops"].unique()


# In[22]:


final_df["Total_Stops"]=final_df["Total_Stops"].map({"non-stop":0,"1 stop":1, " 2 stops":2,"3 stops":3,"4 stops":4 })


# In[23]:


final_df[final_df["Total_Stops"].isnull()]
final_df.head(1)


# In[24]:


'final_df["Additional_Info"].unique()


# In[30]:


final_df["Duration_Hour"]=final_df["Duration"].str.split(" ").str[0].str.split('h').str[0]


# In[31]:


final_df.head()


# In[46]:


final_df[final_df["Duration_Hour"]=="5m"]


# In[50]:


final_df.drop(6474,axis=0,inplace=True)


# In[51]:


final_df.drop(2660,axis=0,inplace=True)


# In[52]:


final_df["Duration_Hour"]=final_df["Duration_Hour"].astype('int')


# In[53]:


final_df["Duration_Hour"]*60


# In[54]:


final_df.drop("Duration",axis=1,inplace=True)


# In[55]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()


# In[59]:


final_df["Airline"]=labelencoder.fit_transform(final_df['Airline'])
final_df["Source"]=labelencoder.fit_transform(final_df['Source'])
final_df["Destination"]=labelencoder.fit_transform(final_df['Destination'])
final_df["Additional_Info"]=labelencoder.fit_transform(final_df["Additional_Info"])


# In[61]:


final_df.head()


# In[66]:


pd.get_dummies(final_df,columns=["Airline", "Source", "Destination"] ,drop_first = True)


# In[ ]:




