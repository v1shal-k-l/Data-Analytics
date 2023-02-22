#!/usr/bin/env python
# coding: utf-8

# ## Zomato Dataset Exploratory Data Analysis

# In[64]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[65]:


df=pd.read_csv("E:\Downloads\Zomatodataset\zomato.csv",encoding='latin-1')
df.head()


# In[66]:


df.info()


# In[67]:


df.describe


# In[68]:


df.isnull().sum()


# In[69]:


##there are nine missing values in Cuisines


# In[70]:


[features for features in df.columns if df[features].isnull().sum()>0]


# In[72]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[80]:


df_country=pd.read_excel("E:\Downloads\Zomatodataset\Country-Code.xlsx")


# In[81]:


df_country.head()


# In[82]:


df.columns


# In[85]:


final_df=pd.merge(df,df_country,on="Country Code",how="left")


# In[89]:


## Required rows and columnns
final_df.head(2)


# In[91]:


## To Check data type
final_df.dtypes


# In[93]:


## Shows the number of Zomato Facilities available
final_df.Country.value_counts()


# In[94]:


country_names=final_df.Country.value_counts().index


# In[95]:


outlets_per_country=final_df.Country.value_counts().values


# In[104]:


## Representation of top 3 countries that uses zomato through Piechart
plt.pie(outlets_per_country[:3],labels=country_names[:3],autopct="%1.2f%%")


# Observation:India is the top user of Zomato , followed by the United States , United Kingdom

# In[116]:


ratings= final_df.groupby(['Aggregate rating','Rating text','Rating color']).size().reset_index().rename(columns={0:"Rating Count"})                                                                                                                                                                       


# In[117]:


ratings


#  When the Ratings were > 4.5 then Excellent 

# In[131]:


## GRAPHICAL REPRESENTATION OF ratings
import matplotlib
matplotlib.rcParams['figure.figsize']=(12,6)
sns.barplot(x="Aggregate rating",y="Rating Count",hue='Rating color',palette=['Blue','red','orange','yellow','green',' dark green'],data=ratings)


# In[132]:


##Count plot
sns.countplot(x="Rating color",data=ratings,palette=['Blue','red','orange','yellow','green',' dark green'])


# In[143]:


## The countries which gave "0" ratings
final_df[final_df['Rating color']=='White'].groupby('Country').size().reset_index()


# Observations:Maximum number of "0" ratings are from India

# In[148]:


## The currency used by the respected Countries
final_df[["Country","Currency"]].groupby(["Country","Currency"]).size().reset_index()


# In[151]:


## Online Delievery Facilities given by which Countries
final_df[["Country","Has Online delivery"]].groupby(["Country","Has Online delivery"]).size().reset_index()


# In[160]:


final_df[final_df['Has Online delivery']=="Yes"].Country.value_counts()


# Observation: Most Online Delivery Facilities are offered in India

# In[162]:


## Pie chart for Cities distribution
final_df.City.value_counts().index


# In[176]:


final_df.City.value_counts().values


# In[173]:


city_values=final_df.City.value_counts().values
city_labels=final_df.City.value_counts().index


# In[179]:


plt.pie(city_values[:5],labels=city_labels[:5],autopct="%1.2f%%")


# In[ ]:




