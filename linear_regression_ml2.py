#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


# In[2]:


df = pd.read_csv(r"C:\Users\akash\OneDrive\Documents\dataset\laptop_data.csv")
df 


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df['ScreenResolution'].unique()


# In[4]:


df['Memory'].unique()


# In[5]:


df['Ram']=df['Ram'].str.replace('GB','')
df['Ram']=df['Ram'].astype(int)


# In[6]:


df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)


# In[7]:


df.drop(columns=['Unnamed: 0'],inplace=True)


# In[8]:


df['Company'].value_counts()


# In[11]:


df['Company'].value_counts().plot.bar(color='orange')
plt.title('Number of laptops per company')


# - Dell and Lenovo have more laptops in market

# In[12]:


df['TypeName'].value_counts()


# - Notebook are the Largest number of type of laptop in market

# In[14]:


df[df['Price']==df['Price'].max()]


# In[15]:


df[df['Price']==df['Price'].min()]


# - Razer's Gaming laptop is the most expansive laptop in market (Price = 324954.72)
# - Acer's notbook is cheapest laptop

# In[16]:


df.groupby(['TypeName','Company'])['Price'].agg(['median','max'])


# In[17]:


df.groupby(['TypeName','Company'])['Price'].agg(['median','max']).plot.bar(figsize=(12,8))


# - The above plot shown laptop compies with each type and also shown the most common price and maximum price.

# - Gaming laptop have more price maximum and median price
# 

# In[18]:


df['OpSys'].value_counts()


# In[19]:


df['OpSys'].value_counts().plot.pie()


# - Windows 10 are most common operating system

# In[20]:


df['Ram'].value_counts()


# In[21]:


df[df['Ram']==df['Ram'].max()]


# In[22]:


df[df['Ram']==df['Ram'].min()].head()


# - 64 GB ram are the maximum used in a laptop, Asus Gaming laptop use this feature.
# - 2Gb ram is the less ram use in some laptops
# - Most of the laptops have 8Gb ram

# In[23]:


df['Gpu'].value_counts()


# In[24]:


df['Gpu'].value_counts().head(10).plot.barh(figsize=(8,3),color='green')
plt.title('Number of cpu and its count')


# - Intel HD Graphics 620  are the commonly used cpu in laptop.

# In[25]:


df['Memory'].value_counts().head()


# In[26]:


df['Memory'].value_counts().head(10).plot.barh(figsize=(8,3),color='skyblue')
plt.title('Most used Memory')


# - 256GB SSD are commonly using Memory storage in laptops.

# In[27]:


df['ScreenResolution'].value_counts()


# In[28]:


df['ScreenResolution'].value_counts().head(10).plot.bar(figsize=(8,3),color='blue')
plt.title('Top ScreenResolution types')


# In[29]:


df[df['Weight']==df['Weight'].max()]


# In[30]:


df[df['Weight']==df['Weight'].min()]


# - Asus Gaming laptop are most weigted laptop in market,it has 4.7 kg weight
# - Lenovo 2 in 1 Convertible type laptops are lite weight laptops

# In[31]:


df.corr(numeric_only=True)


# - There is weak correlation between Laptop price and Ram

# In[32]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()


# In[9]:


df['IPS Panel'] = df['ScreenResolution'].str.contains('IPS Panel').astype(int)
df['Touchscreen'] = df['ScreenResolution'].str.contains('Touchscreen').astype(int)


# In[10]:


df['x resolution'] = df['ScreenResolution'].str.extract(r'(\d+)x').astype(int)
df['y resolution'] = df['ScreenResolution'].str.extract(r'x(\d+)').astype(int)


# In[35]:


df.head()


# In[11]:


df.drop(columns=['ScreenResolution'],inplace=True)


# In[12]:


df['Cpu'].unique()


# In[13]:


df['intel i3'] = df['Cpu'].str.contains('Intel Core i3').astype(int)
df['intel i5'] = df['Cpu'].str.contains('Intel Core i5').astype(int)
df['intel i7'] = df['Cpu'].str.contains('Intel Core i7').astype(int)
df['AMD'] = df['Cpu'].str.contains('AMD').astype(int)


# In[14]:


df['other cpu'] = (~df['Cpu'].str.contains('Intel Core i3|Intel Core i5|Intel Core i7|AMD')).astype(int)


# In[15]:


df.head()


# In[16]:


df['Gpu'].unique()


# In[17]:


df['intel gpu'] = df['Gpu'].str.contains('Intel').astype(int)
df['AMD gpu'] = df['Gpu'].str.contains('AMD').astype(int)
df['Nvidia'] = df['Gpu'].str.contains('Nvidia').astype(int)


# In[18]:


df.drop('Gpu',axis=1,inplace=True)
df.drop('Cpu',axis=1,inplace=True)


# In[19]:


df['Memory'].unique()


# In[20]:


df['SSD']=df['Memory'].str.contains('SSD').astype(int)
df['HDD']=df['Memory'].str.contains('HDD').astype(int)
df['Flash Storage']=df['Memory'].str.contains('Flash Storage').astype(int)
df['Hybrid']=df['Memory'].str.contains('Hybrid').astype(int)


# In[21]:


import re
def convert_to_gb(storage_str):
    # Find all storage amounts in the string
    amounts = re.findall(r'(\d*\.?\d+)([TtGgBb]+)', storage_str)
    total_gb = 0
    for amount, unit in amounts:
        amount = float(amount)
        if 'TB' in unit.upper():
            total_gb += amount * 1024
        elif 'GB' in unit.upper():
            total_gb += amount
       
    return total_gb

df['storage'] = df['Memory'].apply(convert_to_gb)


# In[22]:


df


# In[23]:


df.corr(numeric_only=True).loc[['storage','SSD','HDD','Flash Storage','Hybrid'],['Price']]


# In[24]:


df.corr(numeric_only=True).loc[['storage','SSD','HDD','Flash Storage','Hybrid'],['Price']].plot.bar(color='green')


# - There is a weak positive correlation between SSD storage and laptop price

# In[25]:


df['TypeName'].unique()


# In[26]:


df['Ultrabook'] = df['TypeName'].str.contains('Ultrabook').astype(int)
df['Notebook'] = df['TypeName'].str.contains('Notebook').astype(int)
df['Netbook'] = df['TypeName'].str.contains('Netbook').astype(int)
df['Gaming'] = df['TypeName'].str.contains('Gaming').astype(int)
df['Convertible'] = df['TypeName'].str.contains('2 in 1 Convertible').astype(int)
df['Workstation'] = df['TypeName'].str.contains('Workstation').astype(int)


# In[27]:


df['OpSys'].unique()


# In[28]:


df['macOS'] = df['TypeName'].str.contains('macOS|Mac OS X').astype(int)
df['No OS'] = df['TypeName'].str.contains('No OS').astype(int)
df['Windows'] = df['TypeName'].str.contains('Windows 10|Windows 10 S|Windows 7').astype(int)
df['Linux'] = df['TypeName'].str.contains('Linux').astype(int)
df['Android'] = df['TypeName'].str.contains('Android').astype(int)
df['Chrome OS'] = df['TypeName'].str.contains('Chrome OS').astype(int)


# In[29]:


df['Company'].unique()


# In[30]:


df['Apple'] = df['Company'].str.contains('Apple').astype(int)
df['HP'] = df['Company'].str.contains('HP').astype(int)
df['Acer'] = df['Company'].str.contains('Acer').astype(int)
df['Asus'] = df['Company'].str.contains('Asus').astype(int)

df['Dell'] = df['Company'].str.contains('Dell').astype(int)
df['Lenovo'] = df['Company'].str.contains('Lenovo').astype(int)
df['Chuwi'] = df['Company'].str.contains('Chuwi').astype(int)
df['MSI'] = df['Company'].str.contains('MSI').astype(int)

df['Microsoft'] = df['Company'].str.contains('Microsoft').astype(int)
df['Toshiba'] = df['Company'].str.contains('Toshiba').astype(int)
df['Huawei'] = df['Company'].str.contains('Huawei').astype(int)
df['Xiaomi'] = df['Company'].str.contains('Xiaomi').astype(int)

df['Vero'] = df['Company'].str.contains('Vero').astype(int)
df['Razer'] = df['Company'].str.contains('Razer').astype(int)
df['Mediacom'] = df['Company'].str.contains('Mediacom').astype(int)
df['Samsung'] = df['Company'].str.contains('Samsung').astype(int)

df['Google'] = df['Company'].str.contains('Google').astype(int)
df['Fujitsu'] = df['Company'].str.contains('Fujitsu').astype(int)
df['LG'] = df['Company'].str.contains('LG').astype(int)


# In[31]:


df.drop(columns=['Company','TypeName','OpSys'],inplace=True)


# In[32]:


df.head()


# In[33]:


df.drop('Memory',axis=1,inplace=True)


# In[34]:


df


# In[35]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[36]:


df['x resolution'] = scaler.fit_transform(df[['x resolution']])
df['y resolution'] = scaler.fit_transform(df[['y resolution']])
df.head()


# In[37]:


df['Inches'].unique()


# In[38]:


from sklearn.preprocessing import MinMaxScaler
mmscaler =MinMaxScaler()
df['Inches'] = mmscaler.fit_transform(df[['Inches']])
df.head()


# In[39]:


df['Ram'] = mmscaler.fit_transform(df[['Ram']])
df['Weight'] = mmscaler.fit_transform(df[['Weight']])
df['storage'] = mmscaler.fit_transform(df[['storage']])


# In[40]:


df


# In[41]:


from sklearn.model_selection import train_test_split


# In[42]:


X = df.drop('Price',axis=1)
y = df['Price']


# In[43]:


X.head()


# In[44]:


y.head()


# In[45]:


X_train, X_test, y_train, y_test = train_test_split (X,y, test_size = 0.2, random_state = 7)


# In[46]:


X_train


# In[47]:


y_train


# In[48]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)


# In[49]:


y_pred = lin_reg.predict(X_test)


# In[50]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[51]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,y_pred)


# **Accuracy of the predection of the model is 75.7 %**

# In[54]:


# with open('laptop.pkl', 'wb') as file:
#     pickle.dump(lin_reg, file)
# with open('scaler.pkl', 'wb') as file:
#     pickle.dump(scaler, file)
# with open('mmscaler.pkl', 'wb') as file:
#     pickle.dump(mmscaler, file)
# with open('feature_names.pkl', 'wb') as file:
#     pickle.dump(X.columns.tolist(), file)

# pickle.load(open('laptop_price_model.pkl', 'wb'))
# In[ ]:
with open('laptop_price_model.pkl', 'wb') as file:
    pickle.dump(lin_reg, file)
with open('feature_names.pkl', 'wb') as file:
    pickle.dump(X.columns.tolist(),file)


