#!/usr/bin/env python
# coding: utf-8

# In[5]:


## Importing necessary libraries


# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.metrics import accuracy_score,confusion_matrix


# In[7]:


df=pd.read_csv('/Users/priyankaswain/Desktop/AQI Predict/data.csv',encoding='unicode_escape')
# Reading the dataset


# ## Data Understanding

# In[8]:


df.head()
# Loading the dataset


# In[9]:


df.shape
# As we can see that there are 4,35,742 rows and 13 columns in the dataset


# In[10]:


df.info()
# Checking the over all information on the dataset.


# In[11]:


df.isnull().sum()
# There are a lot of missing values present in the dataset


# In[12]:


df.describe()
# Checking the descriptive stats of the numeric values present in the data like mean, standard deviation, min values and max value present in the data


# In[13]:


df.nunique()
# These are all the unique values present in the dataframe


# In[14]:


df.columns
# These are all the columns present in the dataset.


# stn_code (station code)
# sampling_date (date of sample collection)
# state (Indian State)
# location (location of sample collection)
# agency
# type (type of area)
# so2 (sulphur dioxide concentration)
# no2 (nitrogen dioxide concentration)
# rspm (respirable suspended particualte matter concentration)
# spm (suspended particulate matter)
# location_monitoring_station
# pm2_5 (particulate matter 2.5)
# date (date)

# ## Data Visualization

# In[15]:


sns.pairplot(data=df)


# In[16]:


df['state'].value_counts()
# Viewing the count of values present in the state column


# In[17]:


plt.figure(figsize=(15, 6))
plt.xticks(rotation=90)
df.state.hist()
plt.xlabel('state')
plt.ylabel('Frequencies')
plt.plot()
# The visualization shows us the count of states present in the dataset.


# In[18]:


df['type'].value_counts()
# Viewing the count of values present in the type column


# In[19]:


plt.figure(figsize=(15, 6))
plt.xticks(rotation=90)
df.type.hist()
plt.xlabel('Type')
plt.ylabel('Frequencies')
plt.plot()
# The visualization shows us the count of Types present in the dataset.


# In[20]:


df['agency'].value_counts()
# Viewing the counts of values present in the agency column


# In[21]:


plt.figure(figsize=(15, 6))
plt.xticks(rotation=90)
df.agency.hist()
plt.xlabel('Agency')
plt.ylabel('Frequencies')
plt.plot()
# The visualization shows us the count of Agency present in the dataset.


# In[22]:


plt.figure(figsize=(30, 10))
plt.xticks(rotation=90)
sns.barplot(x='state',y='so2',data=df);
# This visualization shows the name of the state having higher so2 levels in the air which is Uttaranchal followed by Uttarakhand


# In[23]:


plt.rcParams['figure.figsize']=(30,10)


# In[24]:


df[['so2','state']].groupby(["state"]).mean().sort_values(by='so2').plot.bar(color='purple')
plt.show()
# We can also use the groupby function to sort values in an ascending order based on the x-axis, y-axis and its keys
# Below we get a clear picture of the states in an increasing order based on their so2 levels.


# In[25]:


plt.figure(figsize=(30, 10))
plt.xticks(rotation=90)
sns.barplot(x='state',y='no2',data=df);
# West bengal has a higher no2 level compared to other states 


# In[26]:


df[['no2','state']].groupby(["state"]).mean().sort_values(by='no2').plot.bar(color='purple')
plt.show()
# We can also use the groupby function to sort values in an ascending order based on the x-axis, y-axis and its keys
# Below we get a clear picture of the states in an increasing order based on their no2 levels.


# In[27]:


plt.figure(figsize=(30, 10))
plt.xticks(rotation=90)
sns.barplot(x='state',y='rspm',data=df);
# Delhi has higher rspm level compared to other states 


# In[28]:


plt.figure(figsize=(30, 10))
plt.xticks(rotation=90)
sns.barplot(x='state',y='spm',data=df);
# Delhi has higher spm level compared to other states 


# In[29]:


plt.figure(figsize=(30, 10))
plt.xticks(rotation=90)
sns.barplot(x='state',y='pm2_5',data=df);
# Delhi has higher pm2_5 level compared to other states 


# ### Checking all null values and treating those null values.

# In[30]:


nullvalues = df.isnull().sum().sort_values(ascending=False)
# Checking all null values


# In[31]:


nullvalues
# higher null values present in pm2_5 followed by spm


# In[32]:


null_values_percentage = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending=False)
#count(returns Non-NAN value)


# In[33]:


missing_data_with_percentage = pd.concat([nullvalues, null_values_percentage], axis=1, keys=['Total', 'Percent'])
# Concatenating total null values and their percentage of missing values for further imputation or column deletion


# In[34]:


missing_data_with_percentage
# As you can see below these are the percentages of null values present in the dataset


# In[35]:


df.drop(['agency'],axis=1,inplace=True)
df.drop(['stn_code'],axis=1,inplace=True)
df.drop(['date'],axis=1,inplace=True)
df.drop(['sampling_date'],axis=1,inplace=True)
df.drop(['location_monitoring_station'],axis=1,inplace=True)
# Dropping unnecessary columns


# In[36]:


df.isnull().sum()
# Now checking the null values


# In[37]:


df


# In[38]:


df['location']=df['location'].fillna(df['location'].mode()[0])
df['type']=df['type'].fillna(df['type'].mode()[0])
# Null value Imputation for categorical data


# In[39]:


df.fillna(0, inplace=True)
# null values are replaced with zeros for the numerical data


# In[40]:


df.isnull().sum()
# Now we have successfully imputed null values which were present in the dataset


# In[41]:


df
# The following features are important for our machine learning models.


# # CALCULATE AIR QUALITY INDEX FOR SO2 BASED ON FORMULA
# The air quality index is a piecewise linear function of the pollutant concentration. At the boundary between AQI categories, there is a discontinuous jump of one AQI unit. To convert from concentration to AQI this equation is used

# ### Function to calculate so2 individual pollutant index(si)

# In[42]:


def cal_SOi(so2):
    si=0
    if (so2<=40):
     si= so2*(50/40)
    elif (so2>40 and so2<=80):
     si= 50+(so2-40)*(50/40)
    elif (so2>80 and so2<=380):
     si= 100+(so2-80)*(100/300)
    elif (so2>380 and so2<=800):
     si= 200+(so2-380)*(100/420)
    elif (so2>800 and so2<=1600):
     si= 300+(so2-800)*(100/800)
    elif (so2>1600):
     si= 400+(so2-1600)*(100/800)
    return si
df['SOi']=df['so2'].apply(cal_SOi)
data= df[['so2','SOi']]
data.head()
# calculating the individual pollutant index for so2(sulphur dioxide)


# ### Function to calculate no2 individual pollutant index(ni)

# In[43]:


def cal_Noi(no2):
    ni=0
    if(no2<=40):
     ni= no2*50/40
    elif(no2>40 and no2<=80):
     ni= 50+(no2-40)*(50/40)
    elif(no2>80 and no2<=180):
     ni= 100+(no2-80)*(100/100)
    elif(no2>180 and no2<=280):
     ni= 200+(no2-180)*(100/100)
    elif(no2>280 and no2<=400):
     ni= 300+(no2-280)*(100/120)
    else:
     ni= 400+(no2-400)*(100/120)
    return ni
df['Noi']=df['no2'].apply(cal_Noi)
data= df[['no2','Noi']]
data.head()
# calculating the individual pollutant index for no2(nitrogen dioxide)


# ### Function to calculate rspm individual pollutant index(rpi)

# In[44]:


def cal_RSPMI(rspm):
    rpi=0
    if(rpi<=30):
     rpi=rpi*50/30
    elif(rpi>30 and rpi<=60):
     rpi=50+(rpi-30)*50/30
    elif(rpi>60 and rpi<=90):
     rpi=100+(rpi-60)*100/30
    elif(rpi>90 and rpi<=120):
     rpi=200+(rpi-90)*100/30
    elif(rpi>120 and rpi<=250):
     rpi=300+(rpi-120)*(100/130)
    else:
     rpi=400+(rpi-250)*(100/130)
    return rpi
df['Rpi']=df['rspm'].apply(cal_RSPMI)
data= df[['rspm','Rpi']]
data.head()
# calculating the individual pollutant index for rspm(respirable suspended particualte matter concentration)


# ### Function to calculate spm individual pollutant index(spi)

# In[45]:


def cal_SPMi(spm):
    spi=0
    if(spm<=50):
     spi=spm*50/50
    elif(spm>50 and spm<=100):
     spi=50+(spm-50)*(50/50)
    elif(spm>100 and spm<=250):
     spi= 100+(spm-100)*(100/150)
    elif(spm>250 and spm<=350):
     spi=200+(spm-250)*(100/100)
    elif(spm>350 and spm<=430):
     spi=300+(spm-350)*(100/80)
    else:
     spi=400+(spm-430)*(100/430)
    return spi
   
df['SPMi']=df['spm'].apply(cal_SPMi)
data= df[['spm','SPMi']]
data.head()
# calculating the individual pollutant index for spm(suspended particulate matter)


# ### function to calculate the air quality index (AQI) of every data value

# In[46]:


def cal_aqi(si,ni,rspmi,spmi):
    aqi=0
    if(si>ni and si>rspmi and si>spmi):
     aqi=si
    if(ni>si and ni>rspmi and ni>spmi):
     aqi=ni
    if(rspmi>si and rspmi>ni and rspmi>spmi):
     aqi=rspmi
    if(spmi>si and spmi>ni and spmi>rspmi):
     aqi=spmi
    return aqi

df['AQI']=df.apply(lambda x:cal_aqi(x['SOi'],x['Noi'],x['Rpi'],x['SPMi']),axis=1)
data= df[['state','SOi','Noi','Rpi','SPMi','AQI']]
data.head()
# Caluclating the Air Quality Index.


# In[47]:


def AQI_Range(x):
    if x<=50:
        return "Good"
    elif x>50 and x<=100:
        return "Moderate"
    elif x>100 and x<=200:
        return "Poor"
    elif x>200 and x<=300:
        return "Unhealthy"
    elif x>300 and x<=400:
        return "Very unhealthy"
    elif x>400:
        return "Hazardous"

df['AQI_Range'] = df['AQI'] .apply(AQI_Range)
df.head()
# Using threshold values to classify a particular values as good, moderate, poor, unhealthy, very unhealthy and Hazardous


# In[48]:


df['AQI_Range'].value_counts()
# These are the counts of values present in the AQI_Range column.


# ### Splitting the dataset into Dependent and Independent columns

# In[49]:


X=df[['SOi','Noi','Rpi','SPMi']]
Y=df['AQI']
X.head()
# we only select columns like soi, noi, rpi, spmi


# In[50]:


Y.head()
# the AQI column is the target column


# In[51]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=70)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
# splitting the data into training and testing data


# ### Linear Regression

# In[52]:


model=LinearRegression()
model.fit(X_train,Y_train)


# In[53]:


#predicting train
train_pred=model.predict(X_train)
#predicting on test
test_pred=model.predict(X_test)


# In[54]:


RMSE_train=(np.sqrt(metrics.mean_squared_error(Y_train,train_pred)))
RMSE_test=(np.sqrt(metrics.mean_squared_error(Y_test,test_pred)))
print("RMSE TrainingData = ",str(RMSE_train))
print("RMSE TestData = ",str(RMSE_test))
print('-'*50)
print('RSquared value on train:',model.score(X_train, Y_train))
print('RSquared value on test:',model.score(X_test, Y_test))


# ### Decision Tree Regressor

# In[55]:


DT=DecisionTreeRegressor()
DT.fit(X_train,Y_train)


# In[56]:


#predicting train
train_preds=DT.predict(X_train)
#predicting on test
test_preds=DT.predict(X_test)


# In[57]:


RMSE_train=(np.sqrt(metrics.mean_squared_error(Y_train,train_preds)))
RMSE_test=(np.sqrt(metrics.mean_squared_error(Y_test,test_preds)))
print("RMSE TrainingData = ",str(RMSE_train))
print("RMSE TestData = ",str(RMSE_test))
print('-'*50)
print('RSquared value on train:',DT.score(X_train, Y_train))
print('RSquared value on test:',DT.score(X_test, Y_test))


# ### Random Forest Regressor

# In[58]:


RF=RandomForestRegressor().fit(X_train,Y_train)


# In[59]:


#predicting train
train_preds1=RF.predict(X_train)
#predicting on test
test_preds1=RF.predict(X_test)


# In[60]:


RMSE_train=(np.sqrt(metrics.mean_squared_error(Y_train,train_preds1)))
RMSE_test=(np.sqrt(metrics.mean_squared_error(Y_test,test_preds1)))
print("RMSE TrainingData = ",str(RMSE_train))
print("RMSE TestData = ",str(RMSE_test))
print('-'*50)
print('RSquared value on train:',RF.score(X_train, Y_train))
print('RSquared value on test:',RF.score(X_test, Y_test))


# # Classification Algorithms

# In[61]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[62]:


X2 = df[['SOi','Noi','Rpi','SPMi']]
Y2 = df['AQI_Range']
# Splitting the data into independent and dependent columns for classification 


# In[63]:


df['AQI_Range'].value_counts()


# In[64]:


X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, Y2, test_size=0.33, random_state=70)
# Splitting the data into training and testing data 


# ### Logistic Regression

# In[65]:


#fit the model on train data 
log_reg = LogisticRegression().fit(X_train2, Y_train2)

#predict on train 
train_preds2 = log_reg.predict(X_train2)
#accuracy on train
print("Model accuracy on train is: ", accuracy_score(Y_train2, train_preds2))

#predict on test
test_preds2 = log_reg.predict(X_test2)
#accuracy on test
print("Model accuracy on test is: ", accuracy_score(Y_test2, test_preds2))
print('-'*50)

# Kappa Score.
print('KappaScore is: ', metrics.cohen_kappa_score(Y_test2,test_preds2))


# In[66]:


log_reg.predict([[727,327.55,78.2,100]]) 


# In[67]:


log_reg.predict([[2.7,45,35.16,23]]) 


# In[68]:


log_reg.predict([[10,2.8,82,20]]) 


# In[69]:


log_reg.predict([[2,45.8,37,32]])


# ### Decision Tree Classifier

# In[70]:


#fit the model on train data 
DT2 = DecisionTreeClassifier().fit(X_train2,Y_train2)

#predict on train 
train_preds3 = DT2.predict(X_train2)
#accuracy on train
print("Model accuracy on train is: ", accuracy_score(Y_train2, train_preds3))

#predict on test
test_preds3 = DT2.predict(X_test2)
#accuracy on test
print("Model accuracy on test is: ", accuracy_score(Y_test2, test_preds3))
print('-'*50)

# Kappa Score
print('KappaScore is: ', metrics.cohen_kappa_score(Y_test2,test_preds3))


# ### Random Forest Classifier

# In[71]:


#fit the model on train data 
RF=RandomForestClassifier().fit(X_train2,Y_train2)
#predict on train 
train_preds4 = RF.predict(X_train2)
#accuracy on train
print("Model accuracy on train is: ", accuracy_score(Y_train2, train_preds4))

#predict on test
test_preds4 = RF.predict(X_test2)
#accuracy on test
print("Model accuracy on test is: ", accuracy_score(Y_test2, test_preds4))
print('-'*50)

# Kappa Score
print('KappaScore is: ', metrics.cohen_kappa_score(Y_test2,test_preds4))


# ### K-Nearest Neighbours

# In[72]:


#fit the model on train data 
KNN = KNeighborsClassifier().fit(X_train2,Y_train2)
#predict on train 
train_preds5 = KNN.predict(X_train2)
#accuracy on train
print("Model accuracy on train is: ", accuracy_score(Y_train2, train_preds5))

#predict on test
test_preds5 = KNN.predict(X_test2)
#accuracy on test
print("Model accuracy on test is: ", accuracy_score(Y_test2, test_preds5))
print('-'*50)

# Kappa Score
print('KappaScore is: ', metrics.cohen_kappa_score(Y_test2,test_preds5))


# In[73]:


KNN.predict([[7.4,47.7,78.182,100]]) 
# Predictions on random values


# In[74]:


KNN.predict([[1,1.2,3.12,0]]) 
# Predictions on random values


# In[75]:


KNN.predict([[325.7,345,798.182,203]]) 
# Predictions on random values


# In[76]:


from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test2,test_preds4)


# In[77]:


import seaborn as sns
import matplotlib as ply
get_ipython().run_line_magic('matplotlib', 'inline')


# In[78]:


sns.heatmap(confusion_matrix(Y_test2,test_preds4),cmap="BuPu")


# In[79]:


sns.heatmap(confusion_matrix(Y_test2,test_preds3),cmap="Greens")


# In[80]:


from xgboost import XGBClassifier
print("XGBoost Classifier imported successfully.")


# In[81]:


from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
le = LabelEncoder()

# Fit and transform Y_train2 and transform Y_test2
Y_train2_encoded = le.fit_transform(Y_train2)
Y_test2_encoded = le.transform(Y_test2)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=70)
xgb_model.fit(X_train2, Y_train2_encoded)

# Predict on train data
train_preds_xgb_encoded = xgb_model.predict(X_train2)

# Predict on test data
test_preds_xgb_encoded = xgb_model.predict(X_test2)

print("XGBoost Classifier trained and predictions made.")

# To evaluate, we need to inverse transform the predictions back to original labels
train_preds_xgb = le.inverse_transform(train_preds_xgb_encoded)
test_preds_xgb = le.inverse_transform(test_preds_xgb_encoded)

# Calculate and print the accuracy score for the training predictions
print("Model accuracy on train is: ", accuracy_score(Y_train2, train_preds_xgb))

# Calculate and print the accuracy score for the test predictions
print("Model accuracy on test is: ", accuracy_score(Y_test2, test_preds_xgb))
print('-'*50)


# In[82]:


import pickle

# Save model
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

# Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

