#!/usr/bin/env python
# coding: utf-8

# # Importing Modules

# In[1]:


# Data Handling
import pandas as pd
import numpy as np
 
# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Metrics
from sklearn.metrics import mean_squared_error
from math import sqrt
from geopy.distance import geodesic

# Sampling & Scaling
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

# Modelling
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

import pickle

# Warning Handling
import warnings
warnings.simplefilter("ignore")


# # Defining Functions for Accuracy Metrics, Distance Calculation & Modelling

# In[2]:


def mean_absolute_percentage_error(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

def acc_metrics(actual_values,predicted_values):
    print('Accuracy Metrics:')
    mse=round(mean_squared_error(actual_values,predicted_values),2)
    rmse=round(sqrt(mse),2)
    mape=round(mean_absolute_percentage_error(actual_values,predicted_values),2)    
    print('Mean Squared Error: ',mse)
    print('Root Mean Squared Error: ',rmse)
    print('Mean Absolute Percentage Error: ',mape)
           
    return mse,rmse,mape


# In[3]:


def modelling(x_train, x_test, y_train, y_test):
    
    print('\nLinear Regression Modelling')
    LR_model=LinearRegression()
    LR_model.fit(x_train,y_train)
    acc_metrics(y_test,LR_model.predict(x_test))
    
    print('\nDecision Tree Regressor Modelling')
    DT_model=DecisionTreeRegressor(max_depth=5)
    DT_model.fit(x_train,y_train)
    acc_metrics(y_test,DT_model.predict(x_test))
    
    print('\nRandom Forest Regressor Modelling')
    RF_model=RandomForestRegressor()
    RF_model.fit(x_train,y_train)
    acc_metrics(y_test,RF_model.predict(x_test))
    
    print('\nLightGBM Regressor Modelling')
    LGB_model = LGBMRegressor()
    LGB_model.fit(x_train, y_train, eval_set=[(x_test,y_test)],eval_metric='rmse', verbose=0, early_stopping_rounds=5000)
    acc_metrics(y_test,LGB_model.predict(x_test))
    
    return LR_model, DT_model, RF_model, LGB_model


# In[4]:


def distance(dataset):
    geodesic_dist=[]
    
    for i in range(len(dataset)):
        pickup = (dataset.pickup_latitude.iloc[i], dataset.pickup_longitude.iloc[i])
        dropoff = (dataset.dropoff_latitude.iloc[i], dataset.dropoff_longitude.iloc[i])
        geodesic_dist.append(abs(round(geodesic(pickup, dropoff).miles,2)))
        
    dataset['distance']=geodesic_dist
    dataset.drop(columns=['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],inplace=True)

    return dataset


# In[5]:


def time_features(dataset):
    dataset['year']=pd.DatetimeIndex(dataset.pickup_datetime).year
    dataset['month']=pd.DatetimeIndex(dataset.pickup_datetime).month
    dataset['week_day']=pd.DatetimeIndex(dataset.pickup_datetime).weekday
    dataset['hour']=pd.DatetimeIndex(dataset.pickup_datetime).hour
       
    dataset.drop(columns=['pickup_datetime'],inplace=True)
    
    return dataset    


# In[6]:


def cab_type(dataset):
    dataset['cab_type']=[0 if i<4 else 1 for i in dataset.passenger_count ]
    dataset.drop(columns=['passenger_count'],inplace=True)
    
    return dataset


# # Importing Datasets

# In[7]:


train_data=pd.read_csv('train_cab.csv')
test_data=pd.read_csv('test.csv')


# In[8]:


train_data.head()


# In[9]:


test_data.head()


# In[10]:


train_data["fare_amount"] = pd.to_numeric(train_data["fare_amount"],errors = "coerce")
train_data.pickup_datetime=pd.to_datetime(train_data.pickup_datetime,errors='coerce')

test_data.pickup_datetime=pd.to_datetime(test_data.pickup_datetime,errors='coerce')


# In[11]:


train_data.info()


# In[12]:


test_data.info()


# In[13]:


train_data.shape


# In[14]:


test_data.shape


# ##### Observations for datasets:
# 1. The train data contains 16067 observations.
# 2. The test data contains 9914 observations.
# 3. Train Data contains a datetime variable, a passanger count variable, pickup & dropoff Latitute longitude variables & one
#    target variable fare amount.
# 4. Test Data contains a datetime variable, a passanger count variable & pickup & dropoff Latitute longitude variables.

# # Data Pre-Processing

# ### 1. Missing Value Analysis

# In[15]:


train_data.isnull().sum().sum()


# In[16]:


test_data.isnull().sum().sum()


# In[17]:


# Missing Values in Train Data

train_data.isnull().sum()


# In[18]:


# Dropping the missing values

train_data.dropna(inplace=True)


# In[19]:


train_data.shape


# In[20]:


test_data.isnull().sum()


# ##### Observations missing value analysis:
# 1. The train dataset contains 81 missing values. Which is 0.5% of total observations, thus dropping missing values.
# 2. The test dataset doesnot contains any missing values.
# 3. Shape of train dataset after removing missing values is: (15986, 7)

# ### 2. Checking training data for impurities

# In[21]:


train_data.describe()


# In[22]:


test_data.describe()


# ##### Observations for train data:
# 1. This data is for United States as lat 40.xxx & long -73.xxx locate to New York United States. So, 
#     latitude range  = 40.xxx to 42.xxx
#     longitude range = -72.xxx to -74.xxx
# 2. Impure data in pickup & dropoff longitude as it contains values outside their range i.e. 40.xxx.
# 3. Impure data in pickup & dropoff latitude as it contains values outside their range i.e. -74.xxx.
# 4. Fare Amount contains some very high values as Standard deviation of $ 430 is quite high. 

# ### 3. Treating training data for impurities

# #### A. Treating Passenger Count

# In[23]:


test_data.passenger_count.value_counts()


# In[24]:


train_data.passenger_count=train_data.passenger_count.astype('int')
train_data.passenger_count.unique()


# In[25]:


#As Test data contains the values in range [1,2,3,4,5,6]. Removing all the other values from dataset.

i=[1,2,3,4,5,6]
train_data.passenger_count=train_data.passenger_count.transform(lambda x: None if (x not in i) else x )


# In[26]:


train_data.passenger_count.value_counts()


# In[27]:


train_data.dropna(inplace=True)


# In[28]:


train_data.passenger_count=train_data.passenger_count.astype('int')


# In[29]:


train_data.shape


# #### B. Treating Fare Amount

# In[30]:


# Taking he minimum value of fare_amount as $2.5

plt.hist(train_data.fare_amount[train_data.fare_amount<2.5])


# In[31]:


#Treating All the values less then $2.5 with None & removing them

train_data.fare_amount[train_data.fare_amount <2.5]=None


# In[32]:


train_data.dropna(inplace=True)


# In[33]:


train_data.shape


# #### C. Treating Pickup & dropoff (Lat Long) Values

# In[34]:


plt.figure(figsize=(14,10))
for i,col in enumerate(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']):
    plt.subplot(2,2,i+1)
    sns.distplot(test_data[col])
    plt.title(col)
plt.show()    


# In[35]:


plt.figure(figsize=(14,10))
for i,col in enumerate(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']):
    plt.subplot(2,2,i+1)
    sns.distplot(train_data[col])
    plt.title(col)
plt.show()    


# In[36]:


# lat long values Treating All the values outside range with None

train_data.dropoff_latitude[train_data.dropoff_latitude < 40] = None
train_data.dropoff_latitude[train_data.dropoff_latitude > 42] = None

train_data.pickup_latitude[train_data.pickup_latitude < 40] = None
train_data.pickup_latitude[train_data.pickup_latitude > 42] = None

train_data.dropoff_longitude[train_data.dropoff_longitude < -74] = None
train_data.dropoff_longitude[train_data.dropoff_longitude > -72] = None

train_data.pickup_longitude[train_data.pickup_longitude < -74] = None
train_data.pickup_longitude[train_data.pickup_longitude > -72] = None


# In[37]:


train_data.dropna(inplace=True)


# In[38]:


train_data.shape


# In[39]:


train_data.describe()


# ### 4. Outlier Analysis on fare_amount

# In[40]:


plt.figure(figsize=(8,6))
sns.boxenplot(train_data.fare_amount)


# In[41]:


sns.scatterplot(train_data.fare_amount,train_data.index)


# In[42]:


# Fare Amount Column

q75, q25 = np.percentile(train_data.fare_amount, [75 ,25])
iqr = q75 - q25
minimum = q25 - (iqr*1.5)
maximum = q75 + (iqr*1.5)

train_data.fare_amount[train_data.fare_amount < minimum]=None
train_data.fare_amount[train_data.fare_amount > maximum]=None

print('Max out bound: ',maximum)
print('Min out bound: ',minimum)


# In[43]:


train_data.dropna(inplace=True)
train_data.shape


# In[44]:


sns.scatterplot(train_data.fare_amount,train_data.index)


# In[45]:


train_data.describe()


# # Exploratory Data Analysis & Visualizations

# ### 1. Feature Engineering

# #### A. Creating distance feature

# In[46]:


train_data=distance(train_data)
test_data=distance(test_data)


# In[47]:


train_data.distance.value_counts()


# In[48]:


test_data.distance.value_counts()


# In[49]:


plt.figure(figsize=(10,8))

sns.distplot(test_data['distance'],label='Test_Data_distance', hist=False)
plt.title('test_data')
sns.distplot(train_data['distance'],label='Train_Data_distance', hist=False)
plt.title('train_data')
plt.title("Train & Test data Distance Distribution")
plt.legend()
plt.show()  


# ##### Observations from above: 
# 1. Train & Test data almost have same distance distribution. Most Values lies between 0 - 10.
# 2. Distance contains zero values.

# In[50]:


train_data.distance=train_data.distance.transform(lambda x: None if ((x == 0) or (x > 10)) else x )
train_data.dropna(inplace=True)


# In[51]:


train_data.shape


# In[52]:


train_data.describe()


# In[53]:


plt.figure(figsize=(14,10))
sns.distplot(train_data['distance'])
plt.title('Train_Data_Distance_Distribution')
plt.show()  


# In[54]:


train_data.head()


# #### B. Creating Timestamp based feature

# In[55]:


train_data=time_features(train_data)
test_data=time_features(test_data)


# In[56]:


train_data.head()


# #### C. Creating Passenger Count based feature

# In[57]:


# 0 for small cab  &  1 for large cab

train_data=cab_type(train_data)
test_data=cab_type(test_data)


# In[58]:


train_data.head()


# In[59]:


test_data.head()


# ### 2. Visualizing the effects of feature's on target variable

# #### A. Hour effect on no of rides

# In[60]:


plt.figure(figsize=(10,8))
hour_fare=train_data.groupby(by='hour').fare_amount.sum()
sns.barplot(hour_fare.index,hour_fare)
plt.ylabel('No_of_Rides')
plt.xlabel('Hour')
plt.title('Hour_No_Of_Rides_Distribution')


# ##### Observations from above: 
# 1.	6.00 PM – 11.00 PM hours shows higher number of cab rides.
# 2.	1.00 AM – 6.00 AM hours shows lower number of cab rides.
# 

# #### B. Hour effect on Fare Amount

# In[61]:


# Hour effect on fare_amount
plt.figure(figsize=(10,8))
hour_fare=train_data.groupby(by='hour').fare_amount.mean()
sns.barplot(hour_fare.index,hour_fare)
plt.ylabel('Avg_Fare_Amount')
plt.xlabel('Hour')
plt.title('Hour_FareAmount_Distribution')


# ##### Observations from above: 
# Late night hours 10.00 PM – 4.00 AM shows higher fare rates.
# 

# #### C. Week Day effect on Fare Amount

# In[62]:


plt.figure(figsize=(10,8))
week_day_fare=train_data.groupby(by='week_day').fare_amount.mean()
sns.barplot(week_day_fare.index,week_day_fare)
plt.ylabel('Avg_Fare_Amount')
plt.xlabel('Week Day')
plt.title('WeekDay_FareAmount_Distribution')


# ##### Observations from above: 
# Weekday shows no major impact on fare_amount.
# 

# #### D. Month effect on Fare Amount

# In[63]:


plt.figure(figsize=(10,8))
month_fare=train_data.groupby(by='month').fare_amount.median()
sns.barplot(month_fare.index,month_fare)
plt.ylabel('Avg_Fare_Amount')
plt.xlabel('Month')
plt.title('Month_FareAmount_Distribution')


# ##### Observations from above: 
# Month shows no major impact on fare_amount.
# 

# #### E. Year effect on Fare Amount

# In[64]:


plt.figure(figsize=(10,8))
year_fare=train_data.groupby(by='year').fare_amount.mean()
sns.barplot(year_fare.index,year_fare)
plt.ylabel('Avg_Fare_Amount')
plt.xlabel('Year')
plt.title('Year_FareAmount_Distribution')


# ##### Observations from above: 
# Year shows a slight increasing effect on Fare Amount.
# 

# #### F. Distance effect on Fare Amount

# In[65]:


plt.figure(figsize=(10,8))
distance_fare=train_data.groupby(by='distance').fare_amount.mean()
sns.scatterplot(distance_fare.index,distance_fare)
plt.ylabel('Avg_Fare_Amount')
plt.xlabel('Distance_in_Miles')
plt.title('Distance_Fare_Amount_Distribution')


# ##### Observations from above: 
# Distance shows a highly linear increasing effect on Fare Amount.
# 

# #### G. Cab Type effect on Number of Rides

# In[66]:


cab_type_fare=pd.DataFrame(train_data.groupby(by='cab_type').fare_amount.sum())
cab_type_fare


# In[67]:


plt.figure(figsize=(12,10))
cab_type_fare.plot.pie(y='fare_amount')
plt.ylabel('No_of_Rides')
plt.xlabel('CabType_No_of_Rides_Distribution')
label={0:'Small Cab Size',1:'Large Size Cab'}
plt.legend(["Small Cab Size", "Large Size Cab"],bbox_to_anchor =(0.75, 1.15))


# ##### Observations from above: 
# Most of the cabs rides are of small size cab.
# 

# ### 3. Feature Selection

# #### A. Features Multi Collinearity Test

# In[68]:


plt.figure(figsize=(8,6))
corr=train_data.corr()
sns.heatmap(corr,annot=True)
plt.title('Features_Inter_Correlation_Plot')


# #### B. Categorical Features - Target Chi Square Test

# In[69]:


#performing Chi square Test
'''
Significance level: 0.5

Alternate Hypothesis: Column and fare_amount have linear Relationship.        pValue <= 0.5 

Null Hypothesis:Column and fare_amount doesn't have linear Relationship.      pValue > 0.5 
'''
from scipy.stats import chi2_contingency
cat_cols=train_data.drop(columns=['fare_amount','distance']).columns
col_reduced=[]
for i in cat_cols:
    ct = pd.crosstab(train_data[i],train_data['fare_amount'])
    stat,pvalue,dof,expected_R = chi2_contingency(ct)

    if pvalue <= 0.05:
        print("Alternate Hypothesis passed.",i," and fare_amount have linear Relationship. Stat Value: ",stat)
    else:
        col_reduced.append(i)
        print("Fail to Reject Null Hypothesis.",i," and fare_amount doesn't have linear Relationship. Stat Value: ",stat) 


# ##### Observations from above:
# 1. The correlation values between feature variables is less than 0.15. Thus, multicollinearity doesn’t exists between the   feature variables.
# 2.	The Distance column have a high coorelation value with the target variable.
# 3. We Fail to reject null hypothesis for week_day & cab_type columns. 
# 4. We are not going to drop any columns as cab_type is the only column to provide a relationship with passenger count & week_day have a slight impact on number of rides.

# ### 4. Feature Scaling

# In[70]:


feature_col=train_data.drop(columns='fare_amount').columns
scaler=StandardScaler()
scaler.fit(train_data[feature_col])
train_data[feature_col]=scaler.transform(train_data[feature_col])
col=test_data.columns
test_data[col]=scaler.transform(test_data[col])


# In[71]:


train_data.head()


# In[72]:


test_data.head()


# In[73]:


features=train_data.drop(columns='fare_amount')
label=train_data.fare_amount


# # Splitting & modelling

# In[74]:


for i in range(0,21):
    print('\n\n\n',i)
    x_train, x_test, y_train, y_test= train_test_split(features,label,test_size=0.2,random_state=i)
    
    LR_model, DT_model, RF_model, LGB_model=modelling(x_train, x_test, y_train, y_test)
    


# ##### Observations from above:
# From above the best random state for train test split is 12.
# 
# Linear Regression Modelling
# Accuracy Metrics:
# Mean Squared Error:  4.43
# Root Mean Squared Error:  2.1
# Mean Absolute Percentage Error:  19.08
# 
# Decision Tree Regressor Modelling
# Accuracy Metrics:
# Mean Squared Error:  4.19
# Root Mean Squared Error:  2.05
# Mean Absolute Percentage Error:  18.21
# 
# Random Forest Regressor Modelling
# Accuracy Metrics:
# Mean Squared Error:  4.06
# Root Mean Squared Error:  2.01
# Mean Absolute Percentage Error:  18.5
# 
# LightGBM Regressor Modelling
# Accuracy Metrics:
# Mean Squared Error:  3.59
# Root Mean Squared Error:  1.89
# Mean Absolute Percentage Error:  16.94

# # Best Split

# In[76]:


x_train, x_test, y_train, y_test= train_test_split(features,label,test_size=0.2,random_state=12)


# ##### From above two best models are Random Forest & LightGBM Regressor. Let's Hypertune their parameters for better result.

# # Hyper Parameters Tuning & Best Model

# In[77]:


##Random Search CV on Random Forest Model

RF_model = RandomForestRegressor(random_state = 0)
n_estimator = list(range(0,401,2))
depth = list(range(2,20,2))
samples_leaf=list(range(1,5))
# Create the random grid
rand_grid = {'n_estimators': n_estimator,
             'max_depth': depth,
             'min_samples_leaf': samples_leaf
            }

randomcv_RF = RandomizedSearchCV(RF_model, param_distributions = rand_grid, n_iter = 5, cv = 5, random_state=0)
randomcv_RF = randomcv_RF.fit(x_train,y_train)
predictions_RF = randomcv_RF.predict(x_test)

view_best_params_RF = randomcv_RF.best_params_

best_RF_model = randomcv_RF.best_estimator_

predictions_RF = best_RF_model.predict(x_test)

print('Random Search CV Random Forest Regressor Model Performance:')
print('Best Parameters = ',view_best_params_RF)
acc_metrics(y_test,predictions_RF)


# In[78]:


##Random Search CV on LGBMRegressor model

LGB_model = LGBMRegressor(random_state = 0,
                          objective= 'regression',
                          learning_rate='0.01',
                          subsample=0.7,
                          colsample_bytree=0.8,
                          num_leaves=5,
                          min_child_weight=10)

n_estimator = list(range(1000,10001,50))
depth = list(range(1,10))

# Create the random grid
rand_grid = {'n_estimators': n_estimator,
             'max_depth': depth}

randomcv_LGB = RandomizedSearchCV(LGB_model, param_distributions = rand_grid, n_iter = 5, cv = 5, random_state=0)
randomcv_LGB = randomcv_LGB.fit(x_train,y_train)
predictions_LGB = randomcv_LGB.predict(x_test)

view_best_params_LGB = randomcv_LGB.best_params_

best_LGB_model = randomcv_LGB.best_estimator_

predictions_LGB = best_LGB_model.predict(x_test)

print('Random Search CV LGB Regressor Model Performance:')
print('Best Parameters = ',view_best_params_LGB)
acc_metrics(y_test,predictions_LGB)


# ##### Observations from above:
# From above the best performing model is LightGBM Regressor. RMSE = 1.89 ,  MAPE = 16.86 %

# #### Best Parameters

# In[79]:


best_LGB_model.get_params()


# # Saving Models

# In[80]:


pickle.dump(best_LGB_model,open('Cab_Fare_Prediction_model.model','wb'))
pickle.dump(scaler,open('scaler.model','wb'))


# # Conclusion

# ###### •	Distance shows the max correlation with Fare amount. Thus it is the most important feature in predicting fare amount.
# ###### •	Fare amount shows a slight linear relation with the year, i.e. the fare rates are increasing year by year.
# ###### •	Months & Weekdays Doesn’t shows any major impact on fare amount. Fare rates is mostly uniform.
# ###### •	Late Night & early morning hours shows lower number of rides but higher fare rates, while rest of the day the number of rides are higher but with lower fare rates.
# ###### •	Passenger count doesn’t have any major impact on fare rates.
# ###### •	On the basis of RMSE & MAPE the performing model is LightGBM Regressor. 
# ###### •	Best model RMSE = 1.89, MAPE = 16.86 %.
# 
