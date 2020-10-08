#Remove all data from environment
rm(list=ls())

#Setting Working Directory
setwd("C:/Users/Admin/Documents/R/R Scripts/cab")

#Loading Libraries
x = c('datetime','gmt','caret','DMwR','rpart','randomForest','lightgbm','dplyr', "corrgram")
install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)

#Importing the dataset
train_data=read.csv('train_cab.csv',header = T,na.strings = c(" ","",NA))
test_data=read.csv('test.csv',header = T,na.strings = c(" ","",NA))

#checking the dimension of the dataset
dim(train_data)
dim(test_data)

#checking the structure of the dataset
str(train_data)
str(test_data)

# Changing the data types of variables
train_data$fare_amount = as.numeric(as.character(train_data$fare_amount))
train_data$pickup_datetime = strptime(as.character(train_data$pickup_datetime),"%Y-%m-%d %H:%M:%S")

#Defining Functions for Accuracy Metrics, Feature creation & Modelling
# 1.Creating Distance feature in Dataset based on lat long positions
distance=function(dataset){
  deg_to_rad = function(deg){
    (deg * pi) / 180
  }
  haversine = function(long1,lat1,long2,lat2){
    #long1rad = deg_to_rad(long1)
    phi1 = deg_to_rad(lat1)
    #long2rad = deg_to_rad(long2)
    phi2 = deg_to_rad(lat2)
    delphi = deg_to_rad(lat2 - lat1)
    dellamda = deg_to_rad(long2 - long1)
    
    a = sin(delphi/2) * sin(delphi/2) + cos(phi1) * cos(phi2) * 
      sin(dellamda/2) * sin(dellamda/2)
    
    c = 2 * atan2(sqrt(a),sqrt(1-a))
    R = 3959
    R * c #calculating in miles
  }
  
  dataset[,'distance']=NA
  dataset$distance = round(haversine(dataset$pickup_longitude,dataset$pickup_latitude,dataset$dropoff_longitude,dataset$dropoff_latitude),2)
  
  dataset=dataset[,-which(names(dataset) %in% c('pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'))]
  
  return(dataset)
}

# 2.Creating Time Based features in Dataset based on Pickup_datetime feature
time_features=function(dataset){
  dataset$year = as.numeric(format(dataset$pickup_datetime,"%Y"))
  dataset$month = as.numeric(format(dataset$pickup_datetime,"%m"))
  dataset$week_day = as.numeric(format(dataset$pickup_datetime,"%w"))# Sunday = 0
  dataset$hour = as.numeric(format(dataset$pickup_datetime,"%H"))
  
  dataset=dataset[,-which(names(dataset) %in% c('pickup_datetime'))]
  
  return(dataset)
}

# 3.Creating Cab_type feature in Dataset based on passanger_count feature [small cab=0, bigger cab=1]
cab_type=function(dataset){
  dataset[,'cab_type']=0
  dataset$cab_type[which(dataset$passenger_count > 3)] = 1
  
  dataset=dataset[,-which(names(dataset) %in% c('passenger_count'))]
  
  return(dataset)
}

# 4.Creating scaling function
scaling=function(dataset){
  data = subset(dataset,select=-fare_amount)
  cnames = colnames(data)
  
  for(i in cnames){
    
    dataset[,i] = (dataset[,i] - mean(dataset[,i])) / sd(dataset[,i])
  }
  
  return(dataset)
}

# 5.Creating accuracy metrics function
acc_metrics=function(actual_values,predicted_values){
  accuracy=regr.eval(actual_values,predicted_values)
  
  return(accuracy)
}

# 6.Creating modelling function to create all the models
modelling=function(train,test){
  # A. Linear regression
  LM_model = lm(fare_amount ~.,data=train)
  predictions_LM = predict(LM_model,test[,2:7])
  accuracy=acc_metrics(test$fare_amount,predictions_LM)
  print("Linear Regression Results: ")
  print(accuracy)
  
  # B. Decision Tree
  DT_model = rpart(fare_amount ~ ., data = train, method = "anova")
  predictions_DT = predict(DT_model, test[,2:7])
  accuracy=acc_metrics(test$fare_amount,predictions_DT)
  print("Decision Tree Results: ")
  print(accuracy)
  
  # C. Random Forest
  RF_model = randomForest(fare_amount ~ ., data = train)
  predictions_RF = predict(RF_model, test[,2:7])
  accuracy=acc_metrics(test$fare_amount,predictions_RF)
  print("Random Forest Results: ")
  print(accuracy)
  
  # D. LightGBM Regressor
  lgb_train = lgb.Dataset(data=as.matrix(train[,2:7]), label=train[,1])
  params=list(
    objective= 'regression',
    boosting_type='gbdt',
    n_estimators=10000,
    learning_rate=0.01,
    subsample=0.7,
    colsample_bytree=0.8,
    max_depth=5,
    num_leaves=5,
    min_child_weight=10,
    force_col_wise=T
  )
  LGB_model = lightgbm(params = params ,metric = 'rmse', lgb_train, verbose=0)
  predictions_LGB = predict(LGB_model, dat = as.matrix(test[,2:7]))
  accuracy=acc_metrics(test$fare_amount,predictions_LGB)
  print("LightGBM Regressor Results: ")
  print(accuracy)
}


#Data Pre Processing
# 1. Missing Value Analysis
sum(is.na(train_data))
train_data=na.omit(train_data)
rownames(train_data) <- NULL
dim(train_data)
sum(is.na(train_data))
train_data[rowSums(is.na(train_data)) > 0, ]
train_data=train_data[-c(1279),]
dim(train_data)
sum(is.na(train_data))
sum(is.na(test_data))

# 2. Checking training data for impurities
summary(train_data)

##### Observations for train data:
##### 1. This data is for United States as lat 40.xxx & long -73.xxx locate to New York United States. So, 
#####     latitude range  = 40.xxx to 42.xxx
#####     longitude range = -72.xxx to -74.xxx
##### 2. Impure data in pickup & dropoff longitude as it contains values outside their range i.e. 40.xxx.
##### 3. Impure data in pickup & dropoff latitude as it contains values outside their range i.e. -74.xxx.
##### 4. Fare Amount contains some very high values as Standard deviation of $ 430 is quite high. 

# 3. Treating training data for impurities
#A. Treating Passanger Count
train_data$passenger_count=round(train_data$passenger_count)
train_data$passenger_count[which(train_data$passenger_count <1 )]=NA
train_data$passenger_count[which(train_data$passenger_count >6 )]=NA
train_data=na.omit(train_data)
dim(train_data)

#B. Treating Fare Amount
train_data$fare_amount[which(train_data$fare_amount <2.5 )]=NA
sum(is.na(train_data))
train_data=na.omit(train_data)
dim(train_data)

#C. Treating Pickup & dropoff (Lat Long) Values
train_data$pickup_longitude[which(train_data$pickup_longitude < -74 )]=NA
train_data$pickup_longitude[which(train_data$pickup_longitude > -72 )]=NA
train_data$dropoff_longitude[which(train_data$dropoff_longitude < -74 )]=NA
train_data$dropoff_longitude[which(train_data$dropoff_longitude > -72 )]=NA
train_data$pickup_latitude[which(train_data$pickup_latitude < 40 )]=NA
train_data$pickup_latitude[which(train_data$pickup_latitude > 42 )]=NA
train_data$dropoff_latitude[which(train_data$dropoff_latitude < 40 )]=NA
train_data$dropoff_latitude[which(train_data$dropoff_latitude > 42 )]=NA
sum(is.na(train_data))
train_data=na.omit(train_data)
dim(train_data)

# 4. Outlier Analysis on fare_amount
val = train_data$fare_amount[train_data$fare_amount %in% boxplot.stats(train_data$fare_amount)$out]
train_data = train_data[which(!train_data$fare_amount %in% val),]
dim(train_data)


#Exploratory Data Analysis & Visualizations

# 1. Feature Engineering
# A. Creating distance feature
train_data=distance(train_data)
# Observations from above: 
#1. Train & Test data almost have same distance distribution. Most Values lies between 0 - 10.
#2. Distance contains zero values.

train_data$distance[which(train_data$distance == 0 )]=NA
train_data$distance[which(train_data$distance > 10 )]=NA
sum(is.na(train_data))
train_data=na.omit(train_data)
dim(train_data)
head(train_data)

# B. Creating Timestamp based feature
train_data=time_features(train_data)
head(train_data)

# C. Creating Passanger Count based feature
train_data=cab_type(train_data)
head(train_data)

# 2.	Visualizing the effects of feature's on target variable
# A. Hour effect on no of rides
x=train_data %>% group_by(hour) %>% summarise_at(vars(fare_amount),funs(sum(.,na.rm=TRUE)))
barplot(names.arg =x$hour,x$fare_amount)
#Observations from above: 
# 1.	6.00 PM - 11.00 PM hours shows higher number of cab rides.
# 2.	1.00 AM - 6.00 AM hours shows lower number of cab rides.


# B. Hour effect on Fare Amount
x=train_data %>% group_by(hour) %>% summarise_at(vars(fare_amount),funs(mean(.,na.rm=TRUE)))
barplot(names.arg =x$hour,x$fare_amount)
#Observations from above: 
#  Late night hours 10.00 PM - 4.00 AM shows higher fare rates.


# C. Week Day effect on Fare Amount
x=train_data %>% group_by(week_day) %>% summarise_at(vars(fare_amount),funs(mean(.,na.rm=TRUE)))
barplot(names.arg =x$week_day,x$fare_amount)
#Observations from above: 
# Weekday shows no major impact on fare_amount.


# D. Month effect on Fare Amount
x=train_data %>% group_by(month) %>% summarise_at(vars(fare_amount),funs(median(.,na.rm=TRUE)))
barplot(names.arg =x$month,x$fare_amount)
#Observations from above: 
# Month shows no major impact on fare_amount.

# E. Year effect on Fare Amount
x=train_data %>% group_by(year) %>% summarise_at(vars(fare_amount),funs(mean(.,na.rm=TRUE)))
barplot(names.arg =x$year,x$fare_amount)
#Observations from above: 
# Year shows a slight increasing effect on Fare Amount.

# F. Distance effect on Fare Amount
x=train_data %>% group_by(distance) %>% summarise_at(vars(fare_amount),funs(mean(.,na.rm=TRUE)))
plot(names.arg =x$distance,x$fare_amount)
#Observations from above: 
# Distance shows a highly linear increasing effect on Fare Amount.


# G. Cab Type effect on Number of Rides
x=train_data %>% group_by(cab_type) %>% summarise_at(vars(fare_amount),funs(sum(.,na.rm=TRUE)))
barplot(names.arg =x$cab_type,x$fare_amount)
#Observations from above: 
# Most of the cabs rides are of small size cab.


# 3. Feature Selection
# A. Features Multi Collinearity Test
corrgram(train_data, order = F,upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")
#Observations from above:
#1.	The correlation values between feature variables is less than 0.15. Thus, multicollinearity doesn't exists between the feature variables.
#2.	The Distance column have a high coorelation value with the target variable.


# B. Categorical Features - Target Chi Square Test
for (i in colnames(train_data[,3:7]))
{
  x=c(chisq.test(table(train_data$fare_amount,train_data[,i])))
  if(x[3]>0.5){
    print(i)
    print('Fail to reject NULL Hypothesis.')
  }
  
}
#Observations from above:
# We Fail to reject null hypothesis for week_day & cab_type columns. 
# But we are not going to drop any columns as cab_type is the only column to provide a relationship with passenger count 
# & week_day have a slight impact on number of rides.


# 4. Feature Scaling
train_data=scaling(train_data)
head(train_data)

#Train Test Split
set.seed(1234)
train.index = createDataPartition(train_data$fare_amount, p = .80, list = FALSE)
train = train_data[ train.index,]
test  = train_data[-train.index,]
rm(train.index)

#Modelling
modelling(train,test)

# "Linear Regression Results: "
#mae       mse      rmse      mape 
#1.5487793 4.9845782 2.2326169 0.1963195 

# "Decision Tree Results: "
#mae       mse      rmse      mape 
#1.6621642 5.4272316 2.3296419 0.2081523 

# "Random Forest Results: "
#mae       mse      rmse      mape 
#1.4680482 4.3627608 2.0887223 0.1861896 

# "LightGBM Regressor Results: "
#mae       mse      rmse      mape 
#1.4294535 4.2745900 2.0675082 0.1777313 

#Observations from above:
#From above the best performing model is LightGBM Regressor. RMSE = 2.067 , MAPE = 17.77 %

##########################################Selecting Best Model######################################


lgb_train = lgb.Dataset(data=as.matrix(train[,2:7]), label=train[,1])
params=list(
  objective= 'regression',
  boosting_type='gbdt',
  n_estimators=10000,
  learning_rate=0.01,
  subsample=0.7,
  colsample_bytree=0.8,
  max_depth=5,
  num_leaves=5,
  min_child_weight=10,
  force_col_wise=T
)
LGB_model = lightgbm(params = params ,metric = 'rmse', lgb_train, verbose=0)
predictions_LGB = predict(LGB_model, dat = as.matrix(test[,2:7]))
accuracy=acc_metrics(test$fare_amount,predictions_LGB)
print("LightGBM Regressor Results: ")
print(accuracy)
#      mae       mse      rmse      mape 
#1.4294535 4.2745900 2.0675082 0.1777313 



#Saving the Final Model
# save the model to disk
saveRDS(LGB_model, "./final_model.rds")


#Conclusion
#.	Distance shows the max correlation with Fare amount. Thus it is the most important feature in predicting fare amount.
#.	Fare amount shows a slight linear relation with the year, i.e. the fare rates are increasing year by year.
#.	Months & Weekdays Doesn't shows any major impact on fare amount. Fare rates is mostly uniform.
#.	Late Night & early morning hours shows lower number of rides but higher fare rates, while rest of the day the number of rides are higher but with lower fare rates.
#.	Passenger count doesn't have any major impact on fare rates.
#.	On the basis of RMSE & MAPE the performing model is LightGBM Regressor. 
#.	Best model RMSE = 1.89, MAPE = 16.86 %.

