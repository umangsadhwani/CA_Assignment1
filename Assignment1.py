#!/usr/bin/env python
# coding: utf-8


# importing various libraries which are used 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing as p
from scipy.stats import f
from sklearn.metrics import mean_squared_error, mean_absolute_error ,r2_score
import sys


# taking the csv file as input to create the model
if len(sys.argv) < 2:
    file=input("csv file_name : ")
else:
    file = sys.argv[1]


# creating a pandas dataframe using the values obtained in the csv
df=pd.read_csv(file)


# printing the head of the data frame to get the gist of the values
print("\n data frame head :- \n",df.head())


# creating a new col in our data frame which consists of the CPI per tuple 
df[['CPI']]=df[['cpu-cycles']].div(df['instructions'], axis=0)
print(df)


# dividing all the values by instruction so that we get values in each coloumn per instruction basis 
df[['l1d.replacement','icache_64b.iftag_miss','l2_rqsts.all_demand_miss','longest_lat_cache.miss','br_inst_retired.all_branches','frontend_retired.itlb_miss','itlb_misses.walk_completed','dtlb_load_misses.walk_completed','dtlb_store_misses.walk_completed','branch-misses']]=df[['l1d.replacement','icache_64b.iftag_miss','l2_rqsts.all_demand_miss','longest_lat_cache.miss','br_inst_retired.all_branches','frontend_retired.itlb_miss','itlb_misses.walk_completed','dtlb_load_misses.walk_completed','dtlb_store_misses.walk_completed','branch-misses']].div(df['instructions'], axis=0)
print(df)



# droping values such as time , instructions , cpu-cycles and br_inst_retired.all_branches
df= df.drop(['time'], axis=1)
df= df.drop(['instructions'], axis=1)
df= df.drop(['cpu-cycles'], axis=1)
df= df.drop(['br_inst_retired.all_branches'], axis=1)



# assigning y as the CPI and then droping it from the dataframe
y=df['CPI']
df= df.drop(['CPI'], axis=1)
print("y values :- \n",y)



# assigning x as the dataframe 
x=df
print("x values :- \n",x)



# creating a heatmap of the correlation matrix 
fig,axis = plt.subplots(figsize = (20,12))
sns.heatmap(x.corr(),annot=True)



# dividing the data set into test and train set in a 20:80 ration with a random state so that
# the model trains on a particular set of values on every execution 
X_train, X_test, y_train, y_test = train_test_split( 
    x, y, test_size=.20,random_state=55) 



# using MinMax Scaler to scale the data within the given range of 0 to 1 such that
#shape of the original distribution is same after transformation
mms = p.MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)



print("X_train :-\n",X_train)



# mean of all the columns of the training set 
df2 = X_train.mean(axis=0)
print(df2)



# creating a linear regression model using sklearn.linear_model 
model = LinearRegression(positive=True)
model.fit(X_train,y_train)


# finding the coefficients given by our model 
c=model.coef_
print("\nCoefficients :- \n",c)



# model intercept i.e. the " Base CPI " 
i=model.intercept_
print("\nBase CPI : ",i)



# making the predictions using our model on the test set 
predictions = model.predict(X_test) 



# Actual CPI
ACPI = y_test.mean()
print("\n Actual CPI : ",ACPI)


# Predicted CPI 
PCPI = predictions.mean()
print("\n Predicted CPI : ",PCPI)



# Finding out RMSE , R^2 , adjusted R^2 using our predictions and test set
RMSE = mean_squared_error(y_test, predictions)
print("\n RMSE : ",RMSE)

R2 = r2_score(y_test, predictions)
print("\n R^2 : ",r2_score(y_test, predictions))

adjusted_r2 = 1 - ( 1-model.score(X_test,y_test) ) * ( len(y_test) - 1 ) / ( len(y_test) - X_test.shape[1] - 1 )
print("\n adjusted R^2 : ",adjusted_r2)



# finding absolute error and accuracy on test set 
err = mean_absolute_error(y_test, predictions)
print ( "\n Test error is :" , err *100 , "% " )
print ( "\n Test Accuracy is :" , (1- err) *100 , "%" )




# F-statistic value which should be > 2.5 and p-value which should be < 0.05
F = (R2/(1-R2))*((X_test.shape[0]-1-X_test.shape[1])/X_test.shape[1])
print("\n F-statistic : ",F)

p = 1-f.cdf(F,X_test.shape[1],(X_test.shape[0]-1-X_test.shape[1]))
print("\n p-value : ",p)



#no of coefficients 
X_test.shape[1]



# no of tuples in the test set 
X_test.shape[0]



# finding the residual for our test set 
residuals = y_test - predictions
print("\n Residual :- \n ",residuals)



# residual graph 
data = {
    'predicted': [i for i in predictions], 
    'residuals': [i for i in residuals]
}

dfr = pd.DataFrame(data)
sns.scatterplot(data=dfr, x="predicted", y="residuals")


# ##### 
