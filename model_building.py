# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 08:50:31 2020

@author: Giats
"""

# preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV

# models
from sklearn.linear_model import LinearRegression, SGDRegressor, RidgeCV ,Ridge
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor 
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, VotingRegressor 
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb

# model tuning
from hyperopt import fmin, hp, tpe, space_eval

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

#read dataset
df = pd.read_csv("data_cleaned.csv")

#delete space from some colors
df['Color'] = df['Color'].apply(lambda x: x.replace(' ',''))

#Preparing to modeling
print("Too new: %d" % df.loc[df.Registration >= 2017].count()['Make_model'])
print("Too old: %d" % df.loc[df.Registration < 1950].count()['Make_model'])
print("Too cheap: %d" % df.loc[df.Price < 500].count()['Make_model'])
print("Too expensive: " , df.loc[df.Price > 150000].count()['Make_model'])
print("Too few km: " , df.loc[df.Mileage < 5000].count()['Make_model'])
print("Too many km: " , df.loc[df.Mileage > 200000].count()['Make_model'])
print("Too few PS: " , df.loc[df.Power < 10].count()['Make_model'])
print("Too many PS: " , df.loc[df.Power > 500].count()['Make_model'])


#drop unnecessary columns
df = df.drop(['Make_model', 'Classified_number','Number_plate','Previous_owners','Zip_code'], axis = 1)

df= pd.get_dummies(df) 

# Fitting Regression Model
X = df.drop("Price", axis = 1)
y = df["Price"]

#Split the data to X_train,  X_test, y_train, y_test
X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =42)



#Metrics Functions for Regression
acc_train_r2 = []
acc_test_r2 = []
acc_train_mae = []
acc_test_mae = []
acc_train_rmse = []
acc_test_rmse = []

def acc_r2(y_meas,y_pred):
    return r2_score(y_meas,y_pred)

def acc_mae(y_meas,y_pred):
    return mean_absolute_error(y_meas,y_pred)

def acc_rmse(y_meas,y_pred):
    return np.sqrt(mean_squared_error(y_meas,y_pred))

def acc_model(num,model,train,test):
    global acc_train_r2, acc_test_r2, acc_train_mae, acc_test_mae, acc_train_rmse, acc_test_rmse
     
    ytrain_pred = model.predict(train)  
    ytest_pred = model.predict(test)
    
    #train metrics
    acc_train_r2.insert(num, acc_r2(y_train,ytrain_pred))
    acc_train_mae.insert(num, acc_mae(y_train,ytrain_pred)) 
    acc_train_rmse.insert(num, acc_rmse(y_train,ytrain_pred))

    #test metrics
    acc_test_r2.insert(num, acc_r2(y_test,ytest_pred))
    acc_test_mae.insert(num, acc_mae(y_test,ytest_pred)) 
    acc_test_rmse.insert(num, acc_rmse(y_test,ytest_pred))

def acc_boosting_model(num,model,train,test,num_iteration=0):
    
    global acc_train_r2, acc_test_r2, acc_train_mae, acc_test_mae, acc_train_rmse, acc_test_rmse
    
    if num_iteration > 0:
        ytrain_pred = model.predict(train, num_iteration = num_iteration)  
        ytest_pred = model.predict(test, num_iteration = num_iteration)
    else:
        ytrain_pred = model.predict(train)  
        ytest_pred = model.predict(test)    
    
    #train metrics
    acc_train_r2.insert(num, acc_r2(y_train,ytrain_pred))
    acc_train_mae.insert(num, acc_mae(y_train,ytrain_pred)) 
    acc_train_rmse.insert(num, acc_rmse(y_train,ytrain_pred))

    #test metrics
    acc_test_r2.insert(num, acc_r2(y_test,ytest_pred))
    acc_test_mae.insert(num, acc_mae(y_test,ytest_pred)) 
    acc_test_rmse.insert(num, acc_rmse(y_test,ytest_pred))



# Tuning models and test for all features 
# Linear Regression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
acc_model(0,linreg,X_train,X_test)    
print("Done")

# Support Vector Machines
svr = SVR()
svr.fit(X_train, y_train)
acc_model(1,svr,X_train,X_test)
print("Done")

# Linear SVR
linear_svr = LinearSVR()
linear_svr.fit(X_train, y_train)
acc_model(2,linear_svr,X_train,X_test)
print("Done")

# MLPRegressor
mlp = MLPRegressor()
param_grid = {'hidden_layer_sizes': [i for i in range(2,20)],
              'activation': ['relu'],
              'solver': ['adam'],
              'learning_rate': ['constant'],
              'learning_rate_init': [0.01],
              'power_t': [0.5],
              'alpha': [0.0001],
              'max_iter': [1000],
              'early_stopping': [True],
              'warm_start': [False]}
mlp_GS = GridSearchCV(mlp, param_grid=param_grid, 
                   cv=2, verbose=True, pre_dispatch='2*n_jobs')
mlp_GS.fit(X_train, y_train)
acc_model(3,mlp_GS,X_train,X_test)
print("Done")

# Stochastic Gradient Descent
sgd = SGDRegressor()
sgd.fit(X_train, y_train)
acc_model(4,sgd,X_train,X_test)
print("Done")

# Decision Tree Regression
decision_tree = DecisionTreeRegressor()
decision_tree.fit(X_train, y_train)
acc_model(5,decision_tree,X_train,X_test)
print("Done")

# Random Forest
random_forest = GridSearchCV(estimator=RandomForestRegressor(), param_grid={'n_estimators': [100, 1000]}, cv=5)
random_forest.fit(X_train, y_train)
print(random_forest.best_params_)
acc_model(6,random_forest,X_train,X_test)
print("Done")

# XGB
xgb_clf = xgb.XGBRegressor() 
parameters = {'n_estimators': [60, 100, 120, 140], 
              'learning_rate': [0.01, 0.1],
              'max_depth': [5, 7],
              'reg_lambda': [0.5]}
xgb_reg = GridSearchCV(estimator=xgb_clf, param_grid=parameters, cv=3, n_jobs=-1).fit(X_train, y_train)
print("Best score: %0.3f" % xgb_reg.best_score_)
print("Best parameters set:", xgb_reg.best_params_)
acc_model(7,xgb_reg,X_train,X_test)    
print("Done")

#LGBM
#split training set to validation set
Xtrain, Xval, Ztrain, Zval = train_test_split(X, y, test_size=0.2, random_state=42)
train_set = lgb.Dataset(Xtrain, Ztrain, silent=False)
valid_set = lgb.Dataset(Xval, Zval, silent=False)

params = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'num_leaves': 31,
        'learning_rate': 0.01,
        'max_depth': -1,
        'subsample': 0.8,
        'bagging_fraction' : 1,
        'max_bin' : 5000 ,
        'bagging_freq': 20,
        'colsample_bytree': 0.6,
        'metric': 'rmse',
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight':1,
        'zero_as_missing': False,
        'seed':0,        
    }
modelL = lgb.train(params, train_set = train_set, num_boost_round=10000,
                   early_stopping_rounds=8000,verbose_eval=500, valid_sets=valid_set)
acc_boosting_model(8,modelL,X_train,X_test,modelL.best_iteration)
print("Done")

#Gradient Boosting
def hyperopt_gb_score(params):
    clf = GradientBoostingRegressor(**params)
    current_score = cross_val_score(clf, X_train, y_train, cv=2).mean()
    print(current_score, params)
    return current_score 
 
space_gb = {
            'n_estimators': hp.choice('n_estimators', range(100, 1000)),
            'max_depth': hp.choice('max_depth', np.arange(2, 10, dtype=int))            
        }
 
best = fmin(fn=hyperopt_gb_score, space=space_gb, algo=tpe.suggest, max_evals=10)
print('best:')
print(best)
params = space_eval(space_gb, best)
print(params)

# Gradient Boosting Regression
gradient_boosting = GradientBoostingRegressor(**params)
gradient_boosting.fit(X_train, y_train)
acc_model(9,gradient_boosting,X_train,X_test)
print("Done")

# Ridge Regressor
ridge = RidgeCV(cv=5)
ridge.fit(X_train, y_train)
acc_model(10,ridge,X_train,X_test)
print("Done")

# Bagging Regressor
bagging = BaggingRegressor()
bagging.fit(X_train, y_train)
acc_model(11,bagging,X_train,X_test)
print("Done")

# Extra Trees Regressor
etr = ExtraTreesRegressor()
etr.fit(X_train, y_train)
acc_model(12,etr,X_train,X_test)
print("Done")

# AdaBoost Regression
Ada_Boost = AdaBoostRegressor()
Ada_Boost.fit(X_train, y_train)
acc_model(13,Ada_Boost,X_train,X_test)
print("Done")

# Voting Regressor
Voting_Reg = VotingRegressor(estimators=[('lin', linreg), ('ridge', ridge), ('sgd', sgd)])
Voting_Reg.fit(X_train, y_train)
acc_model(14,Voting_Reg,X_train,X_test)
print("Done")




#Models comparison
models = pd.DataFrame({
    'Model': ['Linear Regression', 'Support Vector Machines', 'Linear SVR', 
              'MLPRegressor', 'Stochastic Gradient Decent', 
              'Decision Tree Regressor', 'Random Forest',  'XGB', 'LGBM',
              'GradientBoostingRegressor', 'RidgeRegressor', 'BaggingRegressor', 'ExtraTreesRegressor', 
              'AdaBoostRegressor', 'VotingRegressor'],
    
    'r2_train': acc_train_r2,
    'r2_test': acc_test_r2,
    'mae_train': acc_train_mae,
    'mae_test': acc_test_mae,
    'rmse_train': acc_train_rmse,
    'rmse_test': acc_test_rmse
})


# Plots
plt.figure(figsize=[34,25])
xx = models['Model']
plt.tick_params(labelsize=14)
plt.plot(xx, models['r2_train'], label = 'r2_train')
plt.plot(xx, models['r2_test'], label = 'r2_test')
plt.legend()
plt.title('R2-criterion for 15 popular models for train and test datasets')
plt.xlabel('Models')
plt.ylabel('R2-criterion, %')
plt.xticks(xx, rotation='vertical')
plt.savefig('Model_Images/R2.png')
plt.show()

plt.figure(figsize=[34,25])
xx = models['Model']
plt.tick_params(labelsize=14)
plt.plot(xx, models['mae_train'], label = 'd_train')
plt.plot(xx, models['mae_test'], label = 'd_test')
plt.legend()
plt.title('Relative errors for 15 popular models for train and test datasets')
plt.xlabel('Models')
plt.ylabel('Relative error, %')
plt.xticks(xx, rotation='vertical')
plt.savefig('Model_Images/MAE.png')
plt.show()

plt.figure(figsize=[34,25])
xx = models['Model']
plt.tick_params(labelsize=14)
plt.plot(xx, models['rmse_train'], label = 'rmse_train')
plt.plot(xx, models['rmse_test'], label = 'rmse_test')
plt.legend()
plt.title('RMSE for 15 popular models for train and test datasets')
plt.xlabel('Models')
plt.ylabel('RMSE, %')
plt.xticks(xx, rotation='vertical')
plt.savefig('Model_Images/RMSE.png')
plt.show()


#Tuning hyperparameters
ridge_params = {'alpha':[0.1, 0.001, 0.0001 , 1 ]}   

rf = RandomForestRegressor()
rs = RandomizedSearchCV(Ridge() , param_distributions = ridge_params, cv=10 , verbose=2)
rs.fit(X_train, y_train)
print(rs.best_params_)

ytrain_pred = rs.predict(X_train)  
ytest_pred = rs.predict(X_test)

print(r2_score(y_train,ytrain_pred))
print(mean_absolute_error(y_train,ytrain_pred))
print(np.sqrt(mean_squared_error(y_train,ytrain_pred)))

print(r2_score(y_test,ytest_pred))
print(mean_absolute_error(y_test,ytest_pred))
print(np.sqrt(mean_squared_error(y_test,ytest_pred)))

# 0.7624913276813889
# 2743.9413310174095
# 6331.355355865922

# 0.72551927636735
# 2987.6517517165125
# 6801.787554074726

#create model file
pickl = {'model': rs.best_estimator_}
pickle.dump( pickl, open( 'model_file' + ".p", "wb" ))

