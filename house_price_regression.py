###################################
# Julian Cabezas Pena
# Deep Learning Fundamentals
# University of Adelaide
# Assingment 1
# Multi layer Perceptron implementqation and testiong in the PIMA diabetes data
####################################

# Import libraries
import numpy as np
import pandas as pd
import os

from sklearn.metrics import mean_squared_log_error, mean_squared_error

from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.model_selection import KFold

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor

import xgboost as xgb
from xgboost.sklearn import XGBRegressor

from catboost import CatBoostRegressor

import lightgbm as lgb

from sklearn.model_selection import GridSearchCV

from scipy.stats import skew

def rmsle(y,pred):
    return np.sqrt(mean_squared_log_error(y, pred))

def na_report(data):

    # Define the number of missing features
    data_na = (data.isnull().sum() / len(data))
    data_na = data_na.sort_values(ascending=False)
    na_report_data = pd.DataFrame({'NA Ratio' :data_na})
    print("Columns with more NA Values (Top 20)")
    print(na_report_data.head(20))

    return None

def preprocess_house_dataset(data):

    # Drop the ID column
    data.drop("Id", axis = 1, inplace = True)

    # I will manually feature engineer the features with more than 1% of missing data

    # Pool data, as PoolQC is pool quality, an NA means "No pool"
    data['PoolQC'] = data['PoolQC'].fillna("NoPool")

    # Miscelaneoes feature, in this case an NA means there is not a miscelaneoues feature
    data['MiscFeature'] = data['MiscFeature'].fillna("NoMisc")

    # Alley acces, NA means no alley access
    data['Alley'] = data['Alley'].fillna("NoAlleyAccess")

    # Fence, NA means no fence
    data['Fence'] = data['Fence'].fillna("NoFence")

    # Fire Place, NA means no fire place
    data['FireplaceQu'] = data['FireplaceQu'].fillna("NoFirePlace")

    # In the lot Frontage, I will use the mean of the Lot frontage for the Neighborhood
    data['LotFrontage'] = data['LotFrontage'].fillna(data.groupby('Neighborhood')['LotFrontage'].transform('median'))

    # All features related to basement have missing values that mean "No basement"
    data['BsmtQual'] = data['BsmtQual'].fillna("NoBasement")
    data['BsmtCond'] = data['BsmtCond'].fillna("NoBasement")
    data['BsmtExposure'] = data['BsmtExposure'].fillna("NoBasement")
    data['BsmtFinType1'] = data['BsmtFinType1'].fillna("NoBasement")
    data['BsmtFinType2'] = data['BsmtFinType2'].fillna("NoBasement")

    # In the features related to the garage, an NA means no garage
    data['GarageType'] = data['GarageType'].fillna("NoGarage")
    data['GarageFinish'] = data['GarageFinish'].fillna("NoGarage")
    data['GarageQual'] = data['GarageQual'].fillna("NoGarage")
    data['GarageCond'] = data['GarageCond'].fillna("NoGarage")
    data['GarageYrBlt'] = data['GarageYrBlt'].fillna("NoGarage")

    # There are numerical variables that are truly cathegorical, I will transform them into strings
    # Year sold
    data['YrSold'] = data['YrSold'].astype(str)

    # Month sold
    data['MoSold'] = data['MoSold'].astype(str)

    # Building class
    data['MSSubClass'] = data['MSSubClass'].astype(str)

    # Now, for all the other variables, I will generate a loop, that will fill NA with:
    # The Median of the neightbourhood in case of Numeric Variables and with
    # The mode of the neigbourhood in case of categorical Variables

    # For object types:
    for column in data.select_dtypes(['object']):
        data[column] = data[column].fillna(data.groupby('Neighborhood')[column].agg(lambda x:x.value_counts().index[0]))

    # For object types:
    for column in data.select_dtypes(['int64']):
        data[column] = data[column].fillna(data.groupby('Neighborhood')[column].transform('median'))

    # For object types:
    for column in data.select_dtypes(['float64']):
        data[column] = data[column].fillna(data.groupby('Neighborhood')[column].transform('median'))


    # If there are still some NA remaining is because the neighbourhood did not had a mode or median, so I will fill with the general mode or median

    # For object types:
    for column in data.select_dtypes(['object']):
        data[column] = data[column].fillna(data[column].agg(lambda x:x.value_counts().index[0]))

    # For object types:
    for column in data.select_dtypes(['int64']):
        data[column] = data[column].fillna(data[column].median())

    # For float types:
    for column in data.select_dtypes(['float64']):
        data[column] = data[column].fillna(data[column].median())


    # Define skweeness and transform to log

    #for column in data:

    #    if (data[column].dtype == np.float64 or data[column].dtype == np.int64):
            
    #        skewness = skew(data[column])

    #        if skewness > 1.0 or skewness < 1.0:

    #            data[column] = np.log1p(data[column])


    # One hot encoding:
    for column in data.select_dtypes(['object']):
        data = pd.get_dummies(data,prefix=[column], columns = [column], drop_first=False)

    return data

def RFECV_save(model, name, X, y, train, step , cv, n_jobs, scoring, folder):
    
    if not os.path.exists(folder + '/RFE_' + name + '.csv'):

        print('Inizializing the RFE of the' + name + 'GBR model')
        print("This will take a while!")
        rfecv = RFECV(model, step=step, cv=cv,verbose=True,n_jobs=n_jobs,scoring= scoring)
        rfecv = rfecv.fit(X, y)

        selected = train.iloc[:,rfecv.support_.tolist()]
        selected_col = selected.columns

        df = pd.DataFrame(data={"col": selected_col})
        df.to_csv(folder + '/RFE_' + name + '.csv',index=False)

        df = pd.DataFrame(data={"grid_scores": rfecv.grid_scores_})
        df.to_csv(folder + '/RFE_' + name + '_gridscores.csv',index=False)
    else:
        print('The RFE of the' + name + ' model was already perfomed')
        print('We will read it from' + folder + '/RFE_' + name + '.csv')

def GridCV_save(model, name, X, y, parameters , cv, n_jobs, scoring, folder):
    
    if not os.path.exists(folder + '/GridCV_' + name + '.csv'):

        print('Inizializing the Grid search of the' + name + 'GBR model')
        print("This will take a while!")
        gridCV = GridSearchCV(model, parameters,cv=cv,verbose=True,n_jobs=n_jobs,scoring= scoring)
        gridCV = gridCV.fit(X, y)

        df = pd.DataFrame(pd.DataFrame(gridCV.cv_results_))
        df.to_csv(folder + '/GridCV_' + name + '.csv',index=False)

    else:
        print('The Grid search of the' + name + ' model was already perfomed')
        print('We will read it from' + folder + '/GridCV_' + name + '.csv')

# New variables
def main():

    # Read the training data
    print("Reading training data")
    train = pd.read_csv('Input_Data/train.csv')
    y = train['SalePrice'].values
    y = np.log1p(y)
    train = train.drop('SalePrice', axis=1)

    test = pd.read_csv('Input_Data/test.csv')
    test_ID = test["Id"]

    full_data=pd.concat([train,test],axis=0,sort=False)



    na_report(full_data)

    print("Preprocessing the data")
    full_data = preprocess_house_dataset(full_data)

    na_report(full_data)

    train = full_data[0:1460]
    test = full_data[1460:2919]

    #train.to_csv('Results/preprocessed_train.csv')


    # Read the training data
    #print("Reading training data")
    #test = pd.read_csv('Input_Data/test.csv')
    #na_report(test)
    #print("Preprocessing the training data")
    #test = preprocess_house_dataset(test)
    #na_report(test)

    #X = train.drop('SalePrice', axis=1).values
    
    X = train.values

    # Gradient boosting regressor
    estimator = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=3, random_state=0, loss='ls')
    RFECV_save(model = estimator, name = 'GBR', X = X, y = y, train = train,step = 1, cv = 5, n_jobs = -1, scoring= 'neg_mean_squared_error',folder = './Intermediate_Results')

    # AdaBoost
    #estimator = AdaBoostRegressor(n_estimators=500, random_state=0)
    #RFECV_save(model = estimator, name = 'AdaB', X = X, y = y, train = train,step = 1, cv = 6, n_jobs = -1, scoring= 'neg_mean_squared_error',folder = './Intermediate_Results')

    # Xtreme Gradient Boosting
    estimator = XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=3, random_state=0, objective='reg:squarederror')
    RFECV_save(model = estimator, name = 'XGBR', X = X, y = y, train = train,step = 1, cv = 5, n_jobs = -1, scoring= 'neg_mean_squared_error',folder = './Intermediate_Results')

    # CatBoost
    estimator = CatBoostRegressor(n_estimators=500, learning_rate=0.1,  random_state=0, verbose=False)
    RFECV_save(model = estimator, name = 'CatBoost', X = X, y = y, train = train,step = 1, cv = 5, n_jobs = -1, scoring= 'neg_mean_squared_error',folder = './Intermediate_Results')

    # LightGBM
    #estimator = lgb.LGBMRegressor(objective='regression',learning_rate=0.01, n_estimators=500)
    #RFECV_save(model = estimator, name = 'LightGBM', X = X, y = y, train = train,step = 1, cv = 6, n_jobs = -1, scoring= 'neg_mean_squared_error',folder = './Intermediate_Results')

    # RF
    estimator = RandomForestRegressor(n_estimators=500,random_state=0)
    RFECV_save(model = estimator, name = 'RF', X = X, y = y, train = train,step = 1, cv = 5, n_jobs = -1, scoring= 'neg_mean_squared_error',folder = './Intermediate_Results')


    #---------------------------------------------

    # Hyperparameter tuning


    # RF

    col_RFE = pd.read_csv('Intermediate_Results/RFE_RF.csv')
    train_selected = train[train.columns.intersection(col_RFE['col'].values)]
    X_sub = train_selected.values
    parameters = {'n_estimators':[500,1500,3000], 'max_features':[6, 22, 43],'max_depth' : [6, 10, None]}
    estimator = RandomForestRegressor(random_state=24)
    GridCV_save(model = estimator, name = 'RF', X = X_sub, y = y, parameters = parameters , cv = 5, n_jobs = 3, scoring= 'neg_mean_squared_error',folder = './Intermediate_Results')

    # Catboost

    col_RFE = pd.read_csv('Intermediate_Results/RFE_CatBoost.csv')
    train_selected = train[train.columns.intersection(col_RFE['col'].values)]
    X_sub = train_selected.values
    parameters = {'n_estimators':[500, 1000,2000], 'learning_rate':[0.01, 0.05, 0.1], 'depth' : [6,8,10]}
    estimator = CatBoostRegressor(random_state=24)
    GridCV_save(model = estimator, name = 'CatBoost', X = X_sub, y = y, parameters = parameters , cv = 5, n_jobs = 3, scoring= 'neg_mean_squared_error',folder = './Intermediate_Results')

    # GBR

    col_RFE = pd.read_csv('Intermediate_Results/RFE_GBR.csv')
    train_selected = train[train.columns.intersection(col_RFE['col'].values)]
    X_sub = train_selected.values
    parameters = {'n_estimators':[500, 1000,2000], 'learning_rate':[0.01, 0.05, 0.1], 'max_depth' : [6,8,10]}
    estimator = GradientBoostingRegressor(random_state=24, loss='ls')
    GridCV_save(model = estimator, name = 'GBR', X = X_sub, y = y, parameters = parameters , cv = 5, n_jobs = -1, scoring= 'neg_mean_squared_error',folder = './Intermediate_Results')

    # XGB

    col_RFE = pd.read_csv('Intermediate_Results/RFE_XGBR.csv')
    train_selected = train[train.columns.intersection(col_RFE['col'].values)]
    X_sub = train_selected.values
    #parameters = {'n_estimators':[500, 1000,2000], 'max_depth':[3, 4, 5],'learning_rate':[0.01, 0.05, 0.1],'subsample':[0.6, 0.8, 1.0],'gamma':[0,1,5],'colsample_bytree':[0.3,0.5,0.8]}
    parameters = {'n_estimators':[500, 1000,2000], 'max_depth':[3, 4, 5],'learning_rate':[0.01, 0.05, 0.1]}
    estimator = XGBRegressor(random_state=24,objective='reg:squarederror')
    GridCV_save(model = estimator, name = 'XGBR', X = X_sub, y = y, parameters = parameters , cv = 4, n_jobs = -1, scoring= 'neg_mean_squared_error',folder = './Intermediate_Results')


    #------------------------------------------------

    # Final training and testing

    # Testing and submission
    #print("Reading testing data")
    #test = pd.read_csv('Input_Data/test.csv')

    #test_ID = test["Id"]

    #na_report(test)
    #print("Preprocessing the test data")
    #test = preprocess_house_dataset(test)
    #na_report(test)

    #------------------------------------------------

    # RF

    # Get the selected variables
    col_RFE = pd.read_csv('Intermediate_Results/RFE_RF.csv')
    train_selected = train[train.columns.intersection(col_RFE['col'].values)]
    X_sub_RF = train_selected.values

    # Read the grid search dataframe
    grid = pd.read_csv("./Intermediate_Results/GridCV_RF.csv")

    # Search the bigger score index in the dataframe
    row_max = grid['mean_test_score'].argmax()

    # Get the the better performing step and number of iterations
    n_estimators_max = int(grid['param_n_estimators'].values[row_max])
    max_features_max = int(grid['param_max_features'].values[row_max])
    max_depth_max = grid['param_max_depth'].values[row_max]

    if pd.isna(max_depth_max):
        estimator_RF = RandomForestRegressor(random_state=24, n_estimators = n_estimators_max, max_features = max_features_max,max_depth = None)
    else:
        estimator_RF = RandomForestRegressor(random_state=24, n_estimators = n_estimators_max, max_features = max_features_max,max_depth = max_depth_max)
    
    estimator_RF.fit(X_sub_RF,y)

    test_selected = test[test.columns.intersection(col_RFE['col'].values)]
    X_test_RF = test_selected.values
    test_pred = estimator_RF.predict(X_test_RF)

    submission = pd.DataFrame()
    submission['Id'] = test_ID
    submission['SalePrice'] = np.expm1(test_pred)
    submission.to_csv('Results/submission_RF.csv', index=False)
    print('Final RF model trained and resred, predictions can be found in Results/submission_RF.csv')

    # Gradient boosting Regression

    # Get the selected variables
    col_RFE = pd.read_csv('Intermediate_Results/RFE_GBR.csv')
    train_selected = train[train.columns.intersection(col_RFE['col'].values)]
    X_sub_GBR = train_selected.values

    # Read the grid search dataframe
    grid = pd.read_csv("./Intermediate_Results/GridCV_GBR.csv")

    # Search the bigger score index in the dataframe
    row_max = grid['mean_test_score'].argmax()

    # Get the the better performing step and number of iterations
    n_estimators_max = int(grid['param_n_estimators'].values[row_max])
    max_depth_max = int(grid['param_max_depth'].values[row_max])
    learning_rate_max = float(grid['param_learning_rate'].values[row_max])

    estimator_GBR = GradientBoostingRegressor(random_state=24, n_estimators = n_estimators_max,max_depth = max_depth_max,learning_rate = learning_rate_max, loss='ls')
    estimator_GBR.fit(X_sub_GBR,y)

    test_selected = test[test.columns.intersection(col_RFE['col'].values)]
    X_test_GBR = test_selected.values
    test_pred = estimator_GBR.predict(X_test_GBR)

    submission= pd.DataFrame()
    submission['Id'] = test_ID
    submission['SalePrice'] = np.expm1(test_pred)
    submission.to_csv('Results/submission_GBR.csv', index=False)
    print('Final GBR model trained and resred, predictions can be found in Results/submission_GBR.csv')



    # XGBoost

    # Get the selected variables
    col_RFE = pd.read_csv('Intermediate_Results/RFE_XGBR.csv')
    train_selected = train[train.columns.intersection(col_RFE['col'].values)]
    X_sub_XGBR = train_selected.values

    # Read the grid search dataframe
    grid = pd.read_csv("./Intermediate_Results/GridCV_XGBR.csv")

    # Search the bigger score index in the dataframe
    row_max = grid['mean_test_score'].argmax()

    #Get the the better performing step and number of iterations
    n_estimators_max = int(grid['param_n_estimators'].values[row_max])
    max_depth_max = int(grid['param_max_depth'].values[row_max])
    learning_rate_max = float(grid['param_learning_rate'].values[row_max])
    #subsample_max = float(grid['param_subsample'].values[row_max])
    #gamma_max = float(grid['param_gamma'].values[row_max])
    #colsample_bytree_max = float(grid['param_colsample_bytree'].values[row_max])

    #estimator = XGBRegressor(random_state=24, n_estimators = n_estimators_max, max_depth = max_depth_max, 
    #    learning_rate = learning_rate_max, subsample = subsample_max, gamma = gamma_max, colsample_bytree = colsample_bytree_max)

    estimator_XGBR = XGBRegressor(objective ='reg:squarederror',random_state=24, n_estimators = n_estimators_max, max_depth = max_depth_max, learning_rate = learning_rate_max)


    estimator_XGBR.fit(X_sub_XGBR,y)

    test_selected = test[test.columns.intersection(col_RFE['col'].values)]
    X_test_XGBR = test_selected.values
    test_pred = estimator_XGBR.predict(X_test_XGBR)

    submission = pd.DataFrame()
    submission['Id'] = test_ID
    submission['SalePrice'] = np.expm1(test_pred)
    submission.to_csv('Results/submission_XGBR.csv', index=False)
    print('Final XGBR model trained and resred, predictions can be found in Results/submission_XGBR.csv')


    # CatBoost

    # Get the selected variables
    col_RFE = pd.read_csv('Intermediate_Results/RFE_CatBoost.csv')
    train_selected = train[train.columns.intersection(col_RFE['col'].values)]
    X_sub_CatBoost = train_selected.values

    # Read the grid search dataframe
    grid = pd.read_csv("./Intermediate_Results/GridCV_CatBoost.csv")

    # Search the bigger score index in the dataframe
    row_max = grid['mean_test_score'].argmax()

    # Get the the better performing step and number of iterations
    n_estimators_max = int(grid['param_n_estimators'].values[row_max])
    depth_max = int(grid['param_depth'].values[row_max])
    learning_rate_max = float(grid['param_learning_rate'].values[row_max])

    estimator_CatBoost = CatBoostRegressor(random_state=24, n_estimators = n_estimators_max, depth = depth_max, learning_rate = learning_rate_max)

    estimator_CatBoost.fit(X_sub_CatBoost,y)

    test_selected = test[test.columns.intersection(col_RFE['col'].values)]
    X_test_CatBoost = test_selected.values
    test_pred = estimator_CatBoost.predict(X_test_CatBoost)

    submission = pd.DataFrame()
    submission['Id'] = test_ID
    submission['SalePrice'] = np.expm1(test_pred)
    submission.to_csv('Results/submission_CatBoost.csv', index=False)
    print('Final CatBoost model trained and resred, predictions can be found in Results/submission_Catboost.csv')



    

    # Model Stacking

    k_fold = KFold(n_splits=5, random_state=23,shuffle=True)

    # Test steps for the gradient descent from 0.1 to 1
    CatBoost_array = np.arange(start=0.05, stop=1.05, step=0.05)

    # Test steps for the gradient descent from 0.1 to 1
    XGBR_array = np.arange(start=0.05, stop=1.1, step=0.05)

    # Test different numbers of iterations from 10 110
    RF_array = np.arange(start=0.05, stop=1.1, step=0.05)

    # Test different numbers of iterations from 10 110
    GBR_array = np.arange(start=0.05, stop=1.1, step=0.05)

    # Store the partial results in lists
    CatBoost_full = []
    XGBR_full = []
    RF_full = []
    GBR_full = []
    msle_full = []

    # Go though the 5 splits
    for kfold_train_index, kfold_test_index in k_fold.split(X, y):

        # Get the split into train and test for catboost
        kfold_x_train_CatBoost, kfold_x_test_CatBoost = X_sub_CatBoost[kfold_train_index][:], X_sub_CatBoost[kfold_test_index][:]

        # Get the split into train and test for XGBR
        kfold_x_train_XGBR, kfold_x_test_XGBR = X_sub_XGBR[kfold_train_index][:], X_sub_XGBR[kfold_test_index][:]

        # Get the split into train and test for RF
        kfold_x_train_RF, kfold_x_test_RF = X_sub_RF[kfold_train_index][:], X_sub_RF[kfold_test_index][:]

        # Get the split into train and test for GBR
        kfold_x_train_GBR, kfold_x_test_GBR = X_sub_GBR[kfold_train_index][:], X_sub_GBR[kfold_test_index][:]

        # Dependent values
        kfold_y_train, kfold_y_test = y[kfold_train_index], y[kfold_test_index]

                                
        #Catboost
        estimator_CatBoost.fit(kfold_x_train_CatBoost,kfold_y_train)
        predicted_CatBoost_sub = estimator_CatBoost.predict(kfold_x_test_CatBoost)

        #XGBR
        estimator_XGBR.fit(kfold_x_train_XGBR,kfold_y_train)
        predicted_XGBR_sub = estimator_XGBR.predict(kfold_x_test_XGBR)

        #RF
        estimator_RF.fit(kfold_x_train_RF,kfold_y_train)
        predicted_RF_sub = estimator_RF.predict(kfold_x_test_RF)

        #GBR
        estimator_GBR.fit(kfold_x_train_GBR,kfold_y_train)
        predicted_GBR_sub = estimator_GBR.predict(kfold_x_test_GBR)

        for CatBoost_w in CatBoost_array:

            for XGBR_w in XGBR_array:

                for RF_w in RF_array:

                    for GBR_w in GBR_array:

                        # We will execute the search only if the 3 weights add up 1
                        if ((CatBoost_w+XGBR_w+RF_w+GBR_w)>0.99) and ((CatBoost_w+XGBR_w+RF_w+GBR_w)<1.01):

                            predict_stacked = predicted_CatBoost_sub * CatBoost_w + predicted_XGBR_sub * XGBR_w + predicted_RF_sub * RF_w + predicted_GBR_sub * GBR_w

                            msle_k = mean_squared_error(kfold_y_test,predict_stacked)

                            # Store the weights and rmse in the lists
                            CatBoost_full.append(CatBoost_w)
                            XGBR_full.append(XGBR_w)
                            RF_full.append(RF_w)
                            GBR_full.append(GBR_w)
                            msle_full.append(msle_k)
        
    dic = {'CatBoost_w':CatBoost_full,'XGBR_w':XGBR_full,'RF_w':RF_full,'GBR_w':GBR_full,'msle':msle_full}
    df_stacked = pd.DataFrame(dic)
    df_stacked.to_csv('Intermediate_Results/stacked_search.csv')

    df_stacked_mean = df_stacked.groupby(['CatBoost_w', 'XGBR_w','RF_w','GBR_w'],as_index=False)['msle'].mean()
    df_stacked_mean.to_csv('Intermediate_Results/stacked_search_mean.csv')

    # Search the bigger score index in the dataframe
    row_max = df_stacked_mean['msle'].argmin()

    # Get the the better performing step and number of iterations
    CatBoost_w = float(df_stacked_mean['CatBoost_w'].values[row_max])
    XGBR_w = float(df_stacked_mean['XGBR_w'].values[row_max])
    RF_w = float(df_stacked_mean['RF_w'].values[row_max])
    GBR_w = float(df_stacked_mean['GBR_w'].values[row_max])



    #Catboost
    estimator_CatBoost.fit(X_sub_CatBoost,y)
    predicted_CatBoost = estimator_CatBoost.predict(X_test_CatBoost)

    #XGBR
    estimator_XGBR.fit(X_sub_XGBR,y)
    predicted_XGBR = estimator_XGBR.predict(X_test_XGBR)

    #RF
    estimator_RF.fit(X_sub_RF,y)
    predicted_RF = estimator_RF.predict(X_test_RF)

    #GBR
    estimator_GBR.fit(X_sub_GBR,y)
    predicted_GBR = estimator_GBR.predict(X_test_GBR)

    predict_stacked_final = predicted_CatBoost * CatBoost_w + predicted_XGBR * XGBR_w + predicted_RF * RF_w + predicted_GBR * GBR_w
    submission = pd.DataFrame()
    submission['Id'] = test_ID
    submission['SalePrice'] = np.expm1(predict_stacked_final)
    submission.to_csv('Results/submission_stacked.csv', index=False)






if __name__ == '__main__':
	main()







