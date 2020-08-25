###################################
# Julian Cabezas Pena
# Big data Analysis and Project
# Project 1 1
# House price prediction using decision tree based methods
####################################

# Import libraries
# Basic libraries
import numpy as np
import pandas as pd
import os

# Model training and error measures
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# Prediction models
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from catboost import CatBoostRegressor

# Calculates thd root mean squared log error given truth and prediction
def rmsle(y,pred):
    return np.sqrt(mean_squared_log_error(y, pred))

# Generates a report of the top 10 columns with more NA records
def na_report(data):

    # Define the number of missing features and sort them
    data_na = (data.isnull().sum() / len(data))
    data_na = data_na.sort_values(ascending=False)

    # Convert to pandas dataframe and print the top 20
    na_report_data = pd.DataFrame({'NA Ratio' :data_na})
    print("Columns with more NA Values (Top 10)")
    print(na_report_data.head(10))

    return None

# Preprocessing of the housing dataset
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

    # One hot encoding of cathegorical variables:
    for column in data.select_dtypes(['object']):
        data = pd.get_dummies(data,prefix=[column], columns = [column], drop_first=False)

    return data

# Performs Recursive Feature elimination with cross validation of an specific model. It runs if the file does not exists
# model: skitlearn like estimator
# name: String with the name of the model (For the print messages and files)
# train: housing training dataset
# X,y,step,cv,n_jobs,scoring are RFECV model parameters
# folder: route of the folder in which to store the results of this algorithm
def RFECV_save(model, name, X, y, train, step , cv, n_jobs, scoring, folder):
    
    # Check if the RFE has been previosly performed
    if not os.path.exists(folder + '/RFE_' + name + '.csv'):

        print('Initializing the RFE of the ' + name + 'model')
        print("This will take a while!")
        # Perform the RFE and fit the model
        rfecv = RFECV(model, step=step, cv=cv,verbose=True,n_jobs=n_jobs,scoring= scoring)
        rfecv = rfecv.fit(X, y)

        # Get the names of the selected columns
        selected = train.iloc[:,rfecv.support_.tolist()]
        selected_col = selected.columns

        # get the scores (negative mean square error) and the names of the features and print to csv
        df = pd.DataFrame(data={"col": selected_col})
        df.to_csv(folder + '/RFE_' + name + '.csv',index=False)

        df = pd.DataFrame(data={"grid_scores": rfecv.grid_scores_})
        df.to_csv(folder + '/RFE_' + name + '_gridscores.csv',index=False)
    else:
        print('The RFE of the ' + name + ' model was already perfomed')
        print('The RFE Results will be read from ' + folder + '/RFE_' + name + '.csv')

# Performs gridh search to tune the model parameters. It runs if the file does not exists
# model: scikitlearn like estimator
# name: String with the name of the model (For the print messages and files)
# parameters: Dictionaty to the parameters to test
# X,y,step,cv,n_jobs,scoring are GridSearchCV parameters
# folder: route of the folder in which to store the results of this algorithm
def GridCV_save(model, name, X, y, parameters , cv, n_jobs, scoring, folder):
    
    # Check if the grid search has already been performed
    if not os.path.exists(folder + '/GridCV_' + name + '.csv'):

        print('Inizializing the Grid search of the' + name + 'GBR model')
        print("This will take a while!")

        # Performs grid search with all the possible parameters combinations
        gridCV = GridSearchCV(model, parameters,cv=cv,verbose=True,n_jobs=n_jobs,scoring= scoring)
        gridCV = gridCV.fit(X, y)

        # Write results to csv
        df = pd.DataFrame(pd.DataFrame(gridCV.cv_results_))
        df.to_csv(folder + '/GridCV_' + name + '.csv',index=False)

    else:
        print('The Grid search of the ' + name + ' model was already perfomed')
        print('The Grid search results will be read from ' + folder + '/GridCV_' + name + '.csv')


# ------------------------------------------------
# Main execution of the workflow
def main():

    # Step 0: Data reading and preprocessing
    print("")
    print("Step 0: Data reading and preprocessing")
    print("-------------------------------------")


    # Read the training data
    print("Reading training data")
    train = pd.read_csv('Input_Data/train.csv')

    # Het the sale price and apply log(x+1 transformation)
    y = train['SalePrice'].values
    y = np.log1p(y)

    # Drop the target variable 
    train = train.drop('SalePrice', axis=1)

    # Check NAs in columns
    na_report(train)

    #Read the test data and store the ID
    test = pd.read_csv('Input_Data/test.csv')
    test_ID = test["Id"]

    #Concatenate train and test to preprocess the data
    full_data=pd.concat([train,test],axis=0,sort=False)
    print("Preprocessing the data")
    full_data = preprocess_house_dataset(full_data)

    # Split again into train and test
    train = full_data[0:1460]
    test = full_data[1460:2919]
    
    # Generate an X matrix with the explanatory variables
    X = train.values

    #----------------------------------------------------------
    # Step 1: Recursive Feature Elimination

    print("")
    print("Step 1: Recursive Feature Elimination")
    print("-------------------------------------")

    # For each model, perform recursive feature elimination and 5-fold cross validation
    # The results will be stored on .csv files into the "./Intermediate_Results" folder
    # If the RFE was already performed the algorithm will just read the data in the csv

    # Random Forest RFE
    estimator = RandomForestRegressor(n_estimators=500,random_state=0)
    RFECV_save(model = estimator, name = 'RF', X = X, y = y, train = train,step = 1, cv = 5, n_jobs = -1, scoring= 'neg_mean_squared_error',folder = './Intermediate_Results')

    # Gradient boosting regressor RFE
    estimator = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=3, random_state=0, loss='ls')
    RFECV_save(model = estimator, name = 'GBR', X = X, y = y, train = train,step = 1, cv = 5, n_jobs = -1, scoring= 'neg_mean_squared_error',folder = './Intermediate_Results')

    # Xtreme Gradient Boosting RFE
    estimator = XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=3, random_state=0, objective='reg:squarederror')
    RFECV_save(model = estimator, name = 'XGBR', X = X, y = y, train = train,step = 1, cv = 5, n_jobs = -1, scoring= 'neg_mean_squared_error',folder = './Intermediate_Results')

    # CatBoost RFE
    estimator = CatBoostRegressor(n_estimators=500, learning_rate=0.1,  random_state=0, verbose=False)
    RFECV_save(model = estimator, name = 'CatBoost', X = X, y = y, train = train,step = 1, cv = 5, n_jobs = -1, scoring= 'neg_mean_squared_error',folder = './Intermediate_Results')

    #---------------------------------------------
    # Step 2: Hyperparameter tuning using grid search

    print("")
    print("Step 2: Hyperparameter tuning using grid search")
    print("-------------------------------------")

    # Random forest

    # Read the results from the RFE and get the seected variables
    col_RFE = pd.read_csv('Intermediate_Results/RFE_RF.csv')
    train_selected = train[train.columns.intersection(col_RFE['col'].values)]
    X_sub = train_selected.values

    # Define the grid of parameters to test and perform grid search using 5-fold cross validation
    parameters = {'n_estimators':[500,1500,3000], 'max_features':[6, 22, 43],'max_depth' : [6, 10, None]}
    estimator = RandomForestRegressor(random_state=24)
    GridCV_save(model = estimator, name = 'RF', X = X_sub, y = y, parameters = parameters , cv = 5, n_jobs = 3, scoring= 'neg_mean_squared_error',folder = './Intermediate_Results')

    # Gradient Boosting regression

    # Read the results from the RFE and get the seected variables
    col_RFE = pd.read_csv('Intermediate_Results/RFE_GBR.csv')
    train_selected = train[train.columns.intersection(col_RFE['col'].values)]
    X_sub = train_selected.values

    # Define the grid of parameters to test and perform grid search using 5-fold cross validation
    parameters = {'n_estimators':[500, 1000,2000], 'learning_rate':[0.01, 0.05, 0.1], 'max_depth' : [3,4,5]}
    estimator = GradientBoostingRegressor(random_state=24, loss='ls')
    GridCV_save(model = estimator, name = 'GBR', X = X_sub, y = y, parameters = parameters , cv = 5, n_jobs = -1, scoring= 'neg_mean_squared_error',folder = './Intermediate_Results')

    # XGBoosting

    # Read the results from the RFE and get the seected variables
    col_RFE = pd.read_csv('Intermediate_Results/RFE_XGBR.csv')
    train_selected = train[train.columns.intersection(col_RFE['col'].values)]
    X_sub = train_selected.values

    # Define the grid of parameters to test and perform grid search using 5-fold cross validation
    parameters = {'n_estimators':[500, 1000,2000], 'max_depth':[3, 4, 5],'learning_rate':[0.01, 0.05, 0.1]}
    estimator = XGBRegressor(random_state=24,objective='reg:squarederror')
    GridCV_save(model = estimator, name = 'XGBR', X = X_sub, y = y, parameters = parameters , cv = 4, n_jobs = -1, scoring= 'neg_mean_squared_error',folder = './Intermediate_Results')

    # Catboost

    # Read the results from the RFE and get the seected variables
    col_RFE = pd.read_csv('Intermediate_Results/RFE_CatBoost.csv')
    train_selected = train[train.columns.intersection(col_RFE['col'].values)]
    X_sub = train_selected.values

    # Define the grid of parameters to test and perform grid search using 5-fold cross validation
    parameters = {'n_estimators':[500, 1000,2000], 'learning_rate':[0.01, 0.05, 0.1], 'depth' : [6,8,10]}
    estimator = CatBoostRegressor(random_state=24)
    GridCV_save(model = estimator, name = 'CatBoost', X = X_sub, y = y, parameters = parameters , cv = 5, n_jobs = 3, scoring= 'neg_mean_squared_error',folder = './Intermediate_Results')


    #------------------------------------------------
    # Step 3: Final Training, Validation and Testing

    print("")
    print("Step 3: Final training and testing of the models")
    print("-------------------------------------")

    ###############
    # Random Forest

    print("")
    print("Testing the final Random Forest regression")

    # Get the selected variables
    col_RFE = pd.read_csv('Intermediate_Results/RFE_RF.csv')
    train_selected = train[train.columns.intersection(col_RFE['col'].values)]
    X_sub_RF = train_selected.values

    # Read the grid search dataframe
    grid = pd.read_csv("./Intermediate_Results/GridCV_RF.csv")

    # Search the bigger score index in the dataframe
    row_max = grid['mean_test_score'].argmax()
    min_error = np.sqrt(-1 * np.max(grid['mean_test_score']))

    # Get the the better performing parameters
    n_estimators_max = int(grid['param_n_estimators'].values[row_max])
    max_features_max = int(grid['param_max_features'].values[row_max])
    max_depth_max = grid['param_max_depth'].values[row_max]

    print("Optimal parameters:")
    print("n_estimators: ",n_estimators_max)
    print("max_features: ",max_features_max)
    print("max_depth:",max_depth_max)
    print("Minimum RMSLE: ", min_error)

    # Train the final mode, in case the max_depth in NA, interpret is as "None" (it means no maximum depth in RF)
    if pd.isna(max_depth_max):
        estimator_RF = RandomForestRegressor(random_state=24, n_estimators = n_estimators_max, max_features = max_features_max,max_depth = None)
    else:
        estimator_RF = RandomForestRegressor(random_state=24, n_estimators = n_estimators_max, max_features = max_features_max,max_depth = max_depth_max)
    
    # Train the final model
    estimator_RF.fit(X_sub_RF,y)

    # Generate predictions on the test data
    test_selected = test[test.columns.intersection(col_RFE['col'].values)]
    X_test_RF = test_selected.values
    test_pred = estimator_RF.predict(X_test_RF)

    # Generate the submission for Kaggle
    submission = pd.DataFrame()
    submission['Id'] = test_ID
    submission['SalePrice'] = np.expm1(test_pred)
    submission.to_csv('Results/submission_RF.csv', index=False)
    print('Final RF model trained and tested, predictions can be found in Results/submission_RF.csv')

    ##############################
    # Gradient boosting Regression

    print("")
    print("Testing the final Gradinet Boosting regression model")

    # Get the selected variables
    col_RFE = pd.read_csv('Intermediate_Results/RFE_GBR.csv')
    train_selected = train[train.columns.intersection(col_RFE['col'].values)]
    X_sub_GBR = train_selected.values

    # Read the grid search dataframe
    grid = pd.read_csv("./Intermediate_Results/GridCV_GBR.csv")

    # Search the bigger score index in the dataframe
    row_max = grid['mean_test_score'].argmax()
    min_error = np.sqrt(-1 * np.max(grid['mean_test_score']))

    # Get the the better performing step and number of iterations
    n_estimators_max = int(grid['param_n_estimators'].values[row_max])
    max_depth_max = int(grid['param_max_depth'].values[row_max])
    learning_rate_max = float(grid['param_learning_rate'].values[row_max])

    print("Optimal parameters:")
    print("n_estimators: ",n_estimators_max)
    print("depth_max: ",max_depth_max)
    print("learning _ate:",learning_rate_max)
    print("Minimum RMSLE: ", min_error)

    # Train the final model using the best performing parameters
    estimator_GBR = GradientBoostingRegressor(random_state=24, n_estimators = n_estimators_max,max_depth = max_depth_max,learning_rate = learning_rate_max, loss='ls')
    estimator_GBR.fit(X_sub_GBR,y)

    # Generate predictions on the test data
    test_selected = test[test.columns.intersection(col_RFE['col'].values)]
    X_test_GBR = test_selected.values
    test_pred = estimator_GBR.predict(X_test_GBR)

    # Generate the submission for Kaggle
    submission= pd.DataFrame()
    submission['Id'] = test_ID
    submission['SalePrice'] = np.expm1(test_pred)
    submission.to_csv('Results/submission_GBR.csv', index=False)
    print('Final GBR model trained and tested, predictions can be found in Results/submission_GBR.csv')


    #########################
    # XGBoost

    print("")
    print("Testing the final Extreme Gradinet Boosting regression model")

    # Get the selected variables
    col_RFE = pd.read_csv('Intermediate_Results/RFE_XGBR.csv')
    train_selected = train[train.columns.intersection(col_RFE['col'].values)]
    X_sub_XGBR = train_selected.values

    # Read the grid search dataframe
    grid = pd.read_csv("./Intermediate_Results/GridCV_XGBR.csv")

    # Search the bigger score index in the dataframe
    row_max = grid['mean_test_score'].argmax()
    min_error = np.sqrt(-1 * np.max(grid['mean_test_score']))

    # Get the the better performing step and number of iterations
    n_estimators_max = int(grid['param_n_estimators'].values[row_max])
    max_depth_max = int(grid['param_max_depth'].values[row_max])
    learning_rate_max = float(grid['param_learning_rate'].values[row_max])

    print("Optimal parameters:")
    print("n_estimators: ",n_estimators_max)
    print("depth_max: ",max_depth_max)
    print("learning _ate:",learning_rate_max)
    print("Minimum RMSLE: ", min_error)

    # Train the final model using the best performing parameters
    estimator_XGBR = XGBRegressor(objective ='reg:squarederror',random_state=24, n_estimators = n_estimators_max, max_depth = max_depth_max, learning_rate = learning_rate_max, verbose=False)
    estimator_XGBR.fit(X_sub_XGBR,y)

    # Generate predictions on the test data
    test_selected = test[test.columns.intersection(col_RFE['col'].values)]
    X_test_XGBR = test_selected.values
    test_pred = estimator_XGBR.predict(X_test_XGBR)

    # Generate the submission file for Kaggle
    submission = pd.DataFrame()
    submission['Id'] = test_ID
    submission['SalePrice'] = np.expm1(test_pred)
    submission.to_csv('Results/submission_XGBR.csv', index=False)
    print('Final XGBR model trained and tested, predictions can be found in Results/submission_XGBR.csv')

    ##############
    # CatBoost

    print("")
    print("Testing the final Cathegorical Boosting regression model")

    # Get the selected variables
    col_RFE = pd.read_csv('Intermediate_Results/RFE_CatBoost.csv')
    train_selected = train[train.columns.intersection(col_RFE['col'].values)]
    X_sub_CatBoost = train_selected.values

    # Read the grid search dataframe
    grid = pd.read_csv("./Intermediate_Results/GridCV_CatBoost.csv")

    # Search the bigger score index in the dataframe
    row_max = grid['mean_test_score'].argmax()
    min_error = np.sqrt(-1 * np.max(grid['mean_test_score']))

    # Get the the better performing step and number of iterations
    n_estimators_max = int(grid['param_n_estimators'].values[row_max])
    depth_max = int(grid['param_depth'].values[row_max])
    learning_rate_max = float(grid['param_learning_rate'].values[row_max])

    print("Optimal parameters:")
    print("n_estimators: ",n_estimators_max)
    print("depth_max: ",depth_max)
    print("learning _ate:",learning_rate_max)
    print("Minimum RMSLE: ", min_error)

    # Train the final model using the best performing parameters
    estimator_CatBoost = CatBoostRegressor(random_state=24, n_estimators = n_estimators_max, depth = depth_max, learning_rate = learning_rate_max, silent = True)
    estimator_CatBoost.fit(X_sub_CatBoost,y)

    # Predict SateValue from the test data
    test_selected = test[test.columns.intersection(col_RFE['col'].values)]
    X_test_CatBoost = test_selected.values
    test_pred = estimator_CatBoost.predict(X_test_CatBoost)

    # Generate the submission for Kaggle
    submission = pd.DataFrame()
    submission['Id'] = test_ID
    submission['SalePrice'] = np.expm1(test_pred)
    submission.to_csv('Results/submission_CatBoost.csv', index=False)
    print('Final CatBoost model trained and tested, predictions can be found in Results/submission_Catboost.csv')

    print("")
    print('Execution finalized, the .csv for Kaggle dubmission can be found in ./Results')


if __name__ == '__main__':
	main()







