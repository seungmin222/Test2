import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

def sort_dataset(dataset_df):

    sorted_df = dataset_df.sort_values(by='year')
    return sorted_df

def split_dataset(dataset_df):

    dataset_df['salary'] *= 0.001
    
    train_df = dataset_df.iloc[:1718]
    test_df = dataset_df.iloc[1718:]
    
    X_train, Y_train = train_df.drop('salary', axis=1), train_df['salary']
    X_test, Y_test = test_df.drop('salary', axis=1), test_df['salary']
    
    return X_train, X_test, Y_train, Y_test

def extract_numerical_cols(dataset_df):
 
    numerical_cols = ['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']
    numerical_df = dataset_df[numerical_cols]
    return numerical_df

def train_predict_decision_tree(X_train, Y_train, X_test):

    model = DecisionTreeRegressor()
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)
    
    return predictions

def train_predict_random_forest(X_train, Y_train, X_test):

    model = RandomForestRegressor()
    model.fit(X_train, Y_train)
    
    predictions = model.predict(X_test)
    
    return predictions

def train_predict_svm(X_train, Y_train, X_test):

    model = SVR()
    pipeline = StandardScaler().fit(X_train)
    X_train_scaled = pipeline.transform(X_train)
    model.fit(X_train_scaled, Y_train)
    
    X_test_scaled = pipeline.transform(X_test)
    predictions = model.predict(X_test_scaled)
    
    return predictions

def calculate_RMSE(labels, predictions):

    rmse = np.sqrt(mean_squared_error(labels, predictions))
    return rmse


if __name__=='__main__':
	#DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
	data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
	
	sorted_df = sort_dataset(data_df)	
	X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)
	
	X_train = extract_numerical_cols(X_train)
	X_test = extract_numerical_cols(X_test)

	dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
	rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
	svm_predictions = train_predict_svm(X_train, Y_train, X_test)
	
	print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))	
	print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))	
	print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))