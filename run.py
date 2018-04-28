import pandas as pd
import numpy as np
import _user_input as user_input
from _plot_func import *
from mdlp.discretization import MDLP
from _support_func import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


#########################################################################
# data load
#########################################################################

_raw_data = pd.read_csv(user_input._data_dir + user_input._input_file_name, header=0)

# print(_raw_data.head())

_raw_data_imp_cols = _raw_data.drop(user_input._redundant_cols, axis=1)

# print(_raw_data_imp_cols.head())
# print(_raw_data_imp_cols.dtypes)

_raw_data_imp_cols[user_input._categorical_features + [user_input._output_col]] = _raw_data_imp_cols[user_input._categorical_features + [user_input._output_col]]\
    .apply(lambda x: x.astype('category'))

_raw_data_imp_cols[user_input._integer_features] = _raw_data_imp_cols[user_input._integer_features]\
    .apply(lambda x: x.astype('int64'))

_non_float_features = user_input._categorical_features + user_input._integer_features + [user_input._output_col]

_raw_data_imp_cols[_raw_data_imp_cols.columns.difference(_non_float_features)] = _raw_data_imp_cols[
    _raw_data_imp_cols.columns.difference(_non_float_features)]\
    .apply(lambda x: x.astype('float'))

# print("###################--Data Head--#######################\n")
# print(_raw_data_imp_cols.head())
# print("\n#######################################################\n")
# print("Outcome_Variable:" + user_input._output_col)
# print("\n###################--Column Types--####################\n")
# print(_raw_data_imp_cols.dtypes)
# print("#######################################################\n")

#########################################################################
# data visualization
#########################################################################

# print("################--Column Description--##################\n")
# print(_raw_data_imp_cols.describe(include = ["float"]))
# print(_raw_data_imp_cols.describe(include = ["int64"]))
# print(_raw_data_imp_cols.describe(include = ["category"]))
# print("########################################################")
#
# # print the categories of categorical variable
# print_categories(_raw_data_imp_cols, user_input._categorical_features + [user_input._output_col])
# # _raw_data_imp_cols.select_dtypes(include=['category']).apply(lambda x: print(pd.unique(x)))
#
# # save histogram plots of all categorical variables to data directory
# print("Saving Numerical Variable Histogram to Working Directory...")
# for col in (set(list(_raw_data_imp_cols)) - set(user_input._categorical_features)):
#     plt_hist(x= _raw_data_imp_cols[col], colname= col, n_bin= 20, dir_name= user_input._data_dir)
# print("Completed!\n")
# print("#######################################################\n")

#########################################################################
# test train split
#########################################################################
print("Splitting Test and Train Data...\n")
# test_train_splitter(df, y, cat_feature_list, int_feature_list, outcome_type = 'category', split_frac = 0.8)
X_train, y_train, X_test, y_test = test_train_splitter(df= _raw_data_imp_cols, y = user_input._output_col,
                                                       cat_feature_list= user_input._categorical_features,
                                                       int_feature_list= user_input._integer_features)
print("Train Data Length:" + str(len(X_train)))
print("\nTest Data Length:" + str(len(X_test)))
print("\n#######################################################\n")

#########################################################################

#########################################################################
# One hot encoding of categorical variables
#########################################################################

print("Performing One Hot Encoding of Categorical Variables...")

X_train_labelEncoded, X_test_labelEncoded = labelEncoder_cat_features(X_train = X_train, X_test = X_test,
                                            cat_feature_list= user_input._categorical_features)

X_train_oneHotEncoded, X_test_oneHotEncoded = oneHotEncoder_cat_features(X_train_labelEncoded= X_train_labelEncoded,
                                                                         X_test_labelEncoded= X_test_labelEncoded,
                                                                         cat_feature_list= user_input._categorical_features)

print("sample One Hot Encoded Data:\n")
print(X_train_oneHotEncoded.head())
print("\nColumn Types of One Hot Encoded Data:\n")
# print(X_test_oneHotEncoded.head())
print(X_train_oneHotEncoded.dtypes)
# print(X_test_oneHotEncoded.dtypes)
# print(len(X_train_oneHotEncoded))
# print(len(X_test_oneHotEncoded))
print("#######################################################\n")

# #########################################################################
# # model building
# #########################################################################

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 2)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 20, num = 2)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, scoring= 'f1')
# Fit the random search model
rf_random.fit(X_train_oneHotEncoded, y_train['Exited'])

# {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 10, 'bootstrap': False}

print(rf_random.best_params_)
print(rf_random.best_score_)
print(rf_random.cv_results_)
# print(rf_random.grid_scores_)

# rf = RandomForestClassifier(n_estimators= 100, min_samples_split= 2, min_samples_leaf= 4, max_features= 'sqrt',
#                             max_depth= 10, bootstrap= False)
#
# model_rf = rf.fit(X_train_oneHotEncoded, y_train['Exited'])
#
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#
# print(accuracy_score(y_test['Exited'],rf.predict(X_test_oneHotEncoded)))
#
# print(confusion_matrix(y_test['Exited'],rf.predict(X_test_oneHotEncoded)))
#
#
# print(classification_report(y_test['Exited'],rf.predict(X_test_oneHotEncoded)))



