import pandas as pd
import numpy as np
import _user_input as user_input
from _plot_func import *
from _models import *
from _support_func import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import os
import shutil

#########################################################################
# data load
#########################################################################

_raw_data = pd.read_csv(user_input._data_dir + user_input._input_file_name, header=0)

# print(_raw_data.head())

_raw_data_imp_cols = _raw_data.drop(user_input._redundant_cols, axis=1)

# print(_raw_data_imp_cols.head())
# print(_raw_data_imp_cols.dtypes)

if user_input._categorical_features:
    _raw_data_imp_cols[user_input._categorical_features + [user_input._output_col]] = _raw_data_imp_cols[
        user_input._categorical_features + [user_input._output_col]] \
        .apply(lambda x: x.astype('category'))

if user_input._integer_features:
    _raw_data_imp_cols[user_input._integer_features] = _raw_data_imp_cols[user_input._integer_features] \
        .apply(lambda x: x.astype('int64'))

_non_float_features = user_input._categorical_features + user_input._integer_features + [user_input._output_col]

_numeric_features = user_input._integer_features + _raw_data_imp_cols.columns.difference(_non_float_features).values.tolist()

_raw_data_imp_cols[_raw_data_imp_cols.columns.difference(_non_float_features)] = _raw_data_imp_cols[
    _raw_data_imp_cols.columns.difference(_non_float_features)] \
    .apply(lambda x: x.astype('float'))

print("###################--Data Head--#########################\n")
print(_raw_data_imp_cols.head())
print("\n#######################################################\n")
print("Outcome_Variable:" + user_input._output_col + "\n")
print("Outcome Class Distribution:\n")
print(_raw_data_imp_cols[user_input._output_col].value_counts())
print("\n###################--Column Types--####################\n")
print(_raw_data_imp_cols.dtypes)
print("#######################################################\n")

#########################################################################
# data visualization
#########################################################################

print("################--Column Description--##################\n")
print(_raw_data_imp_cols.describe(include=["float"]))
if user_input._integer_features:
    print(_raw_data_imp_cols.describe(include=["int64"]))
if user_input._categorical_features:
    print(_raw_data_imp_cols.describe(include=["category"]))
print("########################################################\n")

# print the categories of categorical variable
if user_input._categorical_features:
    print_categories(_raw_data_imp_cols, user_input._categorical_features + [user_input._output_col])
# _raw_data_imp_cols.select_dtypes(include=['category']).apply(lambda x: print(pd.unique(x)))

# save histogram plots of all categorical variables to data directory
print("########################--Variable EDA--#################\n")

eda_plots(data=_raw_data_imp_cols, cat_feature_list=user_input._categorical_features,
          outcome_col=user_input._output_col, output_dir=user_input._output_dir)

print("#########################################################\n")

########################################################################
# Fit the Scalar on the data
########################################################################
scalar = StandardScaler().fit(_raw_data_imp_cols[_numeric_features])

#########################################################################
# test train split
#########################################################################
print("Splitting Test and Train Data...\n")
# test_train_splitter(df, y, cat_feature_list, int_feature_list, outcome_type = 'category', split_frac = 0.8)
X_train, y_train, X_test, y_test = test_train_splitter(df= _raw_data_imp_cols, y=user_input._output_col,
                                                       cat_feature_list=user_input._categorical_features,
                                                       int_feature_list=user_input._integer_features)
print("Train Data Length:" + str(len(X_train)))
print("\nTest Data Length:" + str(len(X_test)))
print("\n#######################################################\n")

#########################################################################

# ###############################################################
#                        model building                         #
# ###############################################################
available_model_list = ["Logistic_Regression", "svm_linear", "svm_kernel",  "Random_Forest", "Xgboost", "ANN"]

# Loop for all the models provided in user input
for model in user_input._model_list:

    print("#######################################################\n")
    print("                 **" + model + "**                  \n")
    print("#######################################################\n")

    if model in available_model_list:
        if model in ["Logistic_Regression", "svm_linear", "svm_kernel"]:
            # in one hot encoding drop the last dummy variable column to avoid multi-collinearity
            X_train[_numeric_features] = scalar.transform(X_train[_numeric_features])
            X_test[_numeric_features] = scalar.transform(X_test[_numeric_features])
            drop_last_col = True
        else:
            drop_last_col = False
    else:
        print(model + " is not present in existing model list!\n")
        continue

    #########################################################################
    # One hot encoding of categorical variables
    #########################################################################
    if user_input._categorical_features:
        print("Performing One Hot Encoding of Categorical Variables...\n")

        X_train_labelEncoded, X_test_labelEncoded = labelEncoder_cat_features(X_train=X_train, X_test=X_test,
                                                                              cat_feature_list=user_input._categorical_features)

        X_train_oneHotEncoded, X_test_oneHotEncoded = oneHotEncoder_cat_features(X_train_labelEncoded=X_train_labelEncoded,
                                                                                 X_test_labelEncoded=X_test_labelEncoded,
                                                                                 cat_feature_list=user_input._categorical_features,
                                                                                 drop_last=drop_last_col)
        # assigning final train and test X data if one hot encoding is done
        X_train_model_dt = X_train_oneHotEncoded
        X_test_model_dt = X_test_oneHotEncoded


    else:
        # assigning final train and test X data if one hot encoding is not done
        X_train_model_dt = X_train
        X_test_model_dt = X_test

    if user_input.verbose_high == True:
        print("Sample Model Input Data:\n")
        print(X_train_model_dt.head())
        print("\nColumn Types of Final Data:")
        print(X_train_model_dt.dtypes)

        print("#######################################################\n")

    #######################--Logistic Regression--##################

    if model == "Logistic_Regression":

        Logistic_Regresion(X_train_model_dt = X_train_model_dt, y_train = y_train, X_test_model_dt = X_test_model_dt,
                           y_test = y_test)

    ############################--SVM Linear--######################

    elif model == "svm_linear":

        SVM_Linear(X_train_model_dt=X_train_model_dt, y_train=y_train, X_test_model_dt=X_test_model_dt,
                   y_test=y_test)


    ############################--SVM Kernel--######################

    elif model == "svm_kernel":

        SVM_Kernel(X_train_model_dt=X_train_model_dt, y_train=y_train, X_test_model_dt=X_test_model_dt,
                           y_test=y_test)


    #########################--Random Forest--#####################

    elif model == "Random_Forest":

        Random_Forest(X_train_model_dt=X_train_model_dt, y_train=y_train, X_test_model_dt=X_test_model_dt,
                           y_test=y_test)


    #########################-- XGBoost--#####################

    elif model == "Xgboost":

        Xgboost(X_train_model_dt=X_train_model_dt, y_train=y_train, X_test_model_dt=X_test_model_dt,
                           y_test=y_test)

    # #########################--ANN --#####################
    #
    # elif model == "ANN":
    #
    #     ANN(X_train_model_dt=X_train_model_dt, y_train=y_train, X_test_model_dt=X_test_model_dt,
    #                        y_test=y_test)


