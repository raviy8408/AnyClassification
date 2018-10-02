import pandas as pd
import numpy as np
import _user_input as user_input
from _plot_func import *
from mdlp.discretization import MDLP
from _support_func import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score
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

_raw_data_imp_cols[user_input._categorical_features + [user_input._output_col]] = _raw_data_imp_cols[
    user_input._categorical_features + [user_input._output_col]] \
    .apply(lambda x: x.astype('category'))

_raw_data_imp_cols[user_input._integer_features] = _raw_data_imp_cols[user_input._integer_features] \
    .apply(lambda x: x.astype('int64'))

_non_float_features = user_input._categorical_features + user_input._integer_features + [user_input._output_col]

_raw_data_imp_cols[_raw_data_imp_cols.columns.difference(_non_float_features)] = _raw_data_imp_cols[
    _raw_data_imp_cols.columns.difference(_non_float_features)] \
    .apply(lambda x: x.astype('float'))

print("###################--Data Head--#######################\n")
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
print(_raw_data_imp_cols.describe(include=["int64"]))
print(_raw_data_imp_cols.describe(include=["category"]))
print("########################################################")

# print the categories of categorical variable
print_categories(_raw_data_imp_cols, user_input._categorical_features + [user_input._output_col])
# _raw_data_imp_cols.select_dtypes(include=['category']).apply(lambda x: print(pd.unique(x)))

# save histogram plots of all categorical variables to data directory
print("########################--Variable EDA--#################################\n")

eda_plots(data=_raw_data_imp_cols, cat_feature_list=user_input._categorical_features,
          outcome_col=user_input._output_col, output_dir=user_input._output_dir)

print("#########################################################################\n")

#########################################################################
# test train split
#########################################################################
print("Splitting Test and Train Data...\n")
# test_train_splitter(df, y, cat_feature_list, int_feature_list, outcome_type = 'category', split_frac = 0.8)
X_train, y_train, X_test, y_test = test_train_splitter(df=_raw_data_imp_cols, y=user_input._output_col,
                                                       cat_feature_list=user_input._categorical_features,
                                                       int_feature_list=user_input._integer_features)
print("Train Data Length:" + str(len(X_train)))
print("\nTest Data Length:" + str(len(X_test)))
print("\n#######################################################\n")

#########################################################################

# ###############################################################
#                        model building                         #
# ###############################################################

available_model_list = ["Logistic_Regression", "Random_Forest"]

# Loop for all the models provided in user input
for model in user_input._model_list:

    print("#######################################################\n")
    print("                 **" + model + "**                  \n")
    print("#######################################################\n")

    if model in available_model_list:
        if model == "Logistic_Regression":
            # in one hot encoding drop the last dummy variable column to avoid multi-collinearity
            drop_last_col = True
        else:
            drop_last_col = False
    else:
        print(model + " is not present in existing model list!\n")
        continue

    #########################################################################
    # One hot encoding of categorical variables
    #########################################################################
    print("Performing One Hot Encoding of Categorical Variables...\n")

    X_train_labelEncoded, X_test_labelEncoded = labelEncoder_cat_features(X_train=X_train, X_test=X_test,
                                                                          cat_feature_list=user_input._categorical_features)

    X_train_oneHotEncoded, X_test_oneHotEncoded = oneHotEncoder_cat_features(X_train_labelEncoded=X_train_labelEncoded,
                                                                             X_test_labelEncoded=X_test_labelEncoded,
                                                                             cat_feature_list=user_input._categorical_features,
                                                                             drop_last=drop_last_col)

    print("Sample Model Input Data:\n")
    print(X_train_oneHotEncoded.head())
    print("\nColumn Types of Final Data:")
    # print(X_test_oneHotEncoded.head())
    print(X_train_oneHotEncoded.dtypes)
    # print(X_test_oneHotEncoded.dtypes)
    # print(len(X_train_oneHotEncoded))
    # print(len(X_test_oneHotEncoded))

    print("#######################################################\n")

    #######################--Logistic Regression--###################

    if model == "Logistic_Regression":
        from sklearn.linear_model import LogisticRegression

        logreg = LogisticRegression()
        logreg.fit(X_train_oneHotEncoded, y_train[user_input._output_col])

        # print(rf_random.cv_results_)
        # print(rf_random.grid_scores_)

        print("#################--Model Performance--#################\n")

        path = user_input._output_dir + "Model_Result/" + "Logistic_Regression/"
        if not os.path.isdir(path):
            os.makedirs(path)

        # Saving ROC plot to the drive
        plot_ROC(y_test = y_test[user_input._output_col],
                 y_pred_prob= logreg.predict_proba(X_test_oneHotEncoded)[:, 1], model_name= 'Logistic Regression',
                 image_dir= path)
        print("ROC plot saved to the drive!\n")

        print("Model Performance on Test Set:\n")
        print("Accuracy:\n")
        print(str(accuracy_score(y_test[user_input._output_col], logreg.predict(X_test_oneHotEncoded))))
        print("\nConfusion Matrix:\n")
        # print(confusion_matrix(y_test[user_input._output_col],rf_random.best_estimator_.predict(X_test_oneHotEncoded)))
        print(pd.crosstab(y_test[user_input._output_col], logreg.predict(X_test_oneHotEncoded),
                          rownames=['True'], colnames=['Predicted'], margins=True))
        print("\nClassification Report:\n")
        print(classification_report(y_test[user_input._output_col], logreg.predict(X_test_oneHotEncoded)))
        print("\nCohen Kappa:\n")
        print(cohen_kappa_score(y_test[user_input._output_col], logreg.predict(X_test_oneHotEncoded)))

        print("#######################################################\n")


    #########################--Random Forest--#######################

    elif model == "Random_Forest":

        random_grid = {'n_estimators': user_input.n_estimators,
                       'max_features': user_input.max_features,
                       'max_depth': user_input.max_depth,
                       'min_samples_split': user_input.min_samples_split,
                       'min_samples_leaf': user_input.min_samples_leaf,
                       'bootstrap': user_input.bootstrap,
                       'class_weight': user_input.class_weight}

        # print(random_grid)

        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestClassifier()
        # Random search of parameters, using n fold cross validation,
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                       n_iter=user_input.n_iter, cv=user_input.cv, verbose=user_input.verbose,
                                       random_state=42, scoring=user_input.scoring)

        # Fit the random search model
        print(str(user_input.cv) + "-Fold CV in Progress...")
        rf_random.fit(X_train_oneHotEncoded, y_train[user_input._output_col])

        print("###################--CV Result--########################\n")

        print("Best " + user_input.scoring + " Score Obtained:")
        print(rf_random.best_score_)
        print("Best Model Parameter Set for Highest " + user_input.scoring + ":\n")
        print(rf_random.best_params_)

        print("\n#########################################################\n")

        print("#################--Model Performance--###################\n")

        # print(rf_random.cv_results_)
        # print(rf_random.grid_scores_)

        path = user_input._output_dir + "Model_Result/" + "Random_Forest/"
        if not os.path.isdir(path):
            os.makedirs(path)

        # Saving ROC plot to the drive
        plot_ROC(y_test = y_test[user_input._output_col],
                 y_pred_prob= rf_random.best_estimator_.predict_proba(X_test_oneHotEncoded)[:, 1],
                 model_name= 'Random Forest',image_dir= path)
        print("ROC plot saved to the drive!\n")

        print("Model Performance on Test Set:\n")
        print("Accuracy:\n")
        print(str(accuracy_score(y_test[user_input._output_col], rf_random.best_estimator_.predict(X_test_oneHotEncoded))))
        print("\nConfusion Matrix:\n")
        # print(confusion_matrix(y_test[user_input._output_col],rf_random.best_estimator_.predict(X_test_oneHotEncoded)))
        print(pd.crosstab(y_test[user_input._output_col], rf_random.best_estimator_.predict(X_test_oneHotEncoded),
                          rownames=['True'], colnames=['Predicted'], margins=True))
        print("\nClassification Report:\n")
        print(classification_report(y_test[user_input._output_col], rf_random.best_estimator_.predict(X_test_oneHotEncoded)))
        print("\nCohen Kappa:\n")
        print(cohen_kappa_score(y_test[user_input._output_col], rf_random.best_estimator_.predict(X_test_oneHotEncoded)))

        print("#######################################################\n")


