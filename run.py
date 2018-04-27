import pandas as pd
import numpy as np
import _user_input as user_input
from _plot_func import *
from mdlp.discretization import MDLP
from _support_func import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


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

print("###################--Data Head--#######################\n")
print(_raw_data_imp_cols.head())
print("\n#######################################################\n")
print("Outcome_Variable:" + user_input._output_col)
print("\n###################--Column Types--####################\n")
print(_raw_data_imp_cols.dtypes)
print("#######################################################\n")

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
# print("Saving Plots to Working Directory...")
# for col in (set(list(_raw_data_imp_cols)) - set(user_input._categorical_features)):
#     plt_hist(x= _raw_data_imp_cols[col], colname= col, n_bin= 20, dir_name= user_input._data_dir)
# print("Completed!\n")
# print("#######################################################")

#########################################################################
# test train split
#########################################################################

# test_train_splitter(df, y, cat_feature_list, int_feature_list, outcome_type = 'category', split_frac = 0.8)
X_train, X_test, y_train, y_test = test_train_splitter(df= _raw_data_imp_cols, y = user_input._output_col,
                                                       cat_feature_list= user_input._categorical_features,
                                                       int_feature_list= user_input._integer_features)

#########################################################################

#########################################################################
# One hot encoding of categorical variables
#########################################################################

# _one_hot_encode = OneHotEncoder(categorical_features= [user_input._categorical_features])

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
transfomed_label = encoder.fit_transform(X_train)
print(transfomed_label)

#########################################################################
# mld transform
#########################################################################
# print(X_train)

# mdlp_transformer = MDLP(continuous_features= _continuous_var_index)
#
# mdlp_fit = mdlp_transformer.fit_transform(X_train,
#                                 y_train,)
#
# print(mdlp_fit)
# _temp_cont_var = _raw_data_imp_cols[_raw_data_imp_cols.columns.difference(user_input._categorical_features)]
# _temp_cont_var_ndarray = _temp_cont_var.as_matrix()
# _temp_target_ndarray = _raw_data_imp_cols["Exited"].as_matrix()
#
# # print(_temp_cont_var_ndarray)
# # print(_temp_target_ndarray)
#
# transformer = MDLP()
# cont_var_transformed_ndarray = transformer.fit_transform(_temp_cont_var_ndarray,
#                                    _temp_target_ndarray)
#
# _cont_var_transformed = pd.DataFrame(data=cont_var_transformed_ndarray,
#                                      columns= _raw_data_imp_cols.columns.difference(user_input._categorical_features)
#                                      .tolist())
# _cont_var_transformed = _cont_var_transformed.loc[:, (_cont_var_transformed != 0).any(axis=0)]
# _cont_var_transformed = _cont_var_transformed.apply(lambda x: x.astype("category"))
#
# # print(_cont_var_transformed.head())
#
# _binned_data = pd.concat([_cont_var_transformed.reset_index(drop=True),
#                           _raw_data_imp_cols[user_input._categorical_features]], axis=1)


# print(_binned_data.head())
# print(_binned_data.dtypes)
# print_categories(_binned_data, list(_binned_data))

#########################################################################
# model building
#########################################################################




