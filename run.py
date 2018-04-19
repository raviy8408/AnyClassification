import pandas as pd
import numpy as np
import _user_input as user_input
from _plot_func import *
from mdlp.discretization import MDLP
from _support_func import *


#########################################################################
# data load
#########################################################################

_raw_data = pd.read_csv(user_input._data_dir + user_input._input_file_name, header=0)

# print(_raw_data.head())

_raw_data_imp_cols = _raw_data.drop(user_input._redundant_cols, axis=1)

# print(_raw_data_imp_cols.head())
# print(_raw_data_imp_cols.dtypes)

_raw_data_imp_cols[user_input._categorical_features] = _raw_data_imp_cols[user_input._categorical_features]\
    .apply(lambda x: x.astype('category'))

_raw_data_imp_cols[_raw_data_imp_cols.columns.difference(user_input._categorical_features)] = _raw_data_imp_cols[
    _raw_data_imp_cols.columns.difference(user_input._categorical_features)]\
    .apply(lambda x: x.astype('float'))

# print(_raw_data_imp_cols.head())
# print(_raw_data_imp_cols.dtypes)


#########################################################################
# data visualization
#########################################################################

# print(_raw_data_imp_cols.describe(include = ["float"]))
# print(_raw_data_imp_cols.describe(include = ["category"]))

# # print the categories of categorical variable
# print_categories(_raw_data_imp_cols, user_input._categorical_features)
# # _raw_data_imp_cols.select_dtypes(include=['category']).apply(lambda x: print(pd.unique(x)))

# # save histogram plots of all categorical variables to data directory
# for col in (set(list(_raw_data_imp_cols)) - set(user_input._categorical_features)):
#     plt_hist(x= _raw_data_imp_cols[col], colname= col, n_bin= 20, dir_name= user_input._data_dir)

#########################################################################
# data transform
#########################################################################

_temp_cont_var = _raw_data_imp_cols[_raw_data_imp_cols.columns.difference(user_input._categorical_features)]
_temp_cont_var_ndarray = _temp_cont_var.as_matrix()
_temp_target_ndarray = _raw_data_imp_cols["Exited"].as_matrix()

# print(_temp_cont_var_ndarray)
# print(_temp_target_ndarray)

transformer = MDLP()
cont_var_transformed_ndarray = transformer.fit_transform(_temp_cont_var_ndarray,
                                   _temp_target_ndarray)

_cont_var_transformed = pd.DataFrame(data=cont_var_transformed_ndarray,
                                     columns= _raw_data_imp_cols.columns.difference(user_input._categorical_features)
                                     .tolist())
_cont_var_transformed = _cont_var_transformed.loc[:, (_cont_var_transformed != 0).any(axis=0)]
_cont_var_transformed = _cont_var_transformed.apply(lambda x: x.astype("category"))

# print(_cont_var_transformed.head())

_binned_data = pd.concat([_cont_var_transformed.reset_index(drop=True),
                          _raw_data_imp_cols[user_input._categorical_features]], axis=1)


# print(_binned_data.head())
# print(_binned_data.dtypes)
# print_categories(_binned_data, list(_binned_data))

#########################################################################
# model building
#########################################################################




