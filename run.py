import pandas as pd
import numpy as np
import _user_input as input


#########################################################################
# data load
#########################################################################

_raw_data = pd.read_csv(input._data_dir + input._input_file_name, header=0)

# print(_raw_data.head())

_raw_data_imp_cols = _raw_data.drop(input._redundant_cols, axis=1)

# print(_raw_data_imp_cols.head())
# print(_raw_data_imp_cols.dtypes)

_raw_data_imp_cols[input._categorical_features] = _raw_data_imp_cols[input._categorical_features]\
    .apply(lambda x: x.astype('category'))

_raw_data_imp_cols[_raw_data_imp_cols.columns.difference(input._categorical_features)] = _raw_data_imp_cols[
    _raw_data_imp_cols.columns.difference(input._categorical_features)]\
    .apply(lambda x: x.astype('float'))

# print(_raw_data_imp_cols.head())
# print(_raw_data_imp_cols.dtypes)
print(_raw_data_imp_cols.describe(include = ["float"]))
print(_raw_data_imp_cols.describe(include = ["category"]))

# print(pd.DataFrame(_raw_data_imp_cols.cat.categories))




