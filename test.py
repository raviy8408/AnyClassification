# from mdlp.discretization import MDLP
# from sklearn.datasets import load_iris
# transformer = MDLP()
# iris = load_iris()
#
# print(type(iris))
# X, y = iris.data, iris.target
#
# print(type(X))
#
# print(X)
# print(type(y))
# print(y)
# X_disc = transformer.fit_transform(X, y)

# print(X_disc)

# import numpy as np
#
# X_t = [[1,2,3],[4,5,6]]
# X = np.array([[1,2,3], [4,5,6]])
#
# print(type(X))
#
# print(X)
#
# np.random.shuffle(X[:, 2])
#
# print(X)

# a = [15, 2, 3]
# a.remove(15)
#
# print(a)

# #########################################################################
# # mld transform
# #########################################################################
# # print(X_train)
#
# # mdlp_transformer = MDLP(continuous_features= _continuous_var_index)
# #
# # mdlp_fit = mdlp_transformer.fit_transform(X_train,
# #                                 y_train,)
# #
# # print(mdlp_fit)
# # _temp_cont_var = _raw_data_imp_cols[_raw_data_imp_cols.columns.difference(user_input._categorical_features)]
# # _temp_cont_var_ndarray = _temp_cont_var.as_matrix()
# # _temp_target_ndarray = _raw_data_imp_cols["Exited"].as_matrix()
# #
# # # print(_temp_cont_var_ndarray)
# # # print(_temp_target_ndarray)
# #
# # transformer = MDLP()
# # cont_var_transformed_ndarray = transformer.fit_transform(_temp_cont_var_ndarray,
# #                                    _temp_target_ndarray)
# #
# # _cont_var_transformed = pd.DataFrame(data=cont_var_transformed_ndarray,
# #                                      columns= _raw_data_imp_cols.columns.difference(user_input._categorical_features)
# #                                      .tolist())
# # _cont_var_transformed = _cont_var_transformed.loc[:, (_cont_var_transformed != 0).any(axis=0)]
# # _cont_var_transformed = _cont_var_transformed.apply(lambda x: x.astype("category"))
# #
# # # print(_cont_var_transformed.head())
# #
# # _binned_data = pd.concat([_cont_var_transformed.reset_index(drop=True),
# #                           _raw_data_imp_cols[user_input._categorical_features]], axis=1)
#
#
# # print(_binned_data.head())
# # print(_binned_data.dtypes)
# # print_categories(_binned_data, list(_binned_data))
#
#
# import pandas as pd
#
# df = pd.DataFrame(data={'a': [1,2,3],'b': [5,6,7]})
#
# print(df.dtypes)
#
# l = []
#
# if l:
#     print("array is not empty")
# else:
#     print("array is empty")

import numpy as np

C_range = np.logspace(-3, 3, 7)
gamma_range = np.logspace(-9, 3, 13)

print(C_range)
print(gamma_range)