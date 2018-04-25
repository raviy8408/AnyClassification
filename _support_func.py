import numpy as np
import pandas as pd

def print_categories(df,cols):
    '''
    :param df: Pandas DataFrame
    :param cols: Categorical columns
    :return: prints all the categories of the categorical columns given
    '''
    print("\n##########--levels of categorical variable--##########")
    for col in cols:
        print("\n" + col + "categories:")
        print((df[col].cat.categories))
    print("\n######################################################")


def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]

def test_train_splitter(df, y, cat_feature_list,  outcome_type = "category", split_frac = 0.8):
    '''
    Splits the data into test and train in the ration provided and returns 4 data frames:
    x_train, y_train, x_test, y_test
    :param df: complete data
    :param y: outcome variable
    :param cat_feature_list: all the categorical variables
    :return: x_train, y_train, x_test, y_test
    '''
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(df.drop([y], axis=1).as_matrix(),
                                                        df[y].as_matrix(), test_size=0.2)

    # test train dataframe creation
    X_train_df = pd.DataFrame(data=X_train, columns= df.drop([y], axis=1).columns.values)
    y_train_df = pd.DataFrame(data=y_train, columns=[y])

    X_test_df = pd.DataFrame(data=X_test,columns= df.drop([y], axis=1).columns.values)
    y_test_df = pd.DataFrame(data=y_test, columns=[y])

    # assigning variable types
    X_train_df[cat_feature_list] = X_train_df[cat_feature_list].apply(lambda x: x.astype('category'))
    X_test_df[cat_feature_list] = X_test_df[cat_feature_list].apply(lambda x: x.astype('category'))

    X_train_df[df.columns.difference(cat_feature_list)] = X_train_df[df.columns.difference(cat_feature_list)].apply(lambda x: x.astype('float'))
    X_test_df[df.columns.difference(cat_feature_list)] = X_test_df[df.columns.difference(cat_feature_list)].apply(lambda x: x.astype('float'))

    if outcome_type == "category":
        y_train[y] = y_train[y].apply(lambda x: x.astype('category'))
        y_test[y] = y_test[y].apply(lambda x: x.astype('category'))

    elif outcome_type == "float":
        y_train[y] = y_train[y].apply(lambda x: x.astype('float'))
        y_test[y] = y_test[y].apply(lambda x: x.astype('float'))

    return




