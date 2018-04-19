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