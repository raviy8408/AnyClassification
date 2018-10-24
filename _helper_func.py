import pandas as pd
import numpy as np

def appendDFToCSV_void(df, csvFilePath, sep=","):
    import os
    if not os.path.isfile(csvFilePath):
        df.to_csv(csvFilePath, mode='a', index=True, sep=sep)
    elif len(df.columns) != len(pd.read_csv(csvFilePath, nrows=1, sep=sep, index_col=0).columns):
        raise Exception("Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns)) + " columns.")
    elif not (df.columns == pd.read_csv(csvFilePath, nrows=1, sep=sep, index_col=0).columns).all():
        raise Exception("Columns and column order of dataframe and csv file do not match!!")
    else:
        df.to_csv(csvFilePath, mode='a', index=True, sep=sep, header=False)

def result_prep(path, train_test_iter_count):

    result = pd.read_csv(path, sep="\t", index_col=0)

    result_final = result.T
    result_final['avg'] = result_final.iloc[:, 0:train_test_iter_count].mean(axis=1)
    result_final['std'] = result_final.iloc[:, 0:train_test_iter_count].std(axis=1)

    print(result_final)

    result_final.to_csv(path, sep="\t")

def feature_imp_prep(path, train_test_iter_count):

    importance = pd.read_csv(path, sep="\t", index_col=0)

    importance_final = importance.T
    importance_final['avg'] = importance_final.iloc[:, 0:train_test_iter_count].mean(axis=1)
    importance_final['svd'] = importance_final.iloc[:, 0:train_test_iter_count].std(axis=1)

    importance_final = importance_final.sort_values('avg', ascending=False)

    print(importance_final.head(10))

    importance_final.to_csv(path, sep="\t")