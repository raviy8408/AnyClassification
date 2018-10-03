import numpy as np

############################################################################
#                               User Input                                 #
############################################################################

_data_dir = "C://files/churn_test/data/"

_output_dir = "C://files/churn_test/output/"

_input_file_name = "Churn_Modelling.csv"

_redundant_cols = ["RowNumber", "CustomerId", "Surname"]

_categorical_features = ["Geography", "Gender", "HasCrCard", "IsActiveMember"]
_integer_features = ["CreditScore", "Age", "Tenure", "NumOfProducts"]

_output_col = "Exited"

# Available models: Logistic_Regression, Random_Forest
_model_list = ["Logistic_Regression", "Random_Forest"]

################-- Logistic Regression Grid Search Parameters --############

# l1 and l2 regularization parameter
penalty = ['l1', 'l2']

# regularization parameter c = 1/lambda
C = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

################-- Random Forest Grid Search Parameters --##################

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 200, num = 50)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(4, 20, num = 15)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 20]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 6]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Different sampling options to treat imbalanced data
class_weight = ['balanced', 'balanced_subsample']
class_weight.append(None)

#############################-- CV Parameters --############################

# Number of parameter settings that are sampled
n_iter = 50
# cross validation fold
cv = 5
# Integer value, higher the value more text is printed
verbose=1
# model selection criteria
# choose from ‘accuracy’, ‘average_precision’,‘f1’, ‘f1_micro’, ‘f1_macro’, ‘f1_weighted’, ‘f1_samples’,‘neg_log_loss’,
# ‘precision’ etc., ‘recall’ etc., roc_auc’
scoring= 'f1_weighted'

############################################################################



