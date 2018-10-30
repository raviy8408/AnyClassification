import numpy as np
import time

############################################################################
#                               User Input                                 #
############################################################################

_data_dir = "C://files/churn_test/data/"

_output_dir = "C://files/churn_test/output/" + "output_" + time.strftime("%Y%m%d-%H%M%S") + "/"

_input_file_name = "Churn_Modelling.csv"

_redundant_cols = ["RowNumber", "Surname"]

_ID_col = ["CustomerId"]

_categorical_features = ["Geography", "Gender", "HasCrCard", "IsActiveMember"]
_integer_features = ["CreditScore", "Age", "Tenure", "NumOfProducts"]

_output_col = "Exited"

# Available models: Logistic_Regression, Random_Forest
_model_list = ["Xgboost"] # "Logistic_Regression", "SVM_Linear", "SVM_Kernel", "Random_Forest", "Xgboost"

# Printing level set
verbose_high = False

################-- Logistic Regression Grid Search Parameters --############

# l1 and l2 regularization parameter
penalty = ['l1', 'l2']

# regularization parameter c = 1/lambda
C = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

################-- SVM Linear Grid Search Parameters --############

# regularization parameter c = 1/lambda
C_svm_linear = np.logspace(-3, 3, 1)

################-- SVM Kernel Grid Search Parameters --############

# regularization parameter c = 1/lambda
C_svm_kernel = np.logspace(-3, 3, 7)
gamma = np.logspace(-3, 3, 7)
kernel = ['rbf']

################-- Random Forest Grid Search Parameters --##################

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 240, num = 3)]
# Number of features to consider at every split
max_features = ['sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(9, 10, num = 2)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [8, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [2]
# Method of selecting samples for training each tree
bootstrap = [True] # [True, False]
# Different sampling options to treat imbalanced data
class_weight = ['balanced'] # ['balanced', 'balanced_subsample']
# class_weight.append(None)

################-- Xgboost Grid Search Parameters --########################

# no of boosted tree to form
XGB_n_estimators = [125, 150] # [int(x) for x in np.linspace(start = 100, stop = 200, num = 5)]
# minimum sum of weights of all observations required in a child
XGB_min_child_weight = [3, 4]
# gain at each leaf node should be more than gamma for the split to be made, no specific range depends on loss function
XGB_gamma = [0.025, 0.05] # 0.1,0.5,0.75, 1, 1.25, 1.5, 2, 5
# fraction of data to be sampled for each tree building step, smaller value would avoid over-fitting
XGB_subsample = [0.5, 0.6] # 0.5 - 1
# fraction of columns to be randomly samples for each tree
XGB_colsample_bytree = [0.5,0.6] # 0.5 -1
# Maximum number of levels in tree
XGB_max_depth = [4] #[int(x) for x in np.linspace(3, 6, num = 4)]
# learning rate makes model more robust by shrinking the weights on each step
XGB_learning_rate = [0.025] # default is 0.3, typical range 0.01 to 0.2
# a value greater than 0 should be used for high class imbalance
XGB_scale_pos_weight = [2] # default is 1
XGB_objective=['binary:logistic'] #'binary:logistic','reg:linear','reg:logistic','binary:hinge','binary:logitraw'
# default is zero, generally not required, a positive value would help making update step more conservative
XGB_max_delta_step=[0] # 0,1,2,4,6,8,10

# ################-- ANN Grid Search Parameters --########################
#
# NN_epochs = [1, 2]
# NN_batches = [100, 1000]
# NN_optimizers = ['rmsprop', 'adam']
# NN_activation = ['relu', 'sigmoid']
# NN_hidden_layers = [1, 2]

#############################-- Train Test Parameters --############################

# train test split fraction
train_test_split_frac = 0.8
# no of test train iterations to run the modeling with different test set, to ensure that model is not over fitted
train_test_iter = 3

#############################-- CV Parameters --############################

# Number of parameter settings that are sampled
n_iter = 12
# cross validation fold
cv = 5
# Integer value, higher the value more text is printed
verbose = 3
# model selection criteria
# choose from ‘accuracy’, ‘average_precision’,‘f1’, ‘f1_micro’, ‘f1_macro’, ‘f1_weighted’, ‘f1_samples’,‘neg_log_loss’,
# ‘precision’ etc., ‘recall’ etc., roc_auc’
scoring = 'roc_auc'

############################################################################






