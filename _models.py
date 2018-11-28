from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score
import _user_input as user_input
from _plot_func import *
from _support_func import *
import pandas as pd
import numpy as np


def Logistic_Regression(X_train_model_dt, y_train, X_test_model_dt, y_test, **kwargs):
    from sklearn.linear_model import LogisticRegression

    # get the tran test iteration number to store the results per iter
    if ('train_test_iter_num' in kwargs.keys()):
        train_test_iter_num = kwargs.get("train_test_iter_num")
    else:
        train_test_iter_num = 1

    if ('train_ID' in kwargs.keys()):
        train_ID = kwargs.get("train_ID")
    else:
        train_ID = []

    if ('test_ID' in kwargs.keys()):
        test_ID = kwargs.get("test_ID")
    else:
        test_ID = []

    random_grid = {'penalty': user_input.penalty,
                   'C': user_input.C}

    # length of exhaustive set of parameter combination
    max_n_iter = 1
    for key, value in random_grid.items():
        max_n_iter *= len(value)

    logreg = LogisticRegression(class_weight='balanced')

    lr_random = RandomizedSearchCV(estimator=logreg, param_distributions=random_grid,
                                   n_iter=min(user_input.n_iter, max_n_iter),
                                   cv=user_input.cv, verbose=user_input.verbose,
                                   random_state=42, scoring=user_input.scoring)

    # Fit the random search model
    print(str(user_input.cv) + "-Fold CV in Progress...")
    lr_random.fit(X_train_model_dt, y_train[user_input._output_col])

    print("###################--CV Result--########################\n")

    print("Best " + user_input.scoring + " Score Obtained:")
    print(lr_random.best_score_)
    print("\nBest Model Parameter Set for Highest " + user_input.scoring + ":")
    print(lr_random.best_params_)

    #####################################
    print("\nSaving param to drive:")
    path = user_input._output_dir + "Model_Result/" + "Logistic_Regression/"
    if not os.path.isdir(path):
        os.makedirs(path)
    _param_dict = lr_random.best_params_
    _param_dict[user_input.scoring] = lr_random.best_score_
    _param = pd.DataFrame(_param_dict, index=['iter' + str(train_test_iter_num)])
    appendDFToCSV_void(df=_param, csvFilePath=path + 'cv_param.tsv', sep='\t')
    #####################################

    print("\n P values for variables:\n")
    try:
        cal_lr_p_vals(X=X_train_model_dt, y=y_train[user_input._output_col],
                      params=np.append(lr_random.best_estimator_.intercept_, lr_random.best_estimator_.coef_),
                      predictions=lr_random.best_estimator_.predict(X_train_model_dt))
    except:
        print("P-value calculation failed!")
    print("\n#####################################################\n")

    model_performance(X_test_model_dt=X_train_model_dt, y_test=y_train[user_input._output_col],
                      model_name="Logistic_Regression", model_object=lr_random,
                      output_path=user_input._output_dir + "Model_Result/", prob=True,
                      train_test_iter_num=train_test_iter_num, ID=train_ID, train_set=True)

    model_performance(X_test_model_dt=X_test_model_dt, y_test=y_test[user_input._output_col],
                      model_name="Logistic_Regression", model_object=lr_random,
                      output_path=user_input._output_dir + "Model_Result/", prob=True,
                      train_test_iter_num=train_test_iter_num, ID= test_ID, train_set= False)

    print("#######################################################\n")


def SVM_Linear(X_train_model_dt, y_train, X_test_model_dt, y_test, **kwargs):

    from sklearn.svm import SVC

    # get the tran test iteration number to store the results per iter
    if ('train_test_iter_num' in kwargs.keys()):
        train_test_iter_num = kwargs.get("train_test_iter_num")
    else:
        train_test_iter_num = 1

    if ('train_ID' in kwargs.keys()):
        train_ID = kwargs.get("train_ID")
    else:
        train_ID = []

    if ('test_ID' in kwargs.keys()):
        test_ID = kwargs.get("test_ID")
    else:
        test_ID = []

    # print("Running linear SVM..\n")

    random_grid = {'C': user_input.C_svm_linear, 'kernel': ['linear']}

    # length of exhaustive set of parameter combination
    max_n_iter = 1
    for key, value in random_grid.items():
        max_n_iter *= len(value)

    svc_linear = SVC(class_weight='balanced')

    svc_linear_random = RandomizedSearchCV(estimator=svc_linear, param_distributions=random_grid,
                                           n_iter=min(user_input.n_iter, max_n_iter),
                                           cv=user_input.cv, verbose=user_input.verbose,
                                           random_state=42, scoring=user_input.scoring)

    # Fit the random search model
    print(str(user_input.cv) + "-Fold CV in Progress...")
    svc_linear_random.fit(X_train_model_dt, y_train[user_input._output_col])

    print("###################--CV Result--########################\n")

    print("Best " + user_input.scoring + " Score Obtained:")
    print(svc_linear_random.best_score_)
    print("\nBest Model Parameter Set for Highest " + user_input.scoring + ":")
    print(svc_linear_random.best_params_)

    ######################################
    print("\nSaving param to drive:")
    path = user_input._output_dir + "Model_Result/" + "SVM_Linear/"
    if not os.path.isdir(path):
        os.makedirs(path)
    _param_dict = svc_linear_random.best_params_
    _param_dict[user_input.scoring] = svc_linear_random.best_score_
    _param = pd.DataFrame(_param_dict, index=['iter' + str(train_test_iter_num)])
    appendDFToCSV_void(df=_param, csvFilePath=path + 'cv_param.tsv', sep='\t')
    ######################################

    print("\n######################################################\n")

    model_performance(X_test_model_dt=X_train_model_dt, y_test=y_train[user_input._output_col],
                      model_name="SVM_Linear", model_object=svc_linear_random,
                      output_path=user_input._output_dir + "Model_Result/", prob=False,
                      train_test_iter_num=train_test_iter_num, ID=train_ID, train_set=True)

    model_performance(X_test_model_dt=X_test_model_dt, y_test=y_test[user_input._output_col],
                      model_name="SVM_Linear", model_object=svc_linear_random,
                      output_path=user_input._output_dir + "Model_Result/", prob=False,
                      train_test_iter_num=train_test_iter_num, ID= test_ID, train_set= False)

    print("#######################################################\n")


def SVM_Kernel(X_train_model_dt, y_train, X_test_model_dt, y_test, **kwargs):
    from sklearn.svm import SVC

    # get the tran test iteration number to store the results per iter
    if ('train_test_iter_num' in kwargs.keys()):
        train_test_iter_num = kwargs.get("train_test_iter_num")
    else:
        train_test_iter_num = 1

    if ('train_ID' in kwargs.keys()):
        train_ID = kwargs.get("train_ID")
    else:
        train_ID = []

    if ('test_ID' in kwargs.keys()):
        test_ID = kwargs.get("test_ID")
    else:
        test_ID = []

    # print("Running SVM Kernel..\n")

    random_grid = {'C': user_input.C_svm_kernel, 'gamma': user_input.gamma, 'kernel': user_input.kernel}

    # length of exhaustive set of parameter combination
    max_n_iter = 1
    for key, value in random_grid.items():
        max_n_iter *= len(value)

    svc_kernel = SVC(class_weight='balanced')

    svc_kernel_random = RandomizedSearchCV(estimator=svc_kernel, param_distributions=random_grid,
                                           n_iter=min(user_input.n_iter, max_n_iter),
                                           cv=user_input.cv, verbose=user_input.verbose,
                                           random_state=42, scoring=user_input.scoring)

    # Fit the random search model
    print(str(user_input.cv) + "-Fold CV in Progress...")
    svc_kernel_random.fit(X_train_model_dt, y_train[user_input._output_col])

    print("###################--CV Result--########################\n")

    print("Best " + user_input.scoring + " Score Obtained:")
    print(svc_kernel_random.best_score_)
    print("\nBest Model Parameter Set for Highest " + user_input.scoring + ":")
    print(svc_kernel_random.best_params_)

    ######################################
    print("\nSaving param to drive:")
    path = user_input._output_dir + "Model_Result/" + "SVM_Kernel/"
    if not os.path.isdir(path):
        os.makedirs(path)

    _param_dict = svc_kernel_random.best_params_
    _param_dict[user_input.scoring] = svc_kernel_random.best_score_
    _param = pd.DataFrame(_param_dict, index=['iter' + str(train_test_iter_num)])
    appendDFToCSV_void(df=_param, csvFilePath=path + 'cv_param.tsv', sep='\t')
    ######################################

    print("\n#######################################################\n")

    model_performance(X_test_model_dt=X_train_model_dt, y_test=y_train[user_input._output_col],
                      model_name="SVM_Kernel", model_object=svc_kernel_random,
                      output_path=user_input._output_dir + "Model_Result/", prob=False,
                      train_test_iter_num=train_test_iter_num, ID=train_ID, train_set=True)

    model_performance(X_test_model_dt=X_test_model_dt, y_test=y_test[user_input._output_col],
                      model_name="SVM_Kernel", model_object=svc_kernel_random,
                      output_path=user_input._output_dir + "Model_Result/", prob=False,
                      train_test_iter_num=train_test_iter_num, ID= test_ID, train_set= False)

    print("##########################################################\n")


def Random_Forest(X_train_model_dt, y_train, X_test_model_dt, y_test, **kwargs):
    from sklearn.ensemble import RandomForestClassifier

    # get the tran test iteration number to store the results per iter
    if ('train_test_iter_num' in kwargs.keys()):
        train_test_iter_num = kwargs.get("train_test_iter_num")
    else:
        train_test_iter_num = 1

    if ('train_ID' in kwargs.keys()):
        train_ID = kwargs.get("train_ID")
    else:
        train_ID = []

    if ('test_ID' in kwargs.keys()):
        test_ID = kwargs.get("test_ID")
    else:
        test_ID = []

    random_grid = {'n_estimators': user_input.n_estimators,
                   'max_features': user_input.max_features,
                   'max_depth': user_input.max_depth,
                   'min_samples_split': user_input.min_samples_split,
                   'min_samples_leaf': user_input.min_samples_leaf,
                   'bootstrap': user_input.bootstrap,
                   'class_weight': user_input.class_weight}

    # length of exhaustive set of parameter combination
    max_n_iter = 1
    for key, value in random_grid.items():
        max_n_iter *= len(value)

    # print(random_grid)

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using n fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                   n_iter=min(user_input.n_iter, max_n_iter), cv=user_input.cv,
                                   verbose=user_input.verbose,
                                   random_state=42, scoring=user_input.scoring)

    # Fit the random search model
    print(str(user_input.cv) + "-Fold CV in Progress...")
    rf_random.fit(X_train_model_dt, y_train[user_input._output_col])

    print("###################--CV Result--########################\n")

    print("Best " + user_input.scoring + " Score Obtained:")
    print(rf_random.best_score_)
    print("\nBest Model Parameter Set for Highest " + user_input.scoring + ":")
    print(rf_random.best_params_)

    ######################################
    print("\nSaving param to drive:")
    path = user_input._output_dir + "Model_Result/" + "Random_Forest/"
    if not os.path.isdir(path):
        os.makedirs(path)

    _param_dict = rf_random.best_params_
    _param_dict[user_input.scoring] = rf_random.best_score_
    _param = pd.DataFrame(_param_dict, index=['iter' + str(train_test_iter_num)])
    appendDFToCSV_void(df=_param, csvFilePath=path + 'cv_param.tsv', sep='\t')
    ######################################

    importances = rf_random.best_estimator_.feature_importances_

    plt_feature_imp(importances=importances, feature_list=X_train_model_dt.columns.values,
                    n_top_features=min(len(X_train_model_dt.columns.values), 30),
                    image_dir=path, train_test_iter_num=train_test_iter_num)

    print("\n#########################################################\n")

    model_performance(X_test_model_dt=X_train_model_dt, y_test=y_train[user_input._output_col],
                      model_name="Random_Forest", model_object=rf_random,
                      output_path=user_input._output_dir + "Model_Result/", prob=True,
                      train_test_iter_num=train_test_iter_num, ID=train_ID, train_set=True)

    model_performance(X_test_model_dt=X_test_model_dt, y_test=y_test[user_input._output_col],
                      model_name="Random_Forest", model_object=rf_random,
                      output_path=user_input._output_dir + "Model_Result/", prob=True,
                      train_test_iter_num=train_test_iter_num, ID= test_ID, train_set= False)

    print("###########################################################\n")


def Xgboost(X_train_model_dt, y_train, X_test_model_dt, y_test, **kwargs):
    from xgboost import XGBClassifier

    # get the tran test iteration number to store the results per iter
    if ('train_test_iter_num' in kwargs.keys()):
        train_test_iter_num = kwargs.get("train_test_iter_num")
    else:
        train_test_iter_num = 1

    if ('train_ID' in kwargs.keys()):
        train_ID = kwargs.get("train_ID")
    else:
        train_ID = []

    if ('test_ID' in kwargs.keys()):
        test_ID = kwargs.get("test_ID")
    else:
        test_ID = []

    random_grid = {
        'n_estimators' : user_input.XGB_n_estimators,
        'min_child_weight': user_input.XGB_min_child_weight,
        'gamma': user_input.XGB_gamma,
        'subsample': user_input.XGB_subsample,
        'colsample_bytree': user_input.XGB_colsample_bytree,
        'max_depth': user_input.XGB_max_depth,
        'learning_rate': user_input.XGB_learning_rate,
        'scale_pos_weight': user_input.XGB_scale_pos_weight,
        'objective': user_input.XGB_objective,
        'max_delta_step': user_input.XGB_max_delta_step
    }

    # length of exhaustive set of parameter combination
    max_n_iter = 1
    for key, value in random_grid.items():
        max_n_iter *= len(value)

    # print(random_grid)

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    XGBC = XGBClassifier()
    # Random search of parameters, using n fold cross validation,
    # search across 100 different combinations, and use all available cores
    XGBC_random = RandomizedSearchCV(estimator=XGBC, param_distributions=random_grid,
                                     n_iter=min(user_input.n_iter, max_n_iter), cv=user_input.cv,
                                     verbose=user_input.verbose,
                                     random_state=42, scoring=user_input.scoring)

    # Fit the random search model
    print(str(user_input.cv) + "-Fold CV in Progress...")
    XGBC_random.fit(X_train_model_dt, y_train[user_input._output_col])

    print("###################--CV Result--########################\n")

    print("Best " + user_input.scoring + " Score Obtained:")
    print(XGBC_random.best_score_)
    print("\nBest Model Parameter Set for Highest " + user_input.scoring + ":")
    print(XGBC_random.best_params_)

    ######################################
    print("\nSaving param to drive:")
    path = user_input._output_dir + "Model_Result/" + "XgBoost/"
    if not os.path.isdir(path):
        os.makedirs(path)

    _param_dict = XGBC_random.best_params_
    _param_dict[user_input.scoring] = XGBC_random.best_score_
    _param = pd.DataFrame(_param_dict, index=['iter' + str(train_test_iter_num)])
    appendDFToCSV_void(df=_param, csvFilePath=path + 'cv_param.tsv', sep='\t')
    ######################################

    importances = XGBC_random.best_estimator_.feature_importances_

    path = user_input._output_dir + "Model_Result/" + "Xgboost/"
    if not os.path.isdir(path):
        os.makedirs(path)

    plt_feature_imp(importances=importances, feature_list=X_train_model_dt.columns.values,
                    n_top_features=min(len(X_train_model_dt.columns.values), 30),
                    image_dir=path, train_test_iter_num=train_test_iter_num)

    print("\n#########################################################\n")

    model_performance(X_test_model_dt=X_train_model_dt, y_test=y_train[user_input._output_col],
                      model_name="XgBoost", model_object=XGBC_random,
                      output_path=user_input._output_dir + "Model_Result/", prob=True,
                      train_test_iter_num=train_test_iter_num, ID=train_ID, train_set=True)

    model_performance(X_test_model_dt=X_test_model_dt, y_test=y_test[user_input._output_col],
                      model_name="XgBoost", model_object=XGBC_random,
                      output_path=user_input._output_dir + "Model_Result/", prob=True,
                      train_test_iter_num=train_test_iter_num, ID= test_ID, train_set=False)

    print("###########################################################\n")

# def create_ann_model(optimizer='adam', activation = 'sigmoid'):
#
#     from keras import models
#     from keras import layers
#
#     # Initialize the constructor
#     model = models.Sequential()
#     # Add an input layer
#     model.add(layers.Dense(8, activation=activation, input_shape=(12,)))
#
#     # for i in range(hidden_layers):
#     #   Add one hidden layer
#     model.add(layers.Dense(8, activation=activation))
#
#     # Add an output layer
#     model.add(layers.Dense(1, activation='sigmoid'))
#     #compile model
#     model.compile(loss= 'binary_crossentropy', optimizer= optimizer, metrics= ["accuracy"])
#     return model

# def create_model(optimizer='rmsprop', init='glorot_uniform', activation = 'relu', hidden_layers = 1):
#     from keras import models
#     from keras import layers
#     from keras import backend
#
#     # create model
#     model = models.Sequential()
#     model.add(layers.Dense(8, input_dim=None, kernel_initializer=init, activation= activation))
#
#     for i in range(hidden_layers):
#         #   Add one hidden layer
#         model.add(layers.Dense(8, kernel_initializer=init, activation= activation))
#
#     model.add(layers.Dense(1, kernel_initializer=init, activation='sigmoid'))
#
#     # Compile model
#     model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#     return model
#
def create_class_weight(labels_dict,mu=0.15):
    import  math

    total = sum(labels_dict.values())
    keys = labels_dict.keys()
    # print(total)
    class_weight = dict()

    for key in keys:
        # print(float(labels_dict[key]))
        score = math.log(mu*total/(float(labels_dict[key])))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight

def ANN(X_train_model_dt, y_train, X_test_model_dt, y_test, model_def, **kwargs):

    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.utils import class_weight

    # get the tran test iteration number to store the results per iter
    if ('train_test_iter_num' in kwargs.keys()):
        train_test_iter_num = kwargs.get("train_test_iter_num")
    else:
        train_test_iter_num = 1

    if ('train_ID' in kwargs.keys()):
        train_ID = kwargs.get("train_ID")
    else:
        train_ID = []

    if ('test_ID' in kwargs.keys()):
        test_ID = kwargs.get("test_ID")
    else:
        test_ID = []

    random_grid = {
        'epochs': user_input.NN_epochs,
        'batch_size': user_input.NN_batches,
        'init' : user_input.NN_init,
        'optimizer': user_input.NN_optimizers,
        'activation': user_input.NN_activation,
        'lr' : user_input.NN_learn_rate,
        'dropout_rate' : user_input.NN_dropout_rate,
        'weight_constraint' : user_input.NN_weight_constraint,
        'momentum' : user_input.NN_momentum,
        'hidden_layers': user_input.NN_hidden_layers,
        'neurons' : user_input.NN_neurons,
        'decay' : user_input.NN_decay
    }

    # length of exhaustive set of parameter combination
    max_n_iter = 1
    for key, value in random_grid.items():
        max_n_iter *= len(value)

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    # class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train[user_input._output_col]),
    #                                                   y_train[user_input._output_col])
    # class_weight_dict = dict(enumerate(class_weights))

    labels_dict = y_train.groupby([user_input._output_col])[user_input._output_col].count().to_dict()

    print("\nOutcome label count in train set:")
    print(labels_dict)

    class_weight_dict = create_class_weight(labels_dict, mu= user_input.NN_weight_factor)

    print("\nWeight given to the classes:")
    print(class_weight_dict)
    # print(class_weight_dict)
    annc = KerasClassifier(build_fn=model_def, verbose = 0, class_weight = class_weight_dict)

    # Random search of parameters, using n fold cross validation,
    # search across 100 different combinations, and use all available cores
    annc_random = RandomizedSearchCV(estimator=annc, param_distributions=random_grid,
                                     n_iter=min(user_input.n_iter, max_n_iter), cv=user_input.cv,
                                     verbose=user_input.verbose,
                                     random_state=42, scoring=user_input.scoring)

    # Fit the random search model
    print("\n" + str(user_input.cv) + "-Fold CV in Progress...")
    annc_random.fit(X_train_model_dt, y_train[user_input._output_col])

    print("###################--CV Result--########################\n")

    print("Best " + "Accuracy" + " Score Obtained:")
    print(annc_random.best_score_)
    print("Best Model Parameter Set for Highest " + user_input.scoring + ":\n")
    print(annc_random.best_params_)

    ######################################
    print("\nSaving param to drive:")
    path = user_input._output_dir + "Model_Result/" + "ANN/"
    if not os.path.isdir(path):
        os.makedirs(path)

    _param_dict = annc_random.best_params_
    _param_dict[user_input.scoring] = annc_random.best_score_
    _param = pd.DataFrame(_param_dict, index=['iter' + str(train_test_iter_num)])
    appendDFToCSV_void(df=_param, csvFilePath=path + 'cv_param.tsv', sep='\t')
    ######################################

    print("\n#########################################################\n")

    model_performance(X_test_model_dt=X_train_model_dt, y_test=y_train[user_input._output_col],
                      model_name="ANN", model_object=annc_random,
                      output_path=user_input._output_dir + "Model_Result/", prob=True,
                      train_test_iter_num=train_test_iter_num, ID=train_ID, train_set=True)

    model_performance(X_test_model_dt=X_test_model_dt, y_test=y_test[user_input._output_col],
                      model_name="ANN", model_object=annc_random,
                      output_path=user_input._output_dir + "Model_Result/", prob=True,
                      train_test_iter_num=train_test_iter_num, ID=test_ID, train_set=False)

    print("###########################################################\n")





