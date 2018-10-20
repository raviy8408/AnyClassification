from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score
import _user_input as user_input
from _plot_func import *
from _support_func import *
import pandas as pd
import numpy as np


def Logistic_Regresion(X_train_model_dt, y_train, X_test_model_dt, y_test, **kwargs):
    from sklearn.linear_model import LogisticRegression

    # get the tran test iteration number to store the results per iter
    if ('train_test_iter_num' in kwargs.keys()):
        train_test_iter_num = kwargs.get("train_test_iter_num")
    else:
        train_test_iter_num = 1

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

    print("\n P values for variables:\n")
    cal_lr_p_vals(X=X_train_model_dt, y=y_train[user_input._output_col],
                  params=np.append(lr_random.best_estimator_.intercept_, lr_random.best_estimator_.coef_),
                  predictions=lr_random.best_estimator_.predict(X_train_model_dt))

    print("\n#####################################################\n")

    model_performance(X_test_model_dt=X_test_model_dt, y_test=y_test[user_input._output_col],
                      model_name="Logistic_Regression", model_object=lr_random,
                      output_path=user_input._output_dir + "Model_Result/", prob=True,
                      train_test_iter_num=train_test_iter_num)

    print("#######################################################\n")



def SVM_Linear(X_train_model_dt, y_train, X_test_model_dt, y_test, **kwargs):
    from sklearn.svm import SVC

    # get the tran test iteration number to store the results per iter
    if ('train_test_iter_num' in kwargs.keys()):
        train_test_iter_num = kwargs.get("train_test_iter_num")
    else:
        train_test_iter_num = 1

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

    print("\n######################################################\n")

    model_performance(X_test_model_dt=X_test_model_dt, y_test=y_test[user_input._output_col],
                      model_name="SVM_Linear", model_object=svc_linear_random,
                      output_path=user_input._output_dir + "Model_Result/", prob=False,
                      train_test_iter_num=train_test_iter_num)

    print("#######################################################\n")


def SVM_Kernel(X_train_model_dt, y_train, X_test_model_dt, y_test, **kwargs):
    from sklearn.svm import SVC

    # get the tran test iteration number to store the results per iter
    if ('train_test_iter_num' in kwargs.keys()):
        train_test_iter_num = kwargs.get("train_test_iter_num")
    else:
        train_test_iter_num = 1

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

    print("\n#######################################################\n")

    model_performance(X_test_model_dt=X_test_model_dt, y_test=y_test[user_input._output_col],
                      model_name="SVM_Kernel", model_object=svc_kernel_random,
                      output_path=user_input._output_dir + "Model_Result/", prob=False,
                      train_test_iter_num=train_test_iter_num)

    print("##########################################################\n")


def Random_Forest(X_train_model_dt, y_train, X_test_model_dt, y_test, **kwargs):
    from sklearn.ensemble import RandomForestClassifier

    # get the tran test iteration number to store the results per iter
    if ('train_test_iter_num' in kwargs.keys()):
        train_test_iter_num = kwargs.get("train_test_iter_num")
    else:
        train_test_iter_num = 1

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

    importances = rf_random.best_estimator_.feature_importances_

    path = user_input._output_dir + "Model_Result/" + "Random_Forest/"
    if not os.path.isdir(path):
        os.makedirs(path)

    plt_feature_imp(importances=importances, feature_list=X_train_model_dt.columns.values,
                    n_top_features=min(len(X_train_model_dt.columns.values), 30),
                    image_dir=path, train_test_iter_num=train_test_iter_num)

    print("\n#########################################################\n")

    model_performance(X_test_model_dt=X_test_model_dt, y_test=y_test[user_input._output_col],
                      model_name="Random_Forest", model_object=rf_random,
                      output_path=user_input._output_dir + "Model_Result/", prob=True,
                      train_test_iter_num=train_test_iter_num)

    print("###########################################################\n")


def Xgboost(X_train_model_dt, y_train, X_test_model_dt, y_test, **kwargs):
    from xgboost import XGBClassifier

    # get the tran test iteration number to store the results per iter
    if ('train_test_iter_num' in kwargs.keys()):
        train_test_iter_num = kwargs.get("train_test_iter_num")
    else:
        train_test_iter_num = 1

    random_grid = {
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

    importances = XGBC_random.best_estimator_.feature_importances_

    path = user_input._output_dir + "Model_Result/" + "Xgboost/"
    if not os.path.isdir(path):
        os.makedirs(path)

    plt_feature_imp(importances=importances, feature_list=X_train_model_dt.columns.values,
                    n_top_features=min(len(X_train_model_dt.columns.values), 30),
                    image_dir=path, train_test_iter_num=train_test_iter_num)

    print("\n#########################################################\n")

    model_performance(X_test_model_dt=X_test_model_dt, y_test=y_test[user_input._output_col],
                      model_name="XgBoost", model_object=XGBC_random,
                      output_path=user_input._output_dir + "Model_Result/", prob=True,
                      train_test_iter_num=train_test_iter_num)

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

# def create_model(optimizer='rmsprop', init='glorot_uniform', activation = 'relu'):
#     from keras import models
#     from keras import layers
#     # create model
#     model = models.Sequential()
#     model.add(layers.Dense(8, input_dim=12, kernel_initializer=init, activation= activation))
#     model.add(layers.Dense(8, kernel_initializer=init, activation= activation))
#     model.add(layers.Dense(1, kernel_initializer=init, activation='sigmoid'))
#     # Compile model
#     model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#     return model
#
# def ANN(X_train_model_dt, y_train, X_test_model_dt, y_test):
#
#     from keras.wrappers.scikit_learn import KerasClassifier
#     from sklearn.model_selection import GridSearchCV
#
#     random_grid = {
#         'epochs': user_input.NN_epochs,
#         'batch_size': user_input.NN_batches,
#         'optimizer': user_input.NN_optimizers,
#         'activation': user_input.NN_activation,
#         # 'hidden_layers': user_input.NN_hidden_layers
#     }
#
#     # length of exhaustive set of parameter combination
#     max_n_iter = 1
#     for key, value in random_grid.items():
#         max_n_iter *= len(value)
#
#     # # print(random_grid)
#     #
#     # annc = KerasClassifier(build_fn=create_model, verbose = 0)
#     # grid = GridSearchCV(estimator=annc, param_grid=random_grid, n_jobs=1)
#     # grid_result = grid.fit(X_train_model_dt, y_train)
#     # # summarize results
#     # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#     # means = grid_result.cv_results_['mean_test_score']
#     # stds = grid_result.cv_results_['std_test_score']
#     # params = grid_result.cv_results_['params']
#     # for mean, stdev, param in zip(means, stds, params):
#     #     print("%f (%f) with: %r" % (mean, stdev, param))
#
#     ###################################################################################
#
#     # Use the random grid to search for best hyperparameters
#     # First create the base model to tune
#     annc = KerasClassifier(build_fn=create_model, verbose = 1)
#
#     # Random search of parameters, using n fold cross validation,
#     # search across 100 different combinations, and use all available cores
#     annc_random = RandomizedSearchCV(estimator=annc, param_distributions=random_grid,
#                                      n_iter=min(user_input.n_iter, max_n_iter), cv=user_input.cv,
#                                      verbose=user_input.verbose,
#                                      random_state=42, scoring=user_input.scoring)
#
#     # Fit the random search model
#     print(str(user_input.cv) + "-Fold CV in Progress...")
#     annc_random.fit(X_train_model_dt, y_train[user_input._output_col])
#
#     print("###################--CV Result--########################\n")
#
#     print("Best " + "Accuracy" + " Score Obtained:")
#     print(annc_random.best_score_)
#     print("Best Model Parameter Set for Highest " + user_input.scoring + ":\n")
#     print(annc_random.best_params_)
#
#     # importances = annc_random.best_estimator_.feature_importances_
#
#     path = user_input._output_dir + "Model_Result/" + "ANN/"
#     if not os.path.isdir(path):
#         os.makedirs(path)
#
#     # plt_feature_imp(importances=importances, feature_list=X_train_model_dt.columns.values,
#     #                 n_top_features=min(len(X_train_model_dt.columns.values), 30),
#     #                 image_dir=path)
#
#     # print("\n#########################################################\n")
#
#     print("#################--Model Performance--###################\n")
#
#     # print(rf_random.cv_results_)
#     # print(rf_random.grid_scores_)
#
#     # Saving ROC plot to the drive
#     plot_ROC(y_test=y_test[user_input._output_col],
#              y_pred_prob=annc_random.best_estimator_.predict_proba(X_test_model_dt)[:, 1],
#              model_name='ANN', image_dir=path)
#     print("ROC plot saved to the drive!\n")
#
#     print("Model Performance on Test Set:\n")
#     print("Accuracy:\n")
#     print(str(accuracy_score(y_test[user_input._output_col], annc_random.best_estimator_.predict(X_test_model_dt))))
#     print("\nConfusion Matrix:\n")
#     # print(confusion_matrix(y_test[user_input._output_col],rf_random.best_estimator_.predict(X_test_oneHotEncoded)))
#     print(pd.crosstab(y_test[user_input._output_col], annc_random.best_estimator_.predict(X_test_model_dt)[:,0],
#                       rownames=['True'], colnames=['Predicted'], margins=True))
#
#     print("\nClassification Report:\n")
#     print(classification_report(y_test[user_input._output_col], annc_random.best_estimator_.predict(X_test_model_dt)))
#     print("\nCohen Kappa:\n")
#     print(cohen_kappa_score(y_test[user_input._output_col], annc_random.best_estimator_.predict(X_test_model_dt)))
#
#     print("#######################################################\n")





