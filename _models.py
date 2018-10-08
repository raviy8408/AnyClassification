from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score
import _user_input as user_input
from _plot_func import *
from _support_func import *

def Logistic_Regresion(X_train_model_dt, y_train, X_test_model_dt, y_test):
    from sklearn.linear_model import LogisticRegression

    random_grid = {'penalty': user_input.penalty,
                   'C': user_input.C,
                   }

    # length of exhaustive set of parameter combination
    max_n_iter = 1
    for key, value in random_grid.items():
        max_n_iter *= len(value)

    logreg = LogisticRegression(class_weight='balanced')
    # logreg.fit(X_train_model_dt, y_train[user_input._output_col])

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
    print("Best Model Parameter Set for Highest " + user_input.scoring + ":\n")
    print(lr_random.best_params_)

    print("\n P values for variables:\n")
    cal_lr_p_vals(X =X_train_model_dt, y = y_train[user_input._output_col],
                  params = np.append(lr_random.best_estimator_.intercept_, lr_random.best_estimator_.coef_),
                  predictions = lr_random.best_estimator_.predict(X_train_model_dt))

    print("\n#########################################################\n")

    print("#################--Model Performance--#################\n")

    path = user_input._output_dir + "Model_Result/" + "Logistic_Regression/"
    if not os.path.isdir(path):
        os.makedirs(path)

    # Saving ROC plot to the drive
    plot_ROC(y_test=y_test[user_input._output_col],
             y_pred_prob=lr_random.best_estimator_.predict_proba(X_test_model_dt)[:, 1],
             model_name='Logistic Regression',
             image_dir=path)
    print("ROC plot saved to the drive!\n")

    print("Model Performance on Test Set:\n")
    print("Accuracy:\n")
    print(str(
        accuracy_score(y_test[user_input._output_col], lr_random.best_estimator_.predict(X_test_model_dt))))
    print("\nConfusion Matrix:\n")
    # print(confusion_matrix(y_test[user_input._output_col],rf_random.best_estimator_.predict(X_test_oneHotEncoded)))
    print(pd.crosstab(y_test[user_input._output_col], lr_random.best_estimator_.predict(X_test_model_dt),
                      rownames=['True'], colnames=['Predicted'], margins=True))
    print("\nClassification Report:\n")
    print(classification_report(y_test[user_input._output_col],
                                lr_random.best_estimator_.predict(X_test_model_dt)))
    print("\nCohen Kappa:\n")
    print(cohen_kappa_score(y_test[user_input._output_col],
                            lr_random.best_estimator_.predict(X_test_model_dt)))

    print("#######################################################\n")

def SVC_Linear(X_train_model_dt, y_train, X_test_model_dt, y_test):

    from sklearn.svm import SVC

    print("Running linear SVM..\n")

    random_grid = {'C': user_input.C_svm, 'kernel': ['linear']}
    # {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},

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
    print("Best Model Parameter Set for Highest " + user_input.scoring + ":\n")
    print(svc_linear_random.best_params_)

    print("\n#########################################################\n")

    print("#################--Model Performance--#################\n")

    path = user_input._output_dir + "Model_Result/" + "SVM/svm_linear/"
    if not os.path.isdir(path):
        os.makedirs(path)

    print("ROC plot saved to the drive!\n")

    print("Model Performance on Test Set:\n")
    print("Accuracy:\n")
    print(
        str(accuracy_score(y_test[user_input._output_col], svc_linear_random.best_estimator_.predict(X_test_model_dt))))
    print("\nConfusion Matrix:\n")
    # print(confusion_matrix(y_test[user_input._output_col],rf_random.best_estimator_.predict(X_test_oneHotEncoded)))
    print(pd.crosstab(y_test[user_input._output_col], svc_linear_random.best_estimator_.predict(X_test_model_dt),
                      rownames=['True'], colnames=['Predicted'], margins=True))
    print("\nClassification Report:\n")
    print(classification_report(y_test[user_input._output_col],
                                svc_linear_random.best_estimator_.predict(X_test_model_dt)))
    print("\nCohen Kappa:\n")
    print(cohen_kappa_score(y_test[user_input._output_col], svc_linear_random.best_estimator_.predict(X_test_model_dt)))

    print("#######################################################\n")

def Random_Forest(X_train_model_dt, y_train, X_test_model_dt, y_test):

    from sklearn.ensemble import RandomForestClassifier

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
    print("Best Model Parameter Set for Highest " + user_input.scoring + ":\n")
    print(rf_random.best_params_)

    importances = rf_random.best_estimator_.feature_importances_

    path = user_input._output_dir + "Model_Result/" + "Random_Forest/"
    if not os.path.isdir(path):
        os.makedirs(path)

    plt_feature_imp(importances=importances, feature_list=X_train_model_dt.columns.values,
                    n_top_features=min(len(X_train_model_dt.columns.values), 30),
                    image_dir=path)

    print("\n#########################################################\n")

    print("#################--Model Performance--###################\n")

    # print(rf_random.cv_results_)
    # print(rf_random.grid_scores_)

    # Saving ROC plot to the drive
    plot_ROC(y_test=y_test[user_input._output_col],
             y_pred_prob=rf_random.best_estimator_.predict_proba(X_test_model_dt)[:, 1],
             model_name='Random Forest', image_dir=path)
    print("ROC plot saved to the drive!\n")

    print("Model Performance on Test Set:\n")
    print("Accuracy:\n")
    print(str(accuracy_score(y_test[user_input._output_col], rf_random.best_estimator_.predict(X_test_model_dt))))
    print("\nConfusion Matrix:\n")
    # print(confusion_matrix(y_test[user_input._output_col],rf_random.best_estimator_.predict(X_test_oneHotEncoded)))
    print(pd.crosstab(y_test[user_input._output_col], rf_random.best_estimator_.predict(X_test_model_dt),
                      rownames=['True'], colnames=['Predicted'], margins=True))
    print("\nClassification Report:\n")
    print(classification_report(y_test[user_input._output_col], rf_random.best_estimator_.predict(X_test_model_dt)))
    print("\nCohen Kappa:\n")
    print(cohen_kappa_score(y_test[user_input._output_col], rf_random.best_estimator_.predict(X_test_model_dt)))

    print("#######################################################\n")