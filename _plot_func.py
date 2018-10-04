import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import os
rcParams['figure.figsize'] = 15, 6

def plot_num_value_hist(data, field, n_bin, dir_name):
    '''
    :param x: pd.series of numerical values
    :return: saves histograms for categorical variable
    '''
    fig = plt.figure()
    plt.hist(data, bins= n_bin)
    plt.title("Histogram-" + field)
    plt.xlabel(field)
    plt.ylabel("count")
    plt.grid(axis='y', alpha=0.75)
    plt.grid(axis='x', alpha=0.75)

    save_file = os.path.join(dir_name, field + ".png")
    plt.savefig(save_file, bbox_inches='tight')
    plt.close(fig)

def plot_count_hist(data, field, num_bar, x_lim, dir_name , **kwargs):
    """
    plot the histogram of count
    :param data: df
    :param field: feature name(string)
    :param num_bar: bar count
    :param image_dir: dir name
    :return: none
    """

    if ('title' in kwargs.keys()):
        title = kwargs.get('title')
    else:
        title = "Histogram-" + field

    if ('x_label' in kwargs.keys()):
        x_label = kwargs.get('x_label')
    else:
        x_label = field

    fig = plt.figure()
    # print(data.groupby([field]).size())
    ax =data[field].value_counts().sort_index().plot(kind = 'bar',grid=True)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.75)
    # create a list to collect the plt.patches data
    totals = []

    # find the values and append to list
    for i in ax.patches:
        totals.append(i.get_height())

    # set individual bar lables using above list
    total = sum(totals)

    for i in ax.patches[:(num_bar + 1)]:
        # get_x pulls left or right; get_height pushes up or down
        ax.text(i.get_x(), i.get_height()+ 0.1,\
                str(round((i.get_height()/total)*100, 2))+'%', fontsize=13)
    ax.set_xlim(left=None, right= x_lim)
    save_file = os.path.join(dir_name, title + ".png")
    plt.savefig(save_file, bbox_inches='tight')
    plt.close(fig)

def eda_plots(data, cat_feature_list, outcome_col, output_dir):
    """
    Generated univariate and bivariate data plots to provide data insights
    :param data: input data containing both X and y
    :param cat_feature_list: categorical feature list(list of string)
    :param outcome_col: outcome column(string)
    :param output_dir: output folder to store all the eda plots
    :return:
    """
    print("Saving Numerical Variable Histogram to Output Directory...")
    path = output_dir + "eda_plots/" + "numerical_variables/"
    if not os.path.isdir(path):
        os.makedirs(path)
    # else:
    #     shutil.rmtree(path=path)
    #     os.makedirs(path)
    for col in (set(list(data)) - set(cat_feature_list) - set(
            list([outcome_col]))):
        plot_num_value_hist(data=data[col], field=col, n_bin=20, dir_name=path)

    if cat_feature_list:
        print("Saving Categorical Variable Histogram to Output Directory...")
        path = output_dir + "eda_plots/" + "categorical_variables/"
        if not os.path.isdir(path):
            os.makedirs(path)
        # else:
        #     shutil.rmtree(path=path)
        #     os.makedirs(path)
        for col in (set(cat_feature_list)):
            bar_count = len(data[col].unique())
            plot_count_hist(data=data, field=col, num_bar=bar_count, x_lim=bar_count + 0.01, dir_name=path)

    print("Saving Outcome Variable Histogram to Output Directory...")
    path = output_dir + "eda_plots/" + "outcome_variables/"
    if not os.path.isdir(path):
        os.makedirs(path)
    # else:
    #     shutil.rmtree(path=path)
    #     os.makedirs(path)
    for col in (set(list([outcome_col]))):
        bar_count = len(data[col].unique())
        plot_count_hist(data=data, field=col, num_bar=bar_count, x_lim=bar_count + 0.01, dir_name=path)
    print("Completed!\n")


def plot_ROC(y_test, y_pred_prob, model_name, image_dir):

    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve

    rcParams['figure.figsize'] = 8, 6
    logit_roc_auc = roc_auc_score(y_test, y_pred_prob)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    fig = plt.figure()
    plt.plot(fpr, tpr, label= model_name + ' (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(image_dir + 'ROC')
    plt.close(fig)

def plt_feature_imp(importances, feature_list, n_top_features, image_dir):

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [feature_list[i] for i in indices][:n_top_features]

    fig = plt.figure()
    # Barplot: Add bars
    # plt.bar(range(X_train_oneHotEncoded.shape[1]), importances[indices])
    plt.bar(range(n_top_features), importances[indices[:n_top_features]])
    # Add feature names as x-axis labels
    # plt.xticks(range(X_train_oneHotEncoded.shape[1]), names, rotation=20, fontsize = 8)
    plt.xticks(range(n_top_features), names, rotation=90, fontsize=8)
    # Create plot title
    plt.title("Feature Importance")
    # Show plot
    plt.savefig(image_dir + "feature_imp" + '.png', bbox_inches='tight')
    plt.close(fig)