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