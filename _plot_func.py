import matplotlib.pyplot as plt
import os

def plt_hist(x, colname, n_bin, dir_name):
    '''
    :param x: pd.series of numerical values
    :return: saves histograms for categorical variable
    '''
    fig = plt.figure()
    plt.hist(x, bins= n_bin)
    plt.title("histogram-" + colname)
    plt.xlabel(colname)
    plt.ylabel("count")

    save_file = os.path.join(dir_name, colname + ".png")
    plt.savefig(save_file, bbox_inches='tight')
    plt.close(fig)