import numpy as np
import os
import matplotlib.pyplot as plt


def create_dir(dir_name):

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def construct_XA(X, A):
    j = A[0]
    XA = np.prod(X[:, j], 1)[:, np.newaxis]
    i = 1
    while i < len(A):
        j = A[i]
        XA = np.column_stack((XA, np.prod(X[:, j], 1)[:, np.newaxis]))
        i += 1
    return XA


def plot(data_mlp, data_rf, data_lf, data_ls, data_sf, data_ss, n_samples, xlabel, ylabel, fname, ylim, loc = 'upper right', plt_hzl=False, y_hzl=0.9):
    no_labels = [''] * len(n_samples)
    # pdb.set_trace()
    bp_mlp = plt.boxplot(data_mlp, patch_artist=True, positions=[3, 12, 21], labels=no_labels, boxprops=dict(facecolor="C0", alpha=0.5))

    bp_rf = plt.boxplot(data_rf, patch_artist=True, positions=[4, 13, 22], labels=no_labels,
                        boxprops=dict(facecolor="C1", alpha=0.5))

    bp_ls = plt.boxplot(data_ls, patch_artist=True, positions=[5, 14, 23], labels=n_samples,
                        boxprops=dict(facecolor="C2", alpha=0.5))

    bp_lf = plt.boxplot(data_lf, patch_artist=True, positions=[6, 15, 24], labels=no_labels,
                        boxprops=dict(facecolor="C3", alpha=0.5))

    bp_ss = plt.boxplot(data_ss, patch_artist=True, positions=[7, 16, 25], labels=no_labels,
                        boxprops=dict(facecolor="C4", alpha=0.5))

    bp_sf = plt.boxplot(data_sf, patch_artist=True, positions=[8, 17, 26], labels=no_labels,
                        boxprops=dict(facecolor="C5", alpha=0.5))

    plt.tick_params(bottom=False)

    ax = plt.gca()
    ax.legend([bp_mlp["boxes"][0], bp_rf["boxes"][0], bp_ls["boxes"][0], bp_lf["boxes"][0], bp_ss["boxes"][0],
               bp_sf["boxes"][0]], ['mlp', 'rf', 'lasso_s', 'lasso_f', 'shim_s', 'shim_f'], loc=loc,
              prop={'size': 16}, ncol=3)

    ax.set_xlim(0, 30)
    ax.set_ylim(ylim)

    plt.ylabel(ylabel, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    if plt_hzl:
        plt.axhline(y=y_hzl, color='r', linestyle='--')

    plt.tight_layout()

    plt.savefig(fname, dpi=300)
    plt.show()