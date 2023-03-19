from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
import pdb
import matplotlib.pyplot as plt

from lamda_path import run_selection_path
import util_interactions
import stats_all
import time

import os
import shutil
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler


def plot(data_sf, data_ss, data_sj, data_sjj, n_samples, xlabel, ylabel, fname, ylim, loc, y_hzl=None):
    no_labels = [''] * len(n_samples)
    # pdb.set_trace()
    bp_sf = plt.boxplot(data_sf, patch_artist=True, positions=[3, 12, 21], labels=no_labels,
                        boxprops=dict(facecolor="C0", alpha=0.5))

    bp_ss = plt.boxplot(data_ss, patch_artist=True, positions=[4, 13, 22], labels=n_samples,
                        boxprops=dict(facecolor="C1", alpha=0.5))

    bp_sj = plt.boxplot(data_sj, patch_artist=True, positions=[5, 14, 23], labels=no_labels,
                        boxprops=dict(facecolor="C2", alpha=0.5))

    bp_sjj = plt.boxplot(data_sjj, patch_artist=True, positions=[6, 15, 24], labels=no_labels,
                         boxprops=dict(facecolor="C3", alpha=0.5))

    plt.tick_params(bottom=False)

    ax = plt.gca()
    ax.legend([bp_sf["boxes"][0], bp_ss["boxes"][0], bp_sj["boxes"][0], bp_sjj["boxes"][0]], ['full', 'split',
                                                                                              'jacknife', 'jacknife+'],
              loc=loc, prop={'size': 16}, ncol=2)

    ax.set_xlim(0, 30)
    ax.set_ylim(ylim)

    plt.ylabel(ylabel, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    if y_hzl is not None:
        plt.axhline(y=y_hzl, color='r', linestyle='--')

    plt.tight_layout()

    plt.savefig(fname, dpi=300)
    plt.show()
    plt.close()


def read_sample(sample_path):
    npzfile = np.load(sample_path, allow_pickle=True)
    X_train = npzfile['X_train']
    y_train = npzfile['y_train']

    X_test = npzfile['X_test']
    y_test = npzfile['y_test']

    return X_train, y_train, X_test, y_test


def opt_lmd_shim(dir_ms, max_depth):
    lmd_path_opt = dir_ms + 'opt_lmd_ord_' + str(max_depth) + '.npz'  # filepath to 'optimum lmd'.
    npzfile = np.load(lmd_path_opt, allow_pickle=True)
    lmd = npzfile['lmd_opt']

    return lmd


def fit_shim(X, y, lmd, alpha, max_depth):
    list_lmd, list_beta, list_aset, _, _, _ = run_selection_path(X, y, lmd_tgt=lmd, alpha=alpha, max_depth=max_depth,
                                                                 verbose=0)
    if len(list_aset) != 0:
        A = list_aset[-1]
        beta = list_beta[-1]
        return A, beta

    else:
        print("Debug: Inside 'fit_shim': Active set is empty !\n")
        pdb.set_trace()
        return None


def pred_shim(model, x_te):
    A, beta = model
    xA_te = util_interactions.construct_XA(x_te, A)
    y_pred = xA_te.dot(beta)
    # pdb.set_trace()

    return y_pred


def loo(X_tr, y_tr, lmd, alpha, max_depth):
    models = []
    r_loo = []

    for i in range(X_tr.shape[0]):
        x_te_, y_te_ = X_tr[i, :][np.newaxis, :], y_tr[i]
        X_tr_, y_tr_ = np.delete(X_tr, i, 0), np.delete(y_tr, i)

        # Fit model with all data but LOO sample
        model = fit_shim(X_tr_, y_tr_, lmd, alpha, max_depth)

        # Compute residual of LOO sample
        y_loo = pred_shim(model, x_te_).item()
        r = np.abs(y_te_ - y_loo)

        models += [model]
        r_loo += [r]

    return r_loo, models


def cp_jacknife(X_tr, y_tr, x_te, lmd, alpha, max_depth, sig_level):
    n = X_tr.shape[0]

    # Fit with full data
    model = fit_shim(X_tr, y_tr, lmd, alpha, max_depth)
    y_full = pred_shim(model, x_te)

    # Compute LOO residuals
    r_loo, _ = loo(X_tr, y_tr, lmd, alpha, max_depth)

    # Find (1 - sig_level) quantile of the loo residuals
    sorted_residual = np.sort(r_loo)
    index = int((n + 1) * (1 - sig_level))
    L = sorted_residual[index].item()
    # pdb.set_trace()

    # print("L_jacknife:{}\n".format(L))

    return y_full, L


def comp_jj_quantiles(x_te, models, r_loo, sig_level):
    try:
        assert len(models) == len(r_loo)
    except AssertionError:
        pdb.set_trace()

    n = len(models)

    r_loo_plus = []
    r_loo_minus = []

    for i in range(n):
        y_te = pred_shim(models[i], x_te).item()
        r_loo_plus += [y_te + r_loo[i]]
        r_loo_minus += [y_te - r_loo[i]]

    # Find (1 - sig_level) quantile of (y_te + r_loo)
    sorted_r_loo_plus = np.sort(r_loo_plus)
    index = int((n + 1) * (1 - sig_level))
    ub = sorted_r_loo_plus[index].item()

    # Find sig_level quantile of (y_te - r_loo)
    sorted_r_loo_minus = np.sort(r_loo_minus)
    index = int((n + 1) * sig_level)
    lb = sorted_r_loo_minus[index].item()

    # pdb.set_trace()

    return lb, ub


def cp_jacknife_plus(X_tr, y_tr, x_te, lmd, alpha, max_depth, sig_level):
    n_te = x_te.shape[0]

    # Compute LOO residuals
    r_loo, models = loo(X_tr, y_tr, lmd, alpha, max_depth)

    lbs, ubs = [], []
    for i in range(n_te):
        lb, ub = comp_jj_quantiles(x_te[i, :][np.newaxis, :], models, r_loo, sig_level)
        lbs += [lb]
        ubs += [ub]

    cp_set = np.array([lbs, ubs]).T.tolist()

    # pdb.set_trace()

    # Fit with full data
    model = fit_shim(X_tr, y_tr, lmd, alpha, max_depth)
    y_full = pred_shim(model, x_te)

    return y_full, cp_set


def cp_split(X_tr, y_tr, X_te, sig_level, lmd, alpha, max_depth):
    X_train, X_valid, y_train, y_valid = train_test_split(X_tr, y_tr, test_size=0.5)

    model = fit_shim(X_train, y_train, lmd, alpha, max_depth)

    if model is not None:
        y_valid_pred = pred_shim(model, X_valid)
        # pdb.set_trace()
        res = np.abs(y_valid - y_valid_pred)

        L = np.quantile(res, 1 - sig_level)

        y_te_pred = pred_shim(model, X_te)

        A, beta = model

        return y_te_pred, L, A

    else:
        return None


def comp_cp_set(y_te_pred, L):
    lb = y_te_pred - L
    ub = y_te_pred + L
    cp_set = np.array([lb.tolist(), ub.tolist()]).T.tolist()

    return cp_set


def split(X_train, y_train, X_test, y_test, sig_level, lmd, alpha, max_depth):
    n = y_test.shape[0]

    n_reps = 5
    covs, Ls, r2s, t = [], [], [], []

    len_A = []
    for _ in range(n_reps):  # repeat the "data splitting 'n_reps' times."

        start = time.time()

        # Compute the 'split CP'
        y_te_pred, L, A = cp_split(X_train, y_train, X_test, sig_level, lmd, alpha, max_depth)
        cp_set = comp_cp_set(y_te_pred, L)

        # Compute the 'r2score'
        r2 = r2_score(y_test, y_te_pred)

        count = 0
        for i in range(n):
            if cp_set[i][0] <= y_test[i] <= cp_set[i][1]:
                count += 1

        cov = count / n

        covs += [cov]
        Ls += [2 * L]
        r2s += [r2]
        t += [time.time() - start]

        len_A += [len(A)]

    # print("length of A in cp_split_shim:{} ({})\n".format(np.mean(len_A), np.round(np.std(len_A), 2)))
    return covs, Ls, r2s, t


def jacknife(X_train, y_train, X_test, y_test, sig_level, lmd, alpha, max_depth):
    n = y_test.shape[0]

    covs, Ls, r2s = [], [], []

    # Compute the 'split CP'
    y_te_pred, L = cp_jacknife(X_train, y_train, X_test, lmd, alpha, max_depth, sig_level)
    cp_set = comp_cp_set(y_te_pred, L)

    # Compute the 'r2score'
    r2 = r2_score(y_test, y_te_pred)

    count = 0
    for i in range(n):
        if cp_set[i][0] <= y_test[i] <= cp_set[i][1]:
            count += 1

    cov = count / n

    covs += [cov]
    Ls += [2 * L]
    r2s += [r2]

    # pdb.set_trace()

    return covs, Ls, r2s


def jacknife_plus(X_train, y_train, X_test, y_test, sig_level, lmd, alpha, max_depth):
    n = y_test.shape[0]

    covs, Ls, r2s = [], [], []

    # Compute the 'split CP'
    y_te_pred, cp_set = cp_jacknife_plus(X_train, y_train, X_test, lmd, alpha, max_depth, sig_level)

    # Compute the 'r2score'
    r2 = r2_score(y_test, y_te_pred)

    count = 0
    for i in range(n):
        L = cp_set[i][1] - cp_set[i][0]
        Ls += [L]

        if cp_set[i][0] <= y_test[i] <= cp_set[i][1]:
            count += 1

    cov = count / n

    covs += [cov]
    r2s += [r2]

    # pdb.set_trace()

    return covs, Ls, r2s


def create_dir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

    os.makedirs(dir_name)


def call_method(dir_cv, n_sample=3, method="full", ord=1, lmd=1.0):
    covs, cpls, r2s, ts = [], [], [], []
    for i in range(n_sample):
        sample_path = dir_cv + "sample_" + str(i) + "/" + "data.npz"
        X_train, y_train, X_test, y_test = read_sample(sample_path)

        print("dir_cv:{}, n_p_tr:{}, n_p_te:{}\n".format(dir_cv, X_train.shape, X_test.shape))

        if method == "full":
            start = time.time()
            cov, cpl, r2 = stats_all.stat_shim(X_train, y_train, X_test, y_test, split=False, lmd=lmd, alpha=0.001,
                                               max_depth=ord)
            t = [time.time() - start]
            print("Time (full): {}\n".format(t))

        elif method == "split":
            cov, cpl, r2, t = split(X_train, y_train, X_test, y_test, sig_level=0.1, lmd=lmd, alpha=0.001,
                                    max_depth=ord)
            print("Time (split): {}\n".format(np.mean(t)))

        elif method == "jacknife":
            start = time.time()
            cov, cpl, r2 = jacknife(X_train, y_train, X_test, y_test, sig_level=0.1, lmd=lmd, alpha=0.001,
                                    max_depth=ord)
            t = [time.time() - start]

            print("Time (jacknife): {}\n".format(t))

        elif method == "jacknife+":
            start = time.time()
            cov, cpl, r2 = jacknife_plus(X_train, y_train, X_test, y_test, sig_level=0.1, lmd=lmd, alpha=0.001,
                                         max_depth=ord)
            t = [time.time() - start]
            print("Time (jacknife+): {}\n".format(t))

        print("sample: {}, method: {} completed... \n".format(i, method))

        cpls += cpl
        covs += cov
        r2s += r2
        ts += t

    return covs, cpls, r2s, ts


def random_split(X, y, random_state, n_tr=400):
    """ randomly split data.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n, _ = X.shape
    idx = np.arange(n)

    idx_tr = list(np.random.choice(idx, size=n_tr, replace=False))
    idx_te = list(set(idx) - set(idx_tr))

    assert not set(idx_tr).intersection(idx_te)

    X_train, X_test, y_train, y_test = X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]

    return X_train, X_test, y_train, y_test


def regression_data(n, m, m_i, n_tr, random_state):
    X, y = make_regression(n_samples=n, n_features=m, n_informative=m_i, noise=1, random_state=random_state)

    # Scale 'X' and Normalize 'y'
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    y = (y - y.mean()) / y.std()

    return random_split(X, y, random_state=random_state, n_tr=n_tr)


def create_data(n=101, m=100, m_i=90, n_tr=100, random_state=None, n_sample=1, dir_cv=None):
    for i in range(n_sample):
        X_train, X_test, y_train, y_test = regression_data(n=n, m=m, m_i=m_i, n_tr=n_tr, random_state=random_state)
        sample_path = dir_cv + 'sample_' + str(i) + '/'
        fname = sample_path + 'data.npz'
        create_dir(sample_path)
        np.savez(fname, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def run(data_path="../data/make_regression_time/", ord=1, lmd=1.0):
    print("\n data_path: {}, ord={}, lmd={} \n".format(data_path, ord, lmd))

    sample_sizes = [100, 500, 1000]
    names = ['ss1', 'ss2', 'ss3']

    ns = 3
    m, m_i, n_te = 5, 3, 5

    time_sf, time_ss, time_sj, time_sjj = [], [], [], []

    for i in range(len(sample_sizes)):
        dir_cv = data_path + str(names[i]) + "/cv/"

        n = sample_sizes[i] + n_te
        create_data(n=n, m=m, m_i=m_i, n_tr=sample_sizes[i], random_state=None, n_sample=ns, dir_cv=dir_cv)

        _, _, _, ts_ss = call_method(dir_cv, n_sample=ns, method="split", ord=ord, lmd=lmd)

        _, _, _, ts_sf = call_method(dir_cv, n_sample=ns, method="full", ord=ord, lmd=lmd)

        _, _, _, ts_sj = call_method(dir_cv, n_sample=ns, method="jacknife", ord=ord, lmd=lmd)

        _, _, _, ts_sjj = call_method(dir_cv, n_sample=ns, method="jacknife+", ord=ord, lmd=lmd)

        time_ss += [ts_ss]
        time_sf += [ts_sf]
        time_sj += [ts_sj]
        time_sjj += [ts_sjj]

        print("====================== sample_size: {} is done ! ================== \n".format(sample_sizes[i]))

    fa = "_n_tr"
    for i in range(len(sample_sizes)):
        fa += "_" + str(sample_sizes[i])

    # pdb.set_trace()

    stat_path = "../results/make_regression_time/time_ord_" + str(ord) + "_lmd_" + str(lmd) + fa + ".npz"
    np.savez(stat_path, time_ss=time_ss, time_sf=time_sf, time_sj=time_sj, time_sjj=time_sjj, sample_sizes=sample_sizes)

    ylim = (-1, 410)
    loc = 'upper left'
    xlabel, ylabel, fname = "Sample size", "Time (sec)", "../results/make_regression_time/figures/time_ord_" + str(
        ord) + "_lmd_" + str(lmd) + fa + ".pdf"
    plot(time_sf, time_ss, time_sj, time_sjj, sample_sizes, xlabel, ylabel, fname, ylim, loc)


if __name__ == "__main__":
    run(data_path="../data/make_regression_time/", ord=5, lmd=0.1)
