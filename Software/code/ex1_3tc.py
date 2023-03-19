import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import stats_all
import pdb
import util_interactions


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


def load_opt_param_mlp(param_path):
    npzfile = np.load(param_path, allow_pickle=True)
    hl = npzfile["hl"].tolist()
    act = npzfile["act"].tolist()

    return hl, act


def load_opt_param_rf(param_path):
    npzfile = np.load(param_path, allow_pickle=True)
    msl = npzfile["msl"].item()
    n_est = npzfile["n_est"].item()

    return msl, n_est


def call_method(dir_ms, dir_cv, n_sample=15, name="rf", split=False, ord=1):
    model = None,
    lamda = 1.0

    if name == "shim":
        max_depth = ord
        lamda = opt_lmd_shim(dir_ms, max_depth=max_depth)
        print("lamda: {}\n".format(lamda))

        if split:
            print("shim" + str(ord) + "_s...\n")
        else:
            print("shim" + str(ord) + "_f...\n")

    if name == "mlp":
        print("mlp...\n")
        param_path = dir_ms + "opt_param_mlp.npz"
        hl, act = load_opt_param_mlp(param_path)
        model = MLPRegressor(hidden_layer_sizes=hl, activation=act, random_state=0, max_iter=2000)

    elif name == "rf":
        print("rf...\n")
        param_path = dir_ms + "opt_param_rf.npz"
        msl, n_est = load_opt_param_rf(param_path)
        model = RandomForestRegressor(n_jobs=-1, n_estimators=n_est, min_samples_leaf=msl, warm_start=False,
                                      random_state=0)

    covs, cpls, r2s = [], [], []
    for i in range(n_sample):
        sample_path = dir_cv + "sample_" + str(i) + "/" + "data.npz"
        X_train, y_train, X_test, y_test = read_sample(sample_path)

        if name == "shim":
            cov, cpl, r2 = stats_all.stat_shim(X_train, y_train, X_test, y_test, split, lmd=lamda, alpha=0.001,
                                               max_depth=ord)
        else:
            cov, cpl, r2 = stats_all.stat(X_train, y_train, X_test, y_test, model)

        print("sample: {} completed...\n".format(i))

        cpls += cpl
        covs += cov
        r2s += r2

    return covs, cpls, r2s


def run():

    sample_sizes = [20, 50, 150]
    names = ['ss1', 'ss2', 'ss3']

    data_mlp, data_rf, data_sf, data_ss, data_lf, data_ls = [], [], [], [], [], []
    r2_mlps, r2_rfs, r2_sfs, r2_sss, r2_lfs, r2_lss = [], [], [], [], [], []
    cov_mlps, cov_rfs, cov_sfs, cov_sss, cov_lfs, cov_lss = [], [], [], [], [], []

    for i in range(len(sample_sizes)):

        dir_ms = "../data/real/hiv/3tc/" + str(names[i]) + "/ms/"
        dir_cv = "../data/real/hiv/3tc/" + str(names[i]) + "/cv/"

        cov_mlp, cpl_mlp, r2_mlp = call_method(dir_ms, dir_cv, n_sample=3, name="mlp")
        cov_rf, cpl_rf, r2_rf = call_method(dir_ms, dir_cv, n_sample=3, name="rf")
        #
        cov_ls, cpl_ls, r2_ls = call_method(dir_ms, dir_cv, n_sample=3, name="shim", split=True, ord=1)
        cov_lf, cpl_lf, r2_lf = call_method(dir_ms, dir_cv, n_sample=3, name="shim", split=False, ord=1)

        cov_ss, cpl_ss, r2_ss = call_method(dir_ms, dir_cv, n_sample=3, name="shim", split=True, ord=10)
        cov_sf, cpl_sf, r2_sf = call_method(dir_ms, dir_cv, n_sample=3, name="shim", split=False, ord=10)

        data_mlp += [cpl_mlp]
        data_rf += [cpl_rf]

        data_ls += [cpl_ls]
        data_lf += [cpl_lf]

        data_ss += [cpl_ss]
        data_sf += [cpl_sf]

        r2_mlps += [r2_mlp]
        r2_rfs += [r2_rf]

        r2_lss += [r2_ls]
        r2_lfs += [r2_lf]

        r2_sss += [r2_ss]
        r2_sfs += [r2_sf]

        cov_mlps += [cov_mlp]
        cov_rfs += [cov_rf]

        cov_lss += [cov_ls]
        cov_lfs += [cov_lf]

        cov_sss += [cov_ss]
        cov_sfs += [cov_sf]

        print("====================== sample_size: {} is done ! ================== \n".format(sample_sizes[i]))


    # pdb.set_trace()

    stat_path = "../results/real/hiv/3tc/stat_3tc_20_50_150.npz"
    np.savez(stat_path, data_mlp=data_mlp, data_rf=data_rf, data_lf=data_lf, data_ls=data_ls, data_sf=data_sf,
             data_ss=data_ss, r2_mlps=r2_mlps, r2_rfs=r2_rfs, r2_lfs=r2_lfs, r2_lss=r2_lss, r2_sfs=r2_sfs,
             r2_sss=r2_sss, cov_mlps=cov_mlps, cov_rfs=cov_rfs, cov_lss=cov_lss, cov_lfs=cov_lfs, cov_sss=cov_sss,
             cov_sfs=cov_sfs, sample_sizes=sample_sizes)

    ylim = (0, 50)
    xlabel, ylabel, fname = "Sample size", "CI length", "../results/real/hiv/3tc/figures/cpl_3tc_20_50_150.pdf"
    util_interactions.plot(data_mlp, data_rf, data_lf, data_ls, data_sf, data_ss, sample_sizes, xlabel, ylabel, fname, ylim)

    ylim = (-2, +2)
    xlabel, ylabel, fname = "Sample size", "r2 score", "../results/real/hiv/3tc/figures/r2_3tc_20_50_150.pdf"
    util_interactions.plot(r2_mlps, r2_rfs, r2_lfs, r2_lss, r2_sfs, r2_sss, sample_sizes, xlabel, ylabel, fname, ylim)


if __name__ == "__main__":
    run()
