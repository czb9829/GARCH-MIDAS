"""
garch-midas model
"""
import os
from ctypes import *
import numpy as np
import pandas as pd
from collections import namedtuple
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def fit_mfgarch(
        data: object,
        y: object,
        x: object,
        low_freq: object,
        K: object,
        var_ratio_freq: object = None,
        dummy: object = None,
        interact_term: object = None,
        gamma: object = False,
        log_tau: object = True,
        m_restricted: object = False,
        theta_restricted: object = None,
        w_restricted: object = None,
        distribution: object = "norm",
        g_zero: object = 1,
        n_horizons: object = 10,
        control: object = None,
        high_range: object = None,
        out_of_sample: object = 0,
        penalty: object = 0) -> object:
    """
    Maximum likelihood estimation of garch-midas model.
    GARCH-MIDAS模型的极大似然估计

    :param data:               Dataframe containing a column named date of type 'datetime'.
                               包含类型为'datetime'的高频日期的dataframe.
    :param y:                  Name of high frequency dependent variable in the dataframe.
                               高频因变量名称.
    :param x:                  Names of low frequency explained variables in the dataframe.
                               低频解释变量名称.
    :param low_freq:           The low frequency in the table.
                               低频日期.
    :param K:                  Integers specifying lag length K in the long-term component.
                               长期成分中的滞后阶数.
    :param var_ratio_freq:     Specify a frequency column on which the variance ratio should be calculated.
                               计算方差比率用的分组频率.
    :param dummy:              Dummy variables in the dateframe.
                               虚拟变量.
    :param interact_term:      Interactive items of dummy variables and weighted low-frequency variables
                               虚拟变量与低频因子的交互项.
    :param gamma:              If equal to True, an asymmetric GJR-GARCH is used as the short-term
                               component. If false, a simple GARCH(1,1) is employed. The default is True.
                               如果该值等于True，则短期成分采用GJR-GARCH形式; 否则采用GARCH(1,1)形式. 默认为True.
    :param log_tau

    :param m_restricted

    :param theta_restricted

    :param w_restricted:       A vector that sets whether the weight function is restricted.
                               The default is True.
                               如果对应的数值为True, 则权重函数会收到限制. 默认为True.
    :param distribution:       The conditional density to use for the innovations. At the moment, valid choices
                               are "norm" and "std", for the Normal and Student-t distribution, respectively
                               条件概率密度函数. "norm" 表示正态分布, "std" 表示学生t分布.
    :param g_zero:             Initial value of short-term volitality. The default is 1.
                               短期波动的初始值. 默认为1.
    :param n_horizons:         Number of periods for multi-step forecasting. The default is 10.
                               多步预测的期数. 默认为10.
    :param control:            A dict of control parameters.
                               数值最优化的设置.
    :param high_range:         Range of valid samples of high-frequency variables.
                               高频样本的范围.
    :param out_of_sample:      A positive integer indicating the number of periods before the
                               last to keep for out of sample forecasting.
                               样本外预测的期数. 预留数据集中最后一定数量的样本做预测.
    :param penalty:            Penalty coefficient of L2-Regularization.
                               L2-正则化的惩罚系数, 类似于岭回归.
    :return:
    """
    # the number of low frequency variables
    if type(x) == str:
        x = [x]
    n_vars = len(x)

    if type(low_freq) == str:
        low_freq = [low_freq]
    else:
        if len(low_freq) != n_vars:
            raise Exception("The lengths of x and low_freq are inconsistent")

    if type(K) == int:
        K = [K for _ in range(n_vars)]
    else:
        if len(K) != n_vars:
            raise Exception("The lengths of x and K are inconsistent")

    if theta_restricted is None:
        theta_restricted = [False for _ in range(n_vars)]
    elif type(theta_restricted) == bool:
        theta_restricted = [theta_restricted for _ in range(n_vars)]
    else:
        if len(theta_restricted) != n_vars:
            raise Exception("The lengths of x and theta_restricted are inconsistent")

    if w_restricted is None:
        w_restricted = [False for _ in range(n_vars)]
    elif type(w_restricted) == bool:
        w_restricted = [w_restricted for _ in range(n_vars)]
    else:
        if len(w_restricted) != n_vars:
            raise Exception("The lengths of x and w_restricted are inconsistent")

    if distribution not in ["norm", "std"]:
        raise Exception("valid choices are 'norm' and 'std'")

    # check and set var_ratio_freq
    if var_ratio_freq is not None:
        if var_ratio_freq not in low_freq:
            raise Exception("check and set var.ratio.freq " + var_ratio_freq + ".")
    else:
        var_ratio_freq = low_freq[0]
        print("No frequency specified for calculating the variance ratio - default: low.freq[0] = " + low_freq[0])
    var_ratio_var = x[low_freq.index(var_ratio_freq)]

    # Order by high frequency variable
    data["date"] = pd.to_datetime(data["date"], infer_datetime_format=True)
    data = data.sort_values(by="date")

    # returns
    ret = data[y].values.tolist()

    nobs = len(ret)
    low_freq_index = []
    low_freq_data = []
    for i in range(n_vars):
        col = low_freq[i]
        freq = data[col].values
        is_dup = np.zeros(nobs, dtype="int32")
        is_dup[1:] = freq[1:] == freq[:-1]
        idx = 1 - is_dup
        idx[0] = 0
        idx = np.cumsum(idx).tolist()
        low_freq_index.append(idx)
        dta = data[x[i]].values[is_dup == 0].tolist()
        low_freq_data.append(dta)

    # dummy variables
    if dummy is None:
        dummy = []
    elif type(dummy) == str:
        dummy = [dummy]
    dummy_data = []
    for var in dummy:
        dta = data[var].values.tolist()
        dummy_data.append(dta)

    # interaction term
    i_data_d_var = []
    interact_x_var = []
    interact_d_var = []
    interact_data = []
    if interact_term is not None:
        interact_x_var = interact_term["x"]
        interact_d_var = interact_term["dummy"]
        if type(interact_x_var) == str:
            interact_x_var = [interact_x_var]
        if type(interact_d_var) == str:
            interact_d_var = [interact_d_var]
        if len(interact_d_var) == 1 and len(interact_x_var) != 1:
            interact_d_var = [interact_d_var[0] for _ in range(len(interact_x_var))]
        if len(interact_x_var) != len(interact_d_var):
            raise Exception("The number of x and dummy are inconsistent")

        i_data_d_var = list(set(interact_d_var))
        for var in i_data_d_var:
            dta = data[var].values.tolist()
            interact_data.append(dta)

    # constrained optimizer settings
    method = "Nelder-Mead"
    mu = 1e-4
    outer_iterations = 100
    outer_eps = 1e-5
    maxit = 500
    if control is not None:
        if "method" in control:
            method = control["method"]
        if "mu" in control:
            mu = control["mu"]
        if "outer_iterations" in control:
            outer_iterations = control["outer_iterations"]
        if "outer_eps" in control:
            outer_eps = control["outer_eps"]
        if "maxit" in control:
            maxit = control["maxit"]

    # other information
    task = 0
    par_start = []

    # high frequency variable range
    if high_range is None:
        high_range = {}
    if "start" not in high_range:
        high_range["start"] = 0
    else:
        if high_range["start"] < 0:
            raise Exception("The start value of the range must not be lower than 1.")

    if "stop" not in high_range:
        high_range["stop"] = nobs
    else:
        if high_range["stop"] > nobs:
            raise Exception("The stop value of range must be greater than the start value of range.")

    high_range = [high_range["start"], high_range["stop"]]

    result = fit_mfgarch_c(
        task,
        ret,
        dummy,
        dummy_data,
        x,
        var_ratio_var,
        low_freq_index,
        low_freq_data,
        i_data_d_var,
        interact_x_var,
        interact_d_var,
        interact_data,
        K,
        gamma,
        log_tau,
        m_restricted,
        theta_restricted,
        w_restricted,
        distribution,
        n_horizons,
        par_start,
        g_zero,
        mu,
        outer_iterations,
        outer_eps,
        method,
        maxit,
        high_range,
        out_of_sample,
        penalty
    )

    mfgarch_tuple = namedtuple("mfgarch", ["estimate", "fitted", "predicted", "llh", "aic", "bic",
                                           "variance_ratio", "est_weight", "mf_forecast"])

    n_fitted = result["n_fitted"]
    data = data.iloc[-n_fitted:, :]

    # estimeate result
    estimate = pd.DataFrame(
        {
            "estimate": result["estimate"],
            "rob.std.err": result["rob_std_err"],
            "p.value": result["rob_p_value"],
            "opg.std.err": result["opg_std_err"],
            "opg.p.value": result["opg_p_value"]
        },
        index=result["coefficient"])

    # fitted
    fitted = pd.DataFrame(
        {
            "date": data["date"],
            y: data[y],
            "low_freq": data[low_freq[0]],
            "var_ratio_freq": data[var_ratio_freq],
            "tau": result["tau"],
            "g": result["g"],
        }
    )
    fitted["vol"] = fitted["tau"] * fitted["g"]

    # predicted
    predicted = None
    if out_of_sample > 0:
        predicted = fitted.iloc[-out_of_sample:, :]
        fitted = fitted.iloc[:-out_of_sample, :]

    # llh, aic and bic
    llh = result["llh"]
    aic = result["aic"]
    bic = result["bic"]

    # variance ratio
    variance_ratio = result["variance_ratio"]

    # estimate weights
    est_weight = result["est_weight"]
    est_weight = pd.DataFrame(est_weight, columns=["est.weight." + s for s in x])

    # mf_forecast
    mf_forecast = result["mf_forecast"]

    mfgarch = mfgarch_tuple(estimate, fitted, predicted, llh, aic, bic, variance_ratio, est_weight, mf_forecast)

    return mfgarch


def fit_mfgarch_c(
        task,
        ret,
        dummy_var,
        dummy_data,
        low_freq_var,
        var_ratio_var,
        low_freq_index,
        low_freq_data,
        i_data_d_var,
        interact_x_var,
        interact_d_var,
        interact_data,
        K,
        gamma,
        log_tau,
        m_restricted,
        theta_restricted,
        w_restricted,
        distribution,
        n_horizons,
        par_start,
        g_zero,
        mu,
        outer_iterations,
        outer_eps,
        method,
        maxit,
        high_range,
        out_of_sample,
        penalty
):
    """
    :param task:
    :param ret:
    :param dummy_var:
    :param dummy_data:
    :param low_freq_var:
    :param var_ratio_var:
    :param low_freq_index:
    :param low_freq_data:
    :param i_data_d_var:
    :param interact_x_var:
    :param interact_d_var:
    :param interact_data:
    :param K:
    :param gamma:
    :param log_tau:
    :param m_restricted:
    :param theta_restricted
    :param w_restricted:
    :param distribution
    :param n_horizons:
    :param par_start:
    :param g_zero:
    :param mu:
    :param outer_iterations:
    :param outer_eps:
    :param method:
    :param maxit:
    :param high_range:
    :param out_of_sample:
    :param penalty:
    :return:
    """
    """**************************** input ****************************"""
    # task
    task_c = c_int(task)
    # number of observations
    nobs = len(ret)
    nobs_c = c_int(nobs)
    # returns
    ret_c = (c_double * nobs)()
    for i, v in enumerate(ret):
        ret_c[i] = v
    # dummy variables
    dummy_var_c = c_char_p(" ".join(dummy_var).encode("utf-8"))
    # number of dummy variables
    n_d_vars = len(dummy_data)
    n_d_vars_c = c_int(n_d_vars)
    # data of dummy variables
    dummy_data = np.array(dummy_data)
    dummy_data = dummy_data.flatten()
    dummy_data_c = (c_double * (n_d_vars * nobs))()
    for i, v in enumerate(dummy_data):
        dummy_data_c[i] = v
    # number of low frequency variables
    n_vars = len(low_freq_data)
    n_vars_c = c_int(n_vars)
    # low frequency variables
    low_freq_var_c = c_char_p(" ".join(low_freq_var).encode("utf-8"))
    # variance ratio variable
    var_ratio_var_c = c_char_p(var_ratio_var.encode("utf-8"))
    # number of low frequency data
    n_low_freq = []
    for v in low_freq_data:
        n_low_freq.append(len(v))
    n_low_freq_c = (c_int * n_vars)()
    for i, v in enumerate(n_low_freq):
        n_low_freq_c[i] = v
    # low frequency index
    low_freq_index = np.array(low_freq_index)
    low_freq_index = low_freq_index.flatten()
    low_freq_index_c = (c_int * (n_vars * nobs))()
    for i, v in enumerate(low_freq_index):
        low_freq_index_c[i] = v
    # low frequency data
    low_freq_data = np.concatenate(low_freq_data)
    n_low_freq_data = low_freq_data.size
    low_freq_data_c = (c_double * n_low_freq_data)()
    for i, v in enumerate(low_freq_data):
        low_freq_data_c[i] = v
    # number of variables of interactive dataset
    n_i_data_vars = len(interact_data)
    n_i_data_vars_c = c_int(n_i_data_vars)
    # number of interaction terms
    n_i_vars = len(interact_x_var)
    n_i_vars_c = c_int(n_i_vars)
    # interactive data
    interact_data = np.array(interact_data)
    interact_data = interact_data.flatten()
    interact_data_c = (c_double * (n_i_data_vars * nobs))()
    for i, v in enumerate(interact_data):
        interact_data_c[i] = v
    # variables of interactive dataset
    i_data_d_var_c = c_char_p(" ".join(i_data_d_var).encode("utf-8"))
    # interactive low frequency variables
    interact_x_var_c = c_char_p(" ".join(interact_x_var).encode("utf-8"))
    # interactive dummy variables
    interact_d_var_c = c_char_p(" ".join(interact_d_var).encode("utf-8"))
    # K
    K_c = (c_int * n_vars)()
    for i, v in enumerate(K):
        K_c[i] = v
    # gamma
    gamma_c = c_int(gamma)
    # log_tau
    log_tau_c = c_int(log_tau)
    # m_restricted
    m_restricted_c = c_int(m_restricted)
    # theta_restricted
    theta_restricted_c = (c_int * n_vars)()
    for i, v in enumerate(theta_restricted):
        theta_restricted_c[i] = v
    # w_restricted
    w_restricted_c = (c_int * n_vars)()
    for i, v in enumerate(w_restricted):
        w_restricted_c[i] = v
    # distribution
    distribution_c = c_char_p(distribution.encode("utf-8"))
    # n_horizons
    n_horizons_c = c_int(n_horizons)
    # par_start
    n_par = len(par_start)
    n_par_c = c_int(n_par)
    par_start_c = (c_double * n_par)()
    for i, v in enumerate(par_start):
        par_start_c[i] = v
    # g_zero
    g_zero_c = c_double(g_zero)
    # mu
    mu_c = c_double(mu)
    # outer_iteractions
    outer_iterations_c = c_int(outer_iterations)
    # outer_eps
    outer_eps_c = c_double(outer_eps)
    # method
    method_c = c_char_p(method.encode("utf-8"))
    # maxit
    maxit_c = c_int(maxit)
    # high_range
    high_range_c = (c_int * 2)()
    high_range_c[0] = high_range[0]
    high_range_c[1] = high_range[1]
    # max lag
    max_lag_c = (c_int * n_vars)()
    # out of sample
    out_of_sample_c = c_int(out_of_sample)
    # penalty
    penalty_c = c_double(penalty)
    # coefficient
    coefficient_c = c_char_p("".encode("utf-8"))
    # estimate and statistical inference
    max_n_par = 3 * n_vars + n_d_vars + n_i_vars + 5
    estimate_c = (c_double * max_n_par)()
    opg_std_err_c = (c_double * max_n_par)()
    opg_p_value_c = (c_double * max_n_par)()
    rob_std_err_c = (c_double * max_n_par)()
    rob_p_value_c = (c_double * max_n_par)()
    # llh, aic and bic
    llh = 0
    aic = 0
    bic = 0
    llh_c = c_double(llh)
    aic_c = c_double(aic)
    bic_c = c_double(bic)
    # n_fitted
    n_fitted = nobs
    n_fitted_c = c_int(n_fitted)
    # high_index
    high_index_c = (c_int * n_fitted)()
    # tau
    tau_c = (c_double * n_fitted)()
    # g
    g_c = (c_double * n_fitted)()
    # residual
    residual_c = (c_double * n_fitted)()
    # max_K
    max_K = max(K)
    # est_weight
    est_weight_c = (c_double * (n_vars * max_K))()
    # tau_forecast
    tau_forecast = 0
    tau_forecast_c = c_double(tau_forecast)
    # mf_forecast
    mf_forecast_c = (c_double * n_horizons)()
    # variance_ratio
    variance_ratio = 0
    variance_ratio_c = c_double(variance_ratio)
    # n_msg
    n_msg = 5
    n_msg_c = c_int(n_msg)
    # messege
    message_c = (c_int * n_msg)()
    # passed
    passed = 0
    passed_c = c_int(passed)
    # err_msg
    err_msg_c = c_char_p("".encode("utf-8"))

    """**************************** output ****************************"""
    gotsa = os.getenv("GOTSA")
    path = os.path.join(gotsa, "library/mfgarch/mfgarch.dll")
    lib = CDLL(path)
    lib.FitMfgarch(
        byref(task_c),
        byref(nobs_c),
        byref(ret_c),
        byref(n_d_vars_c),
        byref(dummy_var_c),
        byref(dummy_data_c),
        byref(n_vars_c),
        byref(low_freq_var_c),
        byref(var_ratio_var_c),
        byref(n_low_freq_c),
        byref(low_freq_index_c),
        byref(low_freq_data_c),
        byref(n_i_data_vars_c),
        byref(n_i_vars_c),
        byref(interact_data_c),
        byref(i_data_d_var_c),
        byref(interact_x_var_c),
        byref(interact_d_var_c),
        byref(K_c),
        byref(gamma_c),
        byref(distribution_c),
        byref(log_tau_c),
        byref(m_restricted_c),
        byref(theta_restricted_c),
        byref(w_restricted_c),
        byref(n_horizons_c),
        byref(n_par_c),
        byref(par_start_c),
        byref(g_zero_c),
        byref(mu_c),
        byref(outer_iterations_c),
        byref(outer_eps_c),
        byref(method_c),
        byref(maxit_c),
        byref(high_range_c),
        byref(max_lag_c),
        byref(out_of_sample_c),
        byref(penalty_c),
        byref(coefficient_c),
        byref(estimate_c),
        byref(opg_std_err_c),
        byref(opg_p_value_c),
        byref(rob_std_err_c),
        byref(rob_p_value_c),
        byref(llh_c),
        byref(aic_c),
        byref(bic_c),
        byref(n_fitted_c),
        byref(high_index_c),
        byref(tau_c),
        byref(g_c),
        byref(residual_c),
        byref(est_weight_c),
        byref(tau_forecast_c),
        byref(mf_forecast_c),
        byref(variance_ratio_c),
        byref(n_msg_c),
        byref(message_c),
        byref(passed_c),
        byref(err_msg_c)
    )
    passed = passed_c.value
    if passed == 0:
        raise Exception("Please email happyzhichengzhao@163.com to obtain the licence!")

    err_msg = err_msg_c.value
    if len(err_msg) != 0:
        raise Exception(err_msg)

    n_par = n_par_c.value
    coefficient = coefficient_c.value.split()
    coefficient = [s.decode("utf-8") for s in coefficient]
    estimate = list(estimate_c)
    opg_std_err = list(opg_std_err_c)
    opg_p_value = list(opg_p_value_c)
    rob_std_err = list(rob_std_err_c)
    rob_p_value = list(rob_p_value_c)
    n_fitted = n_fitted_c.value
    high_index = list(high_index_c)
    tau = list(tau_c)
    g = list(g_c)
    residual = list(residual_c)
    est_weight = np.array(list(est_weight_c)).reshape((-1, n_vars))
    mf_forecast = list(mf_forecast_c)
    n_msg = n_msg_c.value
    message = list(message_c)
    message = message[:n_msg]

    retsult = {
        "coefficient": coefficient,
        "estimate": estimate[:n_par],
        "opg_std_err": opg_std_err[:n_par],
        "opg_p_value": opg_p_value[:n_par],
        "rob_std_err": rob_std_err[:n_par],
        "rob_p_value": rob_p_value[:n_par],
        "n_fitted": n_fitted,
        "high_index": high_index[:n_fitted],
        "tau": tau[:n_fitted],
        "g": g[:n_fitted],
        "residual": residual[:n_fitted],
        "llh": llh_c.value,
        "aic": aic_c.value,
        "bic": bic_c.value,
        "variance_ratio": variance_ratio_c.value,
        "est_weight": est_weight,
        "mf_forecast": mf_forecast
    }

    return retsult


def optimal_lag_order(
        data,
        y,
        x,
        low_freq,
        var_lags,
        dummy=None,
        interact_term=None,
        gamma=False,
        log_tau=True,
        m_restricted=False,
        theta_restricted=None,
        w_restricted=None,
        distribution="norm",
        g_zero=1,
        control=None,
        n_kernel=4,
        out_of_sample=0):
    """
    Optimal lag order selection for garch-midas model

    :param data:              Dateframe containing a column named date of type 'datetime'.
    :param y:                 Name of high frequency dependent variable in the dateframe.
    :param x:                 Names of low frequency explained variables in the dateframe.
    :param low_freq:          The low frequency variables in the table.
    :param var_lags:          The combination of lag orders that need to be compared.
    :param dummy:             Dummy variables in the table.
    :param interact_term:     Interactive items of dummy variables and weighted low-frequency variables
    :param gamma:             If equal to True, an asymmetric GJR-GARCH is used as the short-term
                              component. If FALSE, a simple GARCH(1,1) is employed. The default is True.
    :param log_tau:
    :param m_restricted:
    :param theta_restricted:
    :param w_restricted:      A vector that sets whether the weight function is restricted.
                              The default is True.
    :param distribution:
    :param g_zero:            Initial value of short-term volitality. The default is 1.
    :param control:           A list of control parameters.
    :param n_kernel:          # of kernels
    :param out_of_sample      A positive integer indicating the number of periods before the
                              last to keep for out of sample forecasting.
    :return:
    """
    # the number of low frequency variables
    if type(x) == str:
        x = [x]
    n_vars = len(x)

    if type(low_freq) == str:
        low_freq = [low_freq]
    else:
        if len(low_freq) != n_vars:
            raise Exception("The lengths of x and low_freq are inconsistent")

    if theta_restricted is None:
        theta_restricted = [False for _ in range(n_vars)]
    elif type(theta_restricted) == bool:
        theta_restricted = [theta_restricted for _ in range(n_vars)]
    else:
        if len(theta_restricted) != n_vars:
            raise Exception("The lengths of x and theta_restricted are inconsistent")

    if w_restricted is None:
        w_restricted = [False for _ in range(n_vars)]
    elif type(w_restricted) == bool:
        w_restricted = [w_restricted for _ in range(n_vars)]
    else:
        if len(w_restricted) != n_vars:
            raise Exception("The lengths of x and w_restricted are inconsistent")

    if distribution not in ["norm", "std"]:
        raise Exception("valid choices are 'norm' and 'std'")

    # order by high frequency variable
    data["date"] = pd.to_datetime(data["date"], infer_datetime_format=True)
    data = data.sort_values(by="date")

    # returns
    ret = data[y].values.tolist()

    nobs = len(ret)
    low_freq_index = []
    low_freq_data = []
    lags = []
    for i in range(n_vars):
        col = low_freq[i]
        freq = data[col].values
        is_dup = np.zeros(nobs, dtype="int32")
        is_dup[1:] = freq[1:] == freq[:-1]
        idx = 1 - is_dup
        idx[0] = 0
        idx = np.cumsum(idx).tolist()
        low_freq_index.append(idx)
        dta = data[x[i]].values[is_dup == 0].tolist()
        low_freq_data.append(dta)
        lags.append(var_lags[x[i]])

    # dummy variables
    if dummy is None:
        dummy = []
    elif type(dummy) == str:
        dummy = [dummy]
    dummy_data = []
    for var in dummy:
        dta = data[var].values.tolist()
        dummy_data.append(dta)

    # interaction term
    i_data_d_var = []
    interact_x_var = []
    interact_d_var = []
    interact_data = []
    if interact_term is not None:
        interact_x_var = interact_term["x"]
        interact_d_var = interact_term["dummy"]
        if type(interact_x_var) == str:
            interact_x_var = [interact_x_var]
        if type(interact_d_var) == str:
            interact_d_var = [interact_d_var]
        if len(interact_d_var) == 1 and len(interact_x_var) != 1:
            interact_d_var = [interact_d_var[0] for _ in range(len(interact_x_var))]
        if len(interact_x_var) != len(interact_d_var):
            raise Exception("The number of x and dummy are inconsistent")

        i_data_d_var = list(set(interact_d_var))
        for var in i_data_d_var:
            dta = data[var].values.tolist()
            interact_data.append(dta)

    # constrained optimizer settings
    method = "Nelder-Mead"
    mu = 1e-4
    outer_iterations = 100
    outer_eps = 1e-5
    maxit = 500
    if control is not None:
        if "method" in control:
            method = control["method"]
        if "mu" in control:
            mu = control["mu"]
        if "outer_iterations" in control:
            outer_iterations = control["outer_iterations"]
        if "outer_eps" in control:
            outer_eps = control["outer_eps"]
        if "maxit" in control:
            maxit = control["maxit"]

    # other information
    task = 0
    par_start = []

    result = optimal_lag_order_c(
        ret,
        dummy,
        dummy_data,
        x,
        low_freq_index,
        low_freq_data,
        i_data_d_var,
        interact_x_var,
        interact_d_var,
        interact_data,
        lags,
        gamma,
        log_tau,
        m_restricted,
        theta_restricted,
        w_restricted,
        distribution,
        par_start,
        g_zero,
        mu,
        outer_iterations,
        outer_eps,
        method,
        maxit,
        n_kernel,
        out_of_sample
    )

    # output
    llh = result["llh"]
    aic = result["aic"]
    bic = result["bic"]
    aic_opt_id = np.argmin(aic)
    bic_opt_id = np.argmin(bic)
    aic_opt_lag = {k: v[aic_opt_id] for k, v in var_lags.items()}
    bic_opt_lag = {k: v[bic_opt_id] for k, v in var_lags.items()}
    AIC_opt = namedtuple("AIC_opt", ["AIC", "lag"])
    BIC_opt = namedtuple("BIC_opt", ["BIC", "lag"])
    opt_lag_order = namedtuple("opt_lag_order", ["var_lags", "llh", "AIC", "BIC", "AIC_opt", "BIC_opt"])
    aic_opt = AIC_opt(aic[aic_opt_id], aic_opt_lag)
    bic_opt = BIC_opt(bic[bic_opt_id], bic_opt_lag)
    output = opt_lag_order(var_lags, llh, aic, bic, aic_opt, bic_opt)
    return output


def optimal_lag_order_c(
        ret,
        dummy_var,
        dummy_data,
        low_freq_var,
        low_freq_index,
        low_freq_data,
        i_data_d_var,
        interact_x_var,
        interact_d_var,
        interact_data,
        lags,
        gamma,
        log_tau,
        m_restricted,
        theta_restricted,
        w_restricted,
        distribution,
        par_start,
        g_zero,
        mu,
        outer_iterations,
        outer_eps,
        method,
        maxit,
        n_kernel,
        out_of_sample
):
    """
    :param ret:
    :param dummy_var:
    :param dummy_data:
    :param low_freq_var:
    :param low_freq_index:
    :param low_freq_data:
    :param i_data_d_var:
    :param interact_x_var:
    :param interact_d_var:
    :param interact_data:
    :param lags:
    :param gamma:
    :param log_tau:
    :param m_restricted:
    :param theta_restricted,
    :param w_restricted:
    :param distribution:
    :param par_start:
    :param g_zero:
    :param mu:
    :param outer_iterations:
    :param outer_eps:
    :param method:
    :param maxit:
    :param n_kernel
    :param out_of_sample:
    :return:
    """
    """**************************** input ****************************"""
    # number of observations
    nobs = len(ret)
    nobs_c = c_int(nobs)
    # returns
    ret_c = (c_double * nobs)()
    for i, v in enumerate(ret):
        ret_c[i] = v
    # dummy variables
    dummy_var_c = c_char_p(" ".join(dummy_var).encode("utf-8"))
    # number of dummy variables
    n_d_vars = len(dummy_data)
    n_d_vars_c = c_int(n_d_vars)
    # data of dummy variables
    dummy_data = np.array(dummy_data)
    dummy_data = dummy_data.flatten()
    dummy_data_c = (c_double * (n_d_vars * nobs))()
    for i, v in enumerate(dummy_data):
        dummy_data_c[i] = v
    # number of low frequency variables
    n_vars = len(low_freq_data)
    n_vars_c = c_int(n_vars)
    # low frequency variables
    low_freq_var_c = c_char_p(" ".join(low_freq_var).encode("utf-8"))
    # number of low frequency data
    n_low_freq = []
    for v in low_freq_data:
        n_low_freq.append(len(v))
    n_low_freq_c = (c_int * n_vars)()
    for i, v in enumerate(n_low_freq):
        n_low_freq_c[i] = v
    # low frequency index
    low_freq_index = np.array(low_freq_index)
    low_freq_index = low_freq_index.flatten()
    low_freq_index_c = (c_int * (n_vars * nobs))()
    for i, v in enumerate(low_freq_index):
        low_freq_index_c[i] = v
    # low frequency data
    low_freq_data = np.concatenate(low_freq_data)
    n_low_freq_data = low_freq_data.size
    low_freq_data_c = (c_double * n_low_freq_data)()
    for i, v in enumerate(low_freq_data):
        low_freq_data_c[i] = v
    # number of variables of interactive dataset
    n_i_data_vars = len(interact_data)
    n_i_data_vars_c = c_int(n_i_data_vars)
    # number of interaction terms
    n_i_vars = len(interact_x_var)
    n_i_vars_c = c_int(n_i_vars)
    # interactive data
    interact_data = np.array(interact_data)
    interact_data = interact_data.flatten()
    interact_data_c = (c_double * (n_i_data_vars * nobs))()
    for i, v in enumerate(interact_data):
        interact_data_c[i] = v
    # variables of interactive dataset
    i_data_d_var_c = c_char_p(" ".join(i_data_d_var).encode("utf-8"))
    # interactive low frequency variables
    interact_x_var_c = c_char_p(" ".join(interact_x_var).encode("utf-8"))
    # interactive dummy variables
    interact_d_var_c = c_char_p(" ".join(interact_d_var).encode("utf-8"))
    # lags
    n_lags = len(lags[0])
    n_lags_c = c_int(n_lags)
    lags = np.array(lags)
    lags = lags.flatten()
    lags_c = (c_int * (n_vars * n_lags))()
    for i, v in enumerate(lags):
        lags_c[i] = v
    # gamma
    gamma_c = c_int(gamma)
    # log_tau
    log_tau_c = c_int(log_tau)
    # m_restricted
    m_restricted_c = c_int(m_restricted)
    # theta_restricted
    theta_restricted_c = (c_int * n_vars)()
    for i, v in enumerate(theta_restricted):
        theta_restricted_c[i] = v
    # w_restricted
    w_restricted_c = (c_int * n_vars)()
    for i, v in enumerate(w_restricted):
        w_restricted_c[i] = v
    # distribution
    distribution_c = c_char_p(distribution.encode("utf-8"))
    # par_start
    n_par = len(par_start)
    n_par_c = c_int(n_par)
    par_start_c = (c_double * n_par)()
    for i, v in enumerate(par_start):
        par_start_c[i] = v
    # g_zero
    g_zero_c = c_double(g_zero)
    # mu
    mu_c = c_double(mu)
    # outer_iteractions
    outer_iterations_c = c_int(outer_iterations)
    # outer_eps
    outer_eps_c = c_double(outer_eps)
    # method
    method_c = c_char_p(method.encode("utf-8"))
    # maxit
    maxit_c = c_int(maxit)
    # number of kernels
    n_kernel_c = c_int(n_kernel)
    # out of sample
    out_of_sample_c = c_int(out_of_sample)
    # llh, aic and bic
    llh_c = (c_double * n_lags)()
    aic_c = (c_double * n_lags)()
    bic_c = (c_double * n_lags)()

    """**************************** output ****************************"""

    gotsa = os.getenv("GOTSA")
    path = os.path.join(gotsa, "library/mfgarch/mfgarch.dll")
    lib = CDLL(path)
    lib.OptimalLagOrder(
        byref(nobs_c),
        byref(ret_c),
        byref(n_d_vars_c),
        byref(dummy_var_c),
        byref(dummy_data_c),
        byref(n_vars_c),
        byref(low_freq_var_c),
        byref(n_low_freq_c),
        byref(low_freq_index_c),
        byref(low_freq_data_c),
        byref(n_i_data_vars_c),
        byref(n_i_vars_c),
        byref(interact_data_c),
        byref(i_data_d_var_c),
        byref(interact_x_var_c),
        byref(interact_d_var_c),
        byref(n_lags_c),
        byref(lags_c),
        byref(gamma_c),
        byref(distribution_c),
        byref(log_tau_c),
        byref(m_restricted_c),
        byref(theta_restricted_c),
        byref(w_restricted_c),
        byref(n_par_c),
        byref(par_start_c),
        byref(g_zero_c),
        byref(mu_c),
        byref(outer_iterations_c),
        byref(outer_eps_c),
        byref(method_c),
        byref(maxit_c),
        byref(out_of_sample_c),
        byref(n_kernel_c),
        byref(llh_c),
        byref(aic_c),
        byref(bic_c)
    )
    llh = list(llh_c)
    aic = list(aic_c)
    bic = list(bic_c)

    retsult = {
        "llh": llh,
        "aic": aic,
        "bic": bic,
    }

    return retsult


def rolling_predict(
        data,
        y,
        x,
        low_freq,
        K,
        high_start,
        high_end,
        n_horizons=1,
        dummy=None,
        interact_term=None,
        gamma=False,
        log_tau=True,
        m_restricted=False,
        theta_restricted=None,
        w_restricted=None,
        distribution="norm",
        g_zero=1,
        n_kernels=4,
        trace=1,
        control=None):
    """
    Rolling predictions for garch-midas model

    :param data:              Dateframe containing a column named date of type 'datetime'.
    :param y:                 Name of high frequency dependent variable in the dateframe.
    :param x:                 Names of low frequency explained variables in the dateframe.
    :param low_freq:          The low frequency variables in the table.
    :param K:                 Integers specifying lag length K in the long-term component.
    :param high_start:        The indexes at the beginning of the time window.
    :param high_end:          The indexes at the beginning of the time window.
    :param n_horizons:        Number of periods for multi-step forecasting. The default is 1.
    :param dummy:             Dummy variables in the table.
    :param interact_term:     Interactive items of dummy variables and weighted low-frequency variables.
    :param gamma:             If equal to True, an asymmetric GJR-GARCH is used as the short-term
                              component. If FALSE, a simple GARCH(1,1) is employed. The default is True.
    :param log_tau:
    :param m_restricted:
    :param theta_restricted:
    :param w_restricted:      A vector that sets whether the weight function is restricted.
                              The default is True.
    :param distribution:
    :param g_zero:            Initial value of short-term volitality. The default is 1.
    :param n_kernels:         The number of kernels in parallel computing.
    :param trace:             If True, tracing information on the progress of rolling predictions is produced.
                              The default is True.
    :param control:           A list of control parameters.
    :return:
    """
    # the number of low frequency variables
    if type(x) == str:
        x = [x]
    n_vars = len(x)

    if type(low_freq) == str:
        low_freq = [low_freq]
    else:
        if len(low_freq) != n_vars:
            raise Exception("The lengths of x and low_freq are inconsistent")

    if type(K) == int:
        K = [K for _ in range(n_vars)]
    else:
        if len(K) != n_vars:
            raise Exception("The lengths of x and K are inconsistent")

    if theta_restricted is None:
        theta_restricted = [False for _ in range(n_vars)]
    elif type(theta_restricted) == bool:
        theta_restricted = [theta_restricted for _ in range(n_vars)]
    else:
        if len(theta_restricted) != n_vars:
            raise Exception("The lengths of x and theta_restricted are inconsistent")

    if w_restricted is None:
        w_restricted = [False for _ in range(n_vars)]
    elif type(w_restricted) == bool:
        w_restricted = [w_restricted for _ in range(n_vars)]
    else:
        if len(w_restricted) != n_vars:
            raise Exception("The lengths of x and w_restricted are inconsistent")

    if distribution not in ["norm", "std"]:
        raise Exception("valid choices are 'norm' and 'std'")

    # check the length of high_start and high_end
    if len(high_start) != len(high_end):
        raise Exception("The length of high_start and high_end must be equal.")

    # Order by high frequency variable
    data["date"] = pd.to_datetime(data["date"], infer_datetime_format=True)
    data = data.sort_values(by="date")

    # returns
    ret = data[y].values.tolist()

    nobs = len(ret)
    low_freq_index = []
    low_freq_data = []
    for i in range(n_vars):
        col = low_freq[i]
        freq = data[col].values
        is_dup = np.zeros(nobs, dtype="int32")
        is_dup[1:] = freq[1:] == freq[:-1]
        idx = 1 - is_dup
        idx[0] = 0
        idx = np.cumsum(idx).tolist()
        low_freq_index.append(idx)
        dta = data[x[i]].values[is_dup == 0].tolist()
        low_freq_data.append(dta)

    # dummy variables
    if dummy is None:
        dummy = []
    elif type(dummy) == str:
        dummy = [dummy]
    dummy_data = []
    for var in dummy:
        dta = data[var].values.tolist()
        dummy_data.append(dta)

    # interaction term
    i_data_d_var = []
    interact_x_var = []
    interact_d_var = []
    interact_data = []
    if interact_term is not None:
        interact_x_var = interact_term["x"]
        interact_d_var = interact_term["dummy"]
        if type(interact_x_var) == str:
            interact_x_var = [interact_x_var]
        if type(interact_d_var) == str:
            interact_d_var = [interact_d_var]
        if len(interact_d_var) == 1 and len(interact_x_var) != 1:
            interact_d_var = [interact_d_var[0] for _ in range(len(interact_x_var))]
        if len(interact_x_var) != len(interact_d_var):
            raise Exception("The number of x and dummy are inconsistent")

        i_data_d_var = list(set(interact_d_var))
        for var in i_data_d_var:
            dta = data[var].values.tolist()
            interact_data.append(dta)

    # constrained optimizer settings
    method = "Nelder-Mead"
    mu = 1e-4
    outer_iterations = 100
    outer_eps = 1e-5
    maxit = 500
    if control is not None:
        if "method" in control:
            method = control["method"]
        if "mu" in control:
            mu = control["mu"]
        if "outer_iterations" in control:
            outer_iterations = control["outer_iterations"]
        if "outer_eps" in control:
            outer_eps = control["outer_eps"]
        if "maxit" in control:
            maxit = control["maxit"]

    result = rolling_predict_c(
        ret,
        dummy,
        dummy_data,
        x,
        low_freq_index,
        low_freq_data,
        i_data_d_var,
        interact_x_var,
        interact_d_var,
        interact_data,
        K,
        gamma,
        log_tau,
        m_restricted,
        theta_restricted,
        w_restricted,
        distribution,
        g_zero,
        mu,
        outer_iterations,
        outer_eps,
        method,
        maxit,
        high_start,
        high_end,
        n_horizons,
        n_kernels,
        trace
    )

    high_start = np.array(high_start)
    high_end = np.array(high_end)
    mf_forecast = result["mf_forecast"]
    mf_forecast = pd.DataFrame(mf_forecast, columns=["step%d" % (1 + i) for i in range(n_horizons)])
    predicted_df = pd.DataFrame(
        {
            "date": data["date"].values[high_end]
        }
    )
    predicted_df = pd.concat([predicted_df, mf_forecast], axis=1)
    high_start = data["date"].values[high_start]
    high_end = data["date"].values[high_end]
    predict = namedtuple("mfgarch_predict", ["high_start", "high_end", "predicted_df"])
    output = predict(high_start, high_end, predicted_df)

    return output


def rolling_predict_c(
        ret,
        dummy_var,
        dummy_data,
        low_freq_var,
        low_freq_index,
        low_freq_data,
        i_data_d_var,
        interact_x_var,
        interact_d_var,
        interact_data,
        K,
        gamma,
        log_tau,
        m_restricted,
        theta_restricted,
        w_restricted,
        distribution,
        g_zero,
        mu,
        outer_iterations,
        outer_eps,
        method,
        maxit,
        high_start,
        high_end,
        n_horizons,
        n_kernels,
        trace
):
    """
    :param ret:
    :param dummy_var:
    :param dummy_data:
    :param low_freq_var:
    :param low_freq_index:
    :param low_freq_data:
    :param i_data_d_var:
    :param interact_x_var:
    :param interact_d_var:
    :param interact_data:
    :param K:
    :param gamma:
    :param log_tau:
    :param m_restricted:
    :param theta_restricted,
    :param w_restricted:
    :param distribution:
    :param g_zero:
    :param mu:
    :param outer_iterations:
    :param outer_eps:
    :param method:
    :param maxit:
    :param high_start:
    :param high_end:
    :param n_horizons:
    :param n_kernels:
    :param trace:
    :return:
    """
    """**************************** input ****************************"""
    # number of observations
    nobs = len(ret)
    nobs_c = c_int(nobs)
    # returns
    ret_c = (c_double * nobs)()
    for i, v in enumerate(ret):
        ret_c[i] = v
    # dummy variables
    dummy_var_c = c_char_p(" ".join(dummy_var).encode("utf-8"))
    # number of dummy variables
    n_d_vars = len(dummy_data)
    n_d_vars_c = c_int(n_d_vars)
    # data of dummy variables
    dummy_data = np.array(dummy_data)
    dummy_data = dummy_data.flatten()
    dummy_data_c = (c_double * (n_d_vars * nobs))()
    for i, v in enumerate(dummy_data):
        dummy_data_c[i] = v
    # number of low frequncy variables
    n_vars = len(low_freq_data)
    n_vars_c = c_int(n_vars)
    # low frequancy variables
    low_freq_var_c = c_char_p(" ".join(low_freq_var).encode("utf-8"))
    # number of low frequency data
    n_low_freq = []
    for v in low_freq_data:
        n_low_freq.append(len(v))
    n_low_freq_c = (c_int * n_vars)()
    for i, v in enumerate(n_low_freq):
        n_low_freq_c[i] = v
    # low frequency index
    low_freq_index = np.array(low_freq_index)
    low_freq_index = low_freq_index.flatten()
    low_freq_index_c = (c_int * (n_vars * nobs))()
    for i, v in enumerate(low_freq_index):
        low_freq_index_c[i] = v
    # low frequency data
    low_freq_data = np.concatenate(low_freq_data)
    n_low_freq_data = low_freq_data.size
    low_freq_data_c = (c_double * n_low_freq_data)()
    for i, v in enumerate(low_freq_data):
        low_freq_data_c[i] = v
    # number of variables of interactive dataset
    n_i_data_vars = len(interact_data)
    n_i_data_vars_c = c_int(n_i_data_vars)
    # number of interaction terms
    n_i_vars = len(interact_x_var)
    n_i_vars_c = c_int(n_i_vars)
    # interactive data
    interact_data = np.array(interact_data)
    interact_data = interact_data.flatten()
    interact_data_c = (c_double * (n_i_data_vars * nobs))()
    for i, v in enumerate(interact_data):
        interact_data_c[i] = v
    # variables of interactive dataset
    i_data_d_var_c = c_char_p(" ".join(i_data_d_var).encode("utf-8"))
    # interactive dummy variables
    interact_x_var_c = c_char_p(" ".join(interact_x_var).encode("utf-8"))
    # interactive low frequency variables
    interact_d_var_c = c_char_p(" ".join(interact_d_var).encode("utf-8"))
    # K
    K_c = (c_int * n_vars)()
    for i, v in enumerate(K):
        K_c[i] = v
    # gamma
    gamma_c = c_int(gamma)
    # log_tau
    log_tau_c = c_int(log_tau)
    # m_restricted
    m_restricted_c = c_int(m_restricted)
    # theta_restricted
    theta_restricted_c = (c_int * n_vars)()
    for i, v in enumerate(theta_restricted):
        theta_restricted_c[i] = v
    # w_restricted
    w_restricted_c = (c_int * n_vars)()
    for i, v in enumerate(w_restricted):
        w_restricted_c[i] = v
    # distribution
    distribution_c = c_char_p(distribution.encode("utf-8"))
    # g_zero
    g_zero_c = c_double(g_zero)
    # mu
    mu_c = c_double(mu)
    # outer_iteractions
    outer_iterations_c = c_int(outer_iterations)
    # outer_eps
    outer_eps_c = c_double(outer_eps)
    # method
    method_c = c_char_p(method.encode("utf-8"))
    # maxit
    maxit_c = c_int(maxit)
    # high_start & high_end
    n_forecast = len(high_start)
    n_forecast_c = c_int(n_forecast)
    high_start_c = (c_int * n_forecast)()
    for i, v in enumerate(high_start):
        high_start_c[i] = v
    high_end_c = (c_int * n_forecast)()
    for i, v in enumerate(high_end):
        high_end_c[i] = v
    # n_horizons
    n_horizons_c = c_int(n_horizons)
    # number of kernels
    n_kernels_c = c_int(n_kernels)
    # whether to print intermediate results
    trace_c = c_int(trace)
    # mf_forecast
    mf_forecast_c = (c_double * (n_forecast * n_horizons))()

    """**************************** output ****************************"""
    gotsa = os.getenv("GOTSA")
    path = os.path.join(gotsa, "library/mfgarch/mfgarch.dll")
    lib = CDLL(path)
    lib.RollingForecast(
        byref(nobs_c),
        byref(ret_c),
        byref(n_d_vars_c),
        byref(dummy_var_c),
        byref(dummy_data_c),
        byref(n_vars_c),
        byref(low_freq_var_c),
        byref(n_low_freq_c),
        byref(low_freq_index_c),
        byref(low_freq_data_c),
        byref(n_i_data_vars_c),
        byref(n_i_vars_c),
        byref(interact_data_c),
        byref(i_data_d_var_c),
        byref(interact_x_var_c),
        byref(interact_d_var_c),
        byref(K_c),
        byref(gamma_c),
        byref(distribution_c),
        byref(log_tau_c),
        byref(m_restricted_c),
        byref(theta_restricted_c),
        byref(w_restricted_c),
        byref(g_zero_c),
        byref(mu_c),
        byref(outer_iterations_c),
        byref(outer_eps_c),
        byref(method_c),
        byref(maxit_c),
        byref(n_horizons_c),
        byref(n_forecast_c),
        byref(high_start_c),
        byref(high_end_c),
        byref(n_kernels_c),
        byref(trace_c),
        byref(mf_forecast_c)
    )
    mf_forecast = np.array(list(mf_forecast_c))
    mf_forecast = mf_forecast.reshape((n_forecast, n_horizons))
    retsult = {
        "mf_forecast": mf_forecast
    }
    return retsult


def plot_mfgarch(result, is_sqrt=False):
    """
    :param result:
    :param is_sqrt: 波动率是否开根号
    :return:
    """
    fitted = result.fitted[['low_freq', 'tau', 'g']]
    fitted = fitted.groupby("low_freq", as_index=False).mean()
    fitted["vol"] = fitted["tau"] * fitted["g"]
    fitted["low_freq"] = pd.to_datetime(fitted["low_freq"], infer_datetime_format=True)
    fitted = fitted.sort_values(by="low_freq")
    tau = fitted["tau"].values
    vol = fitted["vol"].values
    labels = ["secular components:τ^0.5", "conditional volatility:τ*g"]
    if is_sqrt:
        tau = np.sqrt(tau)
        vol = np.sqrt(vol)
        labels[1] = "conditional volatility:(τ*g)^0.5"
    plt.plot(fitted["low_freq"], tau)
    plt.plot(fitted["low_freq"], vol, linestyle=':')
    plt.xlabel("Year")
    plt.ylabel("Volatility")
    plt.legend(labels, fontsize=8)
    plt.show()
