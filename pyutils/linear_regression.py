import warnings
warnings.filterwarnings("ignore", message="LinearRegressionwithErrors requires PyMC3")


from astroML.linear_model import TLS_logL
# from astroML.linear_model.total_least_squares import TLS_logL

import numpy as np
from scipy import optimize
import scipy.odr as odr
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

def _OLS_sklearn(x, y, x_predict=None, return_stats=False):
    """
    Perform linear regression using scikit-learn's LinearRegression, with x as the independent variable and y as the dependent variable.
    If x_predict is None, return slope, intercept, and R2 value.
    If x_predict is an array, use the regression to predict the values of y at the provided values.
    If x_predict is provided AND return_stats is True, return a tuple of the slope, intercept, and R2 value, as well as the predicted y-values.
    """

    X = x.reshape((-1,1))
    reg = LinearRegression(fit_intercept=True).fit(X, y)
    slope, intercept, r2 = reg.coef_[0], reg.intercept_, reg.score(X, y)

    if x_predict is None:
        return (slope, intercept, r2)
    else:
        X_predict = x_predict.reshape((-1,1))
        y_predict = reg.predict(X_predict)
        if return_stats:
            return (slope, intercept, r2), y_predict
        else:
            return y_predict
        
def _OLS_statsmodels(x, y, x_predict=None, return_stats=False):
    """
    Perform linear regression using statmodels' OLS, with x as the independent variable and y as the dependent variable.
    If x_predict is None, return slope, intercept, and R2 value.
    If x_predict is an array, use the regression to predict the values of y at the provided values.
    If x_predict is provided AND return_stats is True, return a tuple of the slope, intercept, and R2 value, as well as the predicted y-values.
    """
    
    X = x.reshape((-1,1))
    X2 = sm.add_constant(X)
    model = sm.OLS(y, X2).fit()
    intercept, slope = model.params
    r2 = model.rsquared
    pval = model.pvalues

    if x_predict is None:
        return (slope, intercept, r2, pval)
    else:
        X_predict = x_predict.reshape((-1,1))
        X_predict2 = sm.add_constant(X_predict)
        y_predict = model.predict(X_predict2)
        if return_stats:
            return (slope, intercept, r2, pval), y_predict
        else:
            return y_predict


def OLS(*args, **kwargs):
    """
    Perform ordinary least squares (OLS) linear regression. If passed a single array, treats the first column as the independent variable and the second column as the dependent variable.
    If passed two arrays, treats the first as the independent variable and the second as the dependent variable.
    """

    if len(args) == 1:
        if args[0].shape[1] == 2:
            x, y = tuple(args[0].T)
        elif args[0].shape[1] == 4:
            x, y, xerr, yerr = tuple(args[0].T)
    elif len(args) == 2:
        x, y = args[0], args[1]
    else:
        print("More than two arguments passed.")
    
    x, y = np.array(x), np.array(y)

    if kwargs.get("type") == "sklearn":
        kwargs.pop("type")
        return _OLS_sklearn(x, y, **kwargs)
    elif kwargs.get("type") == "statsmodels":
        kwargs.pop("type")
        return _OLS_statsmodels(x, y, **kwargs)
    else:
        print("Type not specified or not recognized.")
        return
    

def linear_model(p, x):
    slope, intercept = p
    return slope * x + intercept

def ODR(x, y, xerr, yerr, x_predict=None, return_stats=False):
    """
    Perform orthogonal distance regression using scipy's ODR.
    """
    
    x, y = np.array(x), np.array(y)
    # print(x, y)

    data = odr.RealData(x, y, sx=xerr, sy=yerr)

    model = odr.Model(linear_model)
    odr_obj = odr.ODR(data, model, beta0=[0., 1.])
    
    output = odr_obj.run()

    if x_predict is None:
        return tuple(output.beta)
    else:
        y_predict = linear_model(output.beta, x_predict)
        if return_stats:
            return tuple(output.beta), y_predict
        else:
            return y_predict

# translate between typical slope-intercept representation,
# and the normal vector representation
def get_m_b(beta):
    b = np.dot(beta, beta) / beta[1]
    m = -beta[0] / beta[1]
    return m, b

def TLS(*args, x_predict=None, return_stats=False):
    """
    Perform total least squares (TLS) linear regression using astroML.

    This is based on the astroML demo at: https://www.astroml.org/book_figures/chapter8/fig_total_least_squares.html
    which is in turn based on Hogg Bovy and Lang (2010) https://arxiv.org/pdf/1008.4686.pdf, Exercise 13 / Fig. 9.
    """
    
    if len(args) == 1:
        x, y, xerr, yerr = tuple(args[0].T)
        X = np.array([x, y]).T
        dX = np.array([ [[sx,0],[0,sy]] for sx, sy in zip(xerr,yerr) ])
    elif len(args) == 2:
        X, dX = args[0], args[1]
    elif len(args) == 4:
        x, y, xerr, yerr = args[0], args[1], args[2], args[3]
        X = np.array([x, y]).T
        dX = np.array([ [[sx,0],[0,sy]] for sx, sy in zip(xerr,yerr) ])
    else:
        print("Invalid number of arguments passed.")

    min_func = lambda beta: -TLS_logL(beta, X, dX)
    beta_fit = optimize.fmin(min_func, x0=[-1, 1], disp=False)
    m_fit, b_fit = get_m_b(beta_fit)

    if x_predict is None:
        return m_fit, b_fit
    else:
        y_predict = m_fit * x_predict + b_fit
        if return_stats:
            # print(hellods)
            return (m_fit, b_fit), y_predict
        else:
            return y_predict


def OLS_with_CI(xy, n_iterations=1000, ci=95, **kwargs):
    """

    """
    fit_stats, yhat = OLS(xy, **kwargs)
    m_fit, b_fit, r2_fit = fit_stats[0], fit_stats[1], fit_stats[2]

    m_boot_vals, b_boot_vals, yhat_boot_vals = [], [], []
    for iter in range(n_iterations):
        use_idx = np.random.randint(xy.shape[0], size=xy.shape[0])
        xy_boot = xy[use_idx]
        fit_stats_boot, yhat_boot = OLS(xy_boot, **kwargs)
        m_boot, b_boot = fit_stats_boot[0], fit_stats_boot[1]
        m_boot_vals.append(m_boot)
        b_boot_vals.append(b_boot)
        yhat_boot_vals.append(yhat_boot)

    yhat_boot_vals = np.array(yhat_boot_vals)

    m_lo_ci    = np.percentile(m_boot_vals, 0.5*(100-ci))
    m_hi_ci    = np.percentile(m_boot_vals, 0.5*(100+ci))

    b_lo_ci    = np.percentile(b_boot_vals, 0.5*(100-ci))
    b_hi_ci    = np.percentile(b_boot_vals, 0.5*(100+ci))

    yhat_lo_ci = np.percentile(yhat_boot_vals, 0.5*(100-ci), axis=0)
    yhat_hi_ci = np.percentile(yhat_boot_vals, 0.5*(100+ci), axis=0)

    m_more_extreme = np.sum(np.abs(m_boot_vals-m_fit) > np.abs(m_fit))
    p_m_neq_0 = m_more_extreme / n_iterations

    b_more_extreme = np.sum(np.abs(b_boot_vals-b_fit) > np.abs(b_fit))
    p_b_neq_0 = b_more_extreme / n_iterations

    return {
        "r2"         : r2_fit,    "slope"          : m_fit,     "intercept" : b_fit,       "yhat" : yhat,
        "slope_lo_ci": m_lo_ci,   "intercept_lo_ci": b_lo_ci,   "yhat_lo_ci": yhat_lo_ci,
        "slope_hi_ci": m_hi_ci,   "intercept_hi_ci": b_hi_ci,   "yhat_hi_ci": yhat_hi_ci,
        "slope_p"    : p_m_neq_0, "intercept_p"    : p_b_neq_0
    }

def TLS_with_CI(xy_and_err, n_iterations=1000, ci=95, **kwargs):
    """

    """
    (m_fit, b_fit), yhat = TLS(xy_and_err, **kwargs)

    m_boot_vals, b_boot_vals, yhat_boot_vals = [], [], []
    for iter in range(n_iterations):
        use_idx = np.random.randint(xy_and_err.shape[0], size=xy_and_err.shape[0])
        xy_and_err_boot = xy_and_err[use_idx]
        (m_boot, b_boot), yhat_boot = TLS(xy_and_err_boot, **kwargs)
        m_boot_vals.append(m_boot)
        b_boot_vals.append(b_boot)
        yhat_boot_vals.append(yhat_boot)

    yhat_boot_vals = np.array(yhat_boot_vals)

    m_lo_ci    = np.percentile(m_boot_vals, 0.5*(100-ci))
    m_hi_ci    = np.percentile(m_boot_vals, 0.5*(100+ci))

    b_lo_ci    = np.percentile(b_boot_vals, 0.5*(100-ci))
    b_hi_ci    = np.percentile(b_boot_vals, 0.5*(100+ci))

    yhat_lo_ci = np.percentile(yhat_boot_vals, 0.5*(100-ci), axis=0)
    yhat_hi_ci = np.percentile(yhat_boot_vals, 0.5*(100+ci), axis=0)

    m_more_extreme = np.sum(np.abs(m_boot_vals-m_fit) > np.abs(m_fit))
    p_m_neq_0 = m_more_extreme / n_iterations

    b_more_extreme = np.sum(np.abs(b_boot_vals-b_fit) > np.abs(b_fit))
    p_b_neq_0 = b_more_extreme / n_iterations

    return {
        "slope"      : m_fit,     "intercept"      : b_fit,     "yhat" : yhat,
        "slope_lo_ci": m_lo_ci,   "intercept_lo_ci": b_lo_ci,   "yhat_lo_ci": yhat_lo_ci,
        "slope_hi_ci": m_hi_ci,   "intercept_hi_ci": b_hi_ci,   "yhat_hi_ci": yhat_hi_ci,
        "slope_p"    : p_m_neq_0, "intercept_p"    : p_b_neq_0
    }
        
def to_zscore(values):
    """
    Given a list of values, convert them to z-scores by subtracting the mean and dividing by the variance
    """

    zscores = (values - np.mean(values))/np.std(values)

    return zscores