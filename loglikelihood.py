import numpy as np
from scipy.optimize import minimize

def y_np(assortments, probs, n_prods=31):
    y = np.zeros((len(assortments), n_prods))
    for i in range(len(assortments)):
        prod_num = 0
        for prod in assortments[i]:
            y[i, prod] = probs[i][prod_num]
            prod_num += 1
    return y

def softmax1d(v):
    """
    Args:
        v: (M,) array, utility
    Return:
        q: (M,) array, choice probability
    """
    return np.exp(v)/np.sum(np.exp(v))

def softmax2d(v):
    """
    Args:
        v: (N,M) array where v[i,j] is the utility of alternative j in observation i
    Return:
        q: (N,M) array where q[i,j] is the predicted choice probability of alternative j in observation i
    """
    return np.exp(v)/np.sum(np.exp(v), axis=1, keepdims=True)

def predict(beta, x, y):
    """
    Args:
        beta: (D,) array, parameter for each feature
        x: (M,D) array where x[j,k] is the value of feature k for alternative j
        y: (N,M) array where y[i,j] == 0 iff alternative j is not availible in assortment i
    Return:
        q: (N,M) array where q[i,j] is the predicted choice probability of alternative j in observation i
    """
    v = np.where(y != 0, np.sum(beta*x, axis=1), -np.inf) # utility
    return softmax2d(v)

def ll(beta, x, y):
    """
    Args:
        beta: (D,) array, parameter for each feature
        x: (M,D) array where x[j,k] is the value of feature k for alternative j
        y: (N,M) array where y[i,j] is the true choice probability of alternative j in observation i
    Return:
        ll: float, loglikelihood of beta given x and y
    """
    q = predict(beta, x, y) # choice probability
    q1 = np.where(q != 0, q, 1) # when q is zero, q1 is one so that log(q1) is zero
    return np.sum(y*np.log(q1))

def neg_ll(beta, x, y):
    return -ll(beta, x, y)

def calc_beta(x, y, beta0=None, method='Nelder-Mead', full_ouput=False):
    """
    Args:
        x: (M,D) array where x[j,k] is the value of feature k for alternative j
        y: (N,M) array where y[i,j] is the true choice probability of alternative j in observation i
        beta0: (D,) array, initial parameters
        method: passed to scipy.optimize.minimize
            see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    Return:
        min: output of scipy.optimize.minimize
            min["x"] contains the optimal beta values
    """
    if beta0 is None:
        beta0 = np.zeros(x.shape[1])
    if full_ouput:
        return minimize(neg_ll, beta0, args=(x,y), method=method)
    return minimize(neg_ll, beta0, args=(x,y), method=method)["x"]