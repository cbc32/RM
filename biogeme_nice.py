import numpy as np
import pandas as pd

import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models ###Use my modified source code for biogeme.models
from biogeme.expressions import Beta

def choice_df(df: pd.DataFrame, n_dup: int=1) -> pd.DataFrame:
    """
    Function:
        Convert choice probabilities into n_dups choice class observations
            using the given choice probabilities as a CDF
    """
    df1 = pd.DataFrame(columns=["Assortment", "Choice"])
    if n_dup == 1:
        df1["Assortment"] = df["Assortment"]
        df1["Choice"] = [df["Assortment"][i][pd.Series(df["Probability"][i]).idxmax()] for i in range(len(df))]
        return df1
    
    p_step = 1/(n_dup-1)

    df.reset_index(drop=True, inplace=True)
    for row in df.itertuples(index=False):
        dups = pd.DataFrame(columns=["Assortment", "Choice"])
        probs = row[1]
        assort_len = len(probs)
        cum_prob = np.cumsum(probs)
        p = 0
        for i in range(assort_len):
             while p <= cum_prob[i] + 0.0001:
                p += p_step
                dups.loc[len(dups)] = [row[0], row[0][i]]
        df1 = pd.concat([df1, dups])
    return df1.reset_index(drop=True).astype({"Choice": 'int32'})

def av_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function:
        Convert column "Assortment" to columns ["Av0",...,"Av31"]
            representing the availibility of each product under the assortment
    """
    df1 = df.copy()
    for prod in range(31):
        df1["Av"+str(prod)] = [int(prod in list(df1["Assortment"][i])) for i in range(len(df1))]
    return df1.drop(columns=["Assortment"]).astype("int32")

def calc_beta(df: pd.DataFrame, x: np.array, intercept=False):
    """
    Params:
        df: columns include ["Choice", "Av0", ..., "Av31"]
        x: 31 x 4 feature array
    Return:
        biogeme predicted biogeme models for MNL model Strained on choice class
    """
    database = db.Database("calc_betas",df)
    globals().update(database.variables)

    beta = [Beta("B"+str(i), 0, None, None, 0) for i in range(1,5)]
    if intercept:
        a_param = [Beta("a"+str(i), 0, None, None, 0) for i in range(31)]
        V = {i: a_param[i]+x[i,0]*beta[0]+x[i,1]*beta[1]+x[i,2]*beta[2]+x[i,3]*beta[3] for i in range(1,31)} 
    else:
        V = {i: x[i,0]*beta[0]+x[i,1]*beta[1]+x[i,2]*beta[2]+x[i,3]*beta[3] for i in range(1,31)} 
    V[0] = 0

    av = {i: eval("Av"+str(i)) for i in range(31)}

    logprob = models.loglogit(V, av, Choice)
    biogeme = bio.BIOGEME(database, logprob)
    biogeme.modelName = 'logit'
    results = biogeme.estimate()
    return results.getBetaValues()

def pred(beta_dict: dict, x: np.array, i: int, assortment) -> float:
    """
    Params:
        beta_dict: {"B1": float, ..., "B4": float}
        x: 31 x 4 feature array
        i: product in [0..31]
        assortment: iterable
    Return:
        predicted choice probability for product i in the assortment
    """
    if i not in assortment:
        return 0
    assortment = [int(prod) for prod in assortment]
    if 0 in assortment:
        assortment.remove(0)
    beta = [beta_dict["B"+str(j)] for j in range(1,5)]
    if len(beta_dict) > x.shape[1]: # if intercept
        if i == 0:
            return 1/(np.sum([np.exp(beta_dict["a"+str(j)] + np.dot(beta,x[j])) for j in assortment]) + 1)
        return np.exp(beta_dict["a"+str(i)] + np.dot(beta,x[i]))/(np.sum([np.exp(beta_dict["a"+str(j)] + np.dot(beta,x[j])) for j in assortment]) + 1)
    if i == 0:
        return 1/(np.sum([np.exp(np.dot(beta,x[j])) for j in assortment]) + 1)
    return np.exp(np.dot(beta,x[i]))/(np.sum([np.exp(np.dot(beta,x[j])) for j in assortment]) + 1)

def pred_df(beta_dict: dict, x: np.array, assortments) -> pd.DataFrame:  
    """
    Params:
        beta_dict: {"B1": float, ..., "B4": float}
        x: 31 x 4 feature array
        assortments: iterable
    Return:
        DataFrame with columns for the assortments and predicted choice
            probability for each product in that row's assortment
    """ 
    df1 = pd.DataFrame(columns=["Assortment","Pred"])
    for assort in assortments:
        preds = [pred(beta_dict, x, i, assort) for i in assort]
        df1.loc[len(df1)] = [assort, preds]
    return df1

def mae(y,q):
    """
    Function:
        mean absolute value of error
    """
    return np.mean(np.abs(np.subtract(y,q)))

def sparse_mae(y,q):
    """
    Function:
        average MAE for each row for rows containing lists of different lengths
    """
    return np.mean([mae(y[i], q[i]) for i in range(len(y))])

def get_beta_dict(beta) -> dict:
    """
    Function:
        convert iterable beta to {"B1": float, ..., "B4": float}
    """
    return {"B"+str(i): beta[i-1] for i in range(1,5)}