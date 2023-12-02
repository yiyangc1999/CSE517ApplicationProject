import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, KFold
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.neural_network import MLPRegressor

def RFRegressorErrorCalc(X, Y, splitNum):
    inSampleError = []
    outSampleError = []
    kf = KFold(n_splits = splitNum)
    for train, test in kf.split(X, Y):
        RFRegr = RandomForestRegressor() # Used squared loss, which is default
        RFRegr.fit(X.iloc[train,:], Y.iloc[train])
        inSampleError = np.append(inSampleError,(RFRegr.predict(X.iloc[train,:])-Y.iloc[train])**2)
        outSampleError = np.append(outSampleError,(RFRegr.predict(X.iloc[test,:])-Y.iloc[test])**2)

    return inSampleError, outSampleError


def main():
    dfRaw = pd.read_csv('train.csv')
    dfNormd = pd.read_csv('train_Normd.csv')


    # random forest regression, raw data & normalized data
    XRaw = dfRaw.drop(columns='target')
    YRaw = dfRaw['target']
    XNormd = dfNormd.drop(columns='target')
    YNormd = dfNormd['target']
    cvNum = 5
    
    # errors
    rawDataRFInSampleError, rawDataRFOutSampleError = RFRegressorErrorCalc(XRaw, YRaw, cvNum)
    normdDataRFInSampleError, normdDataRFOutSampleError = RFRegressorErrorCalc(XNormd, YNormd, cvNum)

    # paired t-test
    errorCmpRes = [
    stats.ttest_rel(rawDataRFInSampleError, normdDataRFInSampleError),
    stats.ttest_rel(rawDataRFOutSampleError, normdDataRFOutSampleError),
    ]
    print("Raw Data & Normalized Data In-sample Error T-Test: \n", "T-statistic =", round(tuple(errorCmpRes[0])[0], 4), ", pvalue =", round(tuple(errorCmpRes[0])[1], 4), "\n")
    print("Raw Data & Normalized Data Out-of-sample Error T-Test: \n", "T-statistic =", round(tuple(errorCmpRes[1])[0], 4), ", pvalue =", round(tuple(errorCmpRes[1])[1], 4), "\n")


if __name__ == "__main__":
    main()