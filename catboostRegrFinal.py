import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostRegressor

def normalizeDataTrTe(dataTr, dataTe):
    scaler = MinMaxScaler()
    scaler.fit(dataTr)
    XTrNormd = scaler.transform(dataTr)
    XTeNormd = scaler.transform(dataTe)

    return XTrNormd, XTeNormd

def main():
    dfTr = pd.read_csv('train.csv')
    dfTe = pd.read_csv('test.csv')

    train_self_test_for_model_selection = False
    normalizeData = True
    drop_low_corr_feature = True
    model = CatBoostRegressor(loss_function='RMSE', n_estimators = 10000 ,depth = 6, learning_rate = 0.01, l2_leaf_reg = 3, bagging_temperature = 3)

    if drop_low_corr_feature:
        # Get the whole correlation matrix
        corrMat = dfTr.corr()
        # Get the correlation of features to target
        corr2target = corrMat.target

        corrThr = 0.1
        featureIdx = np.argwhere(np.abs(corr2target) > corrThr)
        featureIdx = featureIdx[1:-1]
        featureList = dfTr.columns[featureIdx.ravel()]

        XTr = dfTr[featureList]
        yTr = dfTr['target']
        XTe = dfTe[featureList]
    else:
        XTr = dfTr.drop(columns='target')
        yTr = dfTr['target']
        XTe = dfTe.drop(columns='Id')
    
    if train_self_test_for_model_selection:
        X_train, X_test, y_train, y_test = train_test_split(XTr, yTr, train_size=0.8, shuffle=False)
        
        if normalizeData:
            X_train, X_test = normalizeDataTrTe(X_train, X_test)

        RFRegr = model.fit(X_train, y_train)

        yTrPred = RFRegr.predict(X_train)
        yTePred = RFRegr.predict(X_test)

        rmseTr = mean_squared_error(y_train, yTrPred, squared=False)
        rmseTe = mean_squared_error(y_test, yTePred, squared=False)

        print('Training RMSE = ', rmseTr)
        print('Testing RMSE = ', rmseTe)
    else:
        XTr = np.array(XTr)
        yTr = np.array(yTr)
        XTe = np.array(XTe)
        idTe = dfTe['Id'].values

        if normalizeData == True:
            XTr, XTe= normalizeDataTrTe(XTr, XTe)

        RFRegr = model.fit(XTr, yTr)

        yTrPred = RFRegr.predict(XTr)
        yTePred = RFRegr.predict(XTe)

        rmseTr = mean_squared_error(yTr, yTrPred, squared=False)

        print('Training RMSE = ', rmseTr)
        
        dfyTePred = pd.DataFrame({'Id': idTe, 'target': yTePred})
        dfyTePred.to_csv('CBRegrPrediction.csv', index=False, header=True)



if __name__ == "__main__":
    main()