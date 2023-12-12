import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

def normalizeDataTrTe(dataTr, dataTe):
    scaler = MinMaxScaler()
    scaler.fit(dataTr)
    XTrNormd = scaler.transform(dataTr)
    XTeNormd = scaler.transform(dataTe)

    return XTrNormd, XTeNormd

def main():
    dfTr = pd.read_csv('train_CoE.csv')
    # XTr = dfTr.drop(columns='target')
    # yTr = dfTr['target']

    XTr = dfTr[['absoluate_roll','current_roll','current_pitch','m','n',
                'omega','set','time14','time13','time12','time11','time10','time9',
                'time8','time7','time6','time5','time4','time3','time2','time1']]
    yTr = dfTr['target']

    dfTe = pd.read_csv('test.csv')
    XTe = dfTe[['absoluate_roll','current_roll','current_pitch','m','n',
                'omega','set','time14','time13','time12','time11','time10','time9',
                'time8','time7','time6','time5','time4','time3','time2','time1']]

    train_self_test_for_model_selection = True
    normalizeData = False
    
    if train_self_test_for_model_selection == True:
        X_train, X_test, y_train, y_test = train_test_split(XTr, yTr, train_size=0.8, shuffle=True)
        
        if normalizeData == True:
            X_train, X_test = normalizeDataTrTe(X_train, X_test)

        model = RandomForestRegressor( n_estimators=100 )
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

        model = RandomForestRegressor() # Used squared loss, which is default
        RFRegr = model.fit(XTr, yTr)

        yTrPred = RFRegr.predict(XTr)
        yTePred = RFRegr.predict(XTe)

        rmseTr = mean_squared_error(yTr, yTrPred, squared=False)

        print('Training RMSE = ', rmseTr)
        
        dfyTePred = pd.DataFrame({'Id': idTe, 'target': yTePred})
        dfyTePred.to_csv('RFRegrPrediction.csv', index=False, header=True)



if __name__ == "__main__":
    main()