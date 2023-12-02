import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy import matlib

def deleteMissingValues(dataFrame):
    dataTr = np.array(dataFrame)
    
    # characterize missing values
    ifDataMiss = np.sum(np.isnan(dataTr))
    # delete missing values
    if ifDataMiss != 0:
        indNaN = np.argwhere(dataTr == np.NaN)
        rowDelete = indNaN[:,0]
        data = np.delete(dataTr, rowDelete, axis=0)
        dataTr = data
    
    return dataTr

def errorCorrection(data):
    N,dim = np.shape(data)

    dataFeatures = data[:,0:-1]
    dataFtMean = np.mean(dataFeatures, axis=0)
    dataFtStd  = np.std(dataFeatures, axis=0)

    dataFtCmp = matlib.repmat(4 * dataFtStd, N, 1)

    indError = np.argwhere(np.abs(dataFeatures - dataFtMean) > dataFtCmp)
    rowError = indError[:,0]
    rowDelete = np.unique(rowError)

    dataCoE = np.delete(data, rowDelete, axis=0)

    return dataCoE

def rescaleData(data):
    N,dim = np.shape(data)

    dataTarget = np.reshape(data[:,-1],[N,1])
    dataFeatures = data[:,0:-1]

    scaler = MinMaxScaler()
    scaler.fit(dataFeatures)
    dataNormdFeatures = scaler.transform(dataFeatures)

    dataNormd = np.append(dataNormdFeatures,dataTarget,axis=1)

    return dataNormd

def main():
    dfTr = pd.read_csv('train.csv')
    # get the names of the column header
    featureNames = list(dfTr.columns)
    
    dataTr = deleteMissingValues(dfTr)

    dataTrCoE = errorCorrection(dataTr)
    dfCoE= pd.DataFrame(dataTrCoE, columns=featureNames)
    dfCoE.to_csv('train_CoE.csv', index=False, header=True) 

    dataTrNormd = rescaleData(dataTr)
    dfNormd= pd.DataFrame(dataTrNormd, columns=featureNames)
    dfNormd.to_csv('train_Normd.csv', index=False, header=True) 

    dataTrCoENormd = rescaleData(dataTrCoE)
    dfCoENormd = pd.DataFrame(dataTrCoENormd, columns=featureNames)
    dfCoENormd.to_csv('train_CoE_Normd.csv', index=False, header=True)  


if __name__ == "__main__":
    main()