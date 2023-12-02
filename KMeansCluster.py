import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 


def main():
    # Replace this with your actual dataset loading code
    df = pd.read_csv('train_normd.csv')
    X = np.array(df.drop('target', axis=1))
    target = np.array(df['target']).reshape(-1,1)
    N, _ = np.shape(X)

    k = 2
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
    KMCl = kmeans.fit(X)
    clCenters = KMCl.cluster_centers_
    clLabels = KMCl.labels_

    distDataCtr = np.zeros(shape=[N, k])
    for i in range(k):
        centerCalc = clCenters[i, :]
        dist = np.linalg.norm((X - centerCalc), axis=1)
        distDataCtr[:, i] = dist

    if k == 2:
        max_dist = np.max([1.05*np.max(distDataCtr[:, 0]), 1.1*np.max(distDataCtr[:, 1])])
        plt.scatter(distDataCtr[:, 0], distDataCtr[:, 1], c=target, cmap='plasma', s=10)
        plt.colorbar()
        plt.plot([0, max_dist], [0, max_dist])
        plt.xlabel('Distance from Cluster Center 1')
        plt.ylabel('Distance from Cluster Center 2')
        plt.title('K-Means Clustering')
        plt.xlim((0, max_dist))
        plt.ylim((0, max_dist))
        plt.show()

    XTr, XTe, yTr, yTe = train_test_split(distDataCtr, target, test_size=0.2)
    
    lrModel = LinearRegression()
    lrModel.fit(XTr, yTr)

    yPredTr = lrModel.predict(XTr)
    yPred = lrModel.predict(XTe)

    rmseTr = mean_squared_error(yTr, yPredTr, squared=False)
    rmseTe = mean_squared_error(yTe, yPred, squared=False)
    print(rmseTr, "\n")
    print(rmseTe, "\n")

    

if __name__ == "__main__":
    main()