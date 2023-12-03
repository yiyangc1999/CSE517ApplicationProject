import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, silhouette_score


def main():
    df = pd.read_csv('train_normd.csv')
    X = np.array(df.drop('target', axis=1))
    target = np.array(df['target']).reshape(-1,1)
    N, dim = np.shape(X)

    kList = np.append(np.array([2]), np.arange(5,501,5))
    silhouetteAvgScoreList = np.array([])
    inertiaList = np.array([])
    rmseTrList = np.array([])
    rmseTeList = np.array([])
    for indk in range(len(kList)):
        k = kList[indk]
        kmeansModel = KMeans(n_clusters=k, random_state=0, n_init="auto")
        KMCl = kmeansModel.fit(X)
        clCenters = KMCl.cluster_centers_
        clLabels = KMCl.labels_

        silhouetteAvgScore = silhouette_score(X, clLabels)
        silhouetteAvgScoreList = np.append(silhouetteAvgScoreList,silhouetteAvgScore)
        inertiaList = np.append(inertiaList, kmeansModel.inertia_)

        distDataCtr = np.zeros(shape=[N, k])
        for i in range(k):
            centerCalc = clCenters[i, :]
            dist = np.linalg.norm((X - centerCalc), axis=1)
            distDataCtr[:, i] = dist

        if k == 2:
            max_dist = np.max([1.05*np.max(distDataCtr[:, 0]), 1.1*np.max(distDataCtr[:, 1])])
            fig1 = plt.figure()
            plt.scatter(distDataCtr[:, 0], distDataCtr[:, 1], c=target, cmap='plasma', s=10)
            plt.colorbar()
            plt.plot([0, max_dist], [0, max_dist])
            plt.xlabel('Distance from Cluster Center 1')
            plt.ylabel('Distance from Cluster Center 2')
            plt.title('K-Means Clustering')
            plt.xlim((0, max_dist))
            plt.ylim((0, max_dist))
            fig1.show()
            fig1.savefig("2 cluster scatter plot.png")
            plt.close(fig1)

        XTr, XTe, yTr, yTe = train_test_split(distDataCtr, target, test_size=0.2)
        
        lrModel = LinearRegression()
        lrModel.fit(XTr, yTr)

        yPredTr = lrModel.predict(XTr)
        yPred = lrModel.predict(XTe)

        rmseTr = mean_squared_error(yTr, yPredTr, squared=False)
        rmseTe = mean_squared_error(yTe, yPred, squared=False)
        if k == 2:
            print("Training RMSE = ", rmseTr)
            print("Testing RMSE = ", rmseTe)

        rmseTrList = np.append(rmseTrList, rmseTr)
        rmseTeList = np.append(rmseTeList, rmseTe)

    figRMSE = plt.figure()
    plt.plot(kList, rmseTrList, label="Training RMSE")
    plt.plot(kList, rmseTeList, label="Testing RMSE")
    figRMSE.legend()
    plt.xlabel('number of clusters')
    plt.ylabel('RMSE')
    figRMSE.show()
    figRMSE.savefig("RMSE_wrt_k.png")
    plt.close(figRMSE)

    figScore = plt.figure()
    plt.plot(kList, silhouetteAvgScoreList)
    plt.xlabel('number of clusters')
    plt.ylabel('silhouette score')
    figScore.show()
    figScore.savefig("score_wrt_k.png")
    plt.close(figScore)

    figInertia = plt.figure()
    plt.plot(kList, inertiaList)
    plt.xlabel('number of clusters')
    plt.ylabel('inertia')
    figInertia.show()
    figInertia.savefig("Inertia_wrt_k.png")
    plt.close(figInertia)

if __name__ == "__main__":
    main()