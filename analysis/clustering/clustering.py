import datetime
import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import torch
from tslearn.clustering import TimeSeriesKMeans, silhouette_score

from utils.tools.func import preprocessData

def plotHistogramme(stateProportionWindowed, save = False, path = ""):
    fig, ax = plt.subplots()
    
    stateProportionWindowed = torch.tensor(stateProportionWindowed).view(4, 3906)
    nbState = stateProportionWindowed.size()[0]

    rolloutBatchTimeStep = 32
    totalTimeStep = 125_000

    X = [i*rolloutBatchTimeStep for i in range(totalTimeStep//rolloutBatchTimeStep)]

    lastProportion = torch.zeros(len(stateProportionWindowed[0]))
    for i in range(nbState):
        ax.fill_between(X, 1000 * lastProportion, 1000 * (lastProportion + stateProportionWindowed[i]), label="State " + str(i))
        lastProportion +=  stateProportionWindowed[i]
    plt.title("State distribution over training")
    plt.xlabel("Environment Steps")
    plt.ylabel("Average Timesteps spent per state per episode")
    plt.ylim(0,1000)
    plt.legend()
    if save:
        plt.savefig(path)
    else:
        plt.show()

    plt.clf()

def main():
    pathTrain = "articleData/TrainingsD3QNTrainClustering/resultData.pkl"
    k = 2
    pathInferenceList = [
        "articleData/TrainingsD3QNTestClustering/resultData.pkl",
        "articleData/TrainingsD3QNHeuristic/resultData.pkl",
    ]
    pathInferenceName = [
        "D3QNTest",
        "D3QNHeuristic"
    ]

    currentTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create the saving path
    try:
        os.mkdir("analysis/clustering/Data")
    except:
        pass
    path = "analysis/clustering/Data/" + str(currentTime)
    os.mkdir(path)
    path += "/"

    with open(pathTrain ,"rb") as f:
        dataRawTrain = pkl.load(f)

    X = preprocessData(dataRawTrain, 250)

    resDict = {}

    km = TimeSeriesKMeans(n_clusters = k, metric="euclidean", n_jobs=20).fit(X)

    for i, center in enumerate(km.cluster_centers_):
        clusterName = "Cluster" + str(i)
        resDict[clusterName] = {}
        indice = np.nonzero(km.labels_ == i)[0].tolist()
        resDict[clusterName]["indice"] = indice
        nbElement = len(indice)

        resDict[clusterName]["nbElement"] = nbElement
        plotHistogramme(center.squeeze(), True, path + "clusterCentroid" + str(i) + ".png")

    silhouette = silhouette_score(X, km.predict(X), metric="euclidean", n_jobs=-1)
    inertie = np.mean(km.inertia_)
    print("Silhouette Score : " + str(silhouette))
    print("Inertia Score : " + str(inertie))

    resDict["silhouette"] = float(silhouette)
    resDict["inertia"] = float(inertie)

    with open(path + "ClusteringLabelListTrain.pkl", "wb") as f:
        pkl.dump(km.labels_, f)


    for pathData, pathName in zip(pathInferenceList, pathInferenceName):

        with open(pathData ,"rb") as f:
            dataRawTestBase = pkl.load(f)

        XTestBase = preprocessData(dataRawTestBase, 250)

        labelTestBase = torch.tensor(km.predict(XTestBase))

        print("Clustering " + pathName)
        print("Cluster 0 : " + str(200 - labelTestBase.nonzero().size()[0]))
        print("Cluster 1 : " + str(labelTestBase.nonzero().size()[0]))

        with open(path + "ClusteringLabelList" + pathName + ".pkl", "wb") as f:
            pkl.dump(labelTestBase, f)



if __name__ == "__main__":
    main()