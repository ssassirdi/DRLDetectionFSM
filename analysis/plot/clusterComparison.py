import os
os.environ["PYTHONWARNINGS"] = "ignore"

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


def main() -> None:
    # Arguments
    pathInferenceData = "articleData/InferenceD3QNTestClustering/resList.pkl"
    labelListPath = "articleData/Clustering/ClusteringLabelListD3QNTest.pkl"
    policyIndiceList = [1,2,3,4,5,6,7,8,9,10]
    clusterList = [0, 1]
    clusterNumber = len(clusterList)

    absciss = [12_500 * modelId for modelId in policyIndiceList]

    with open(pathInferenceData, "rb") as f:
        resList = pkl.load(f)

    with open(labelListPath, "rb") as f:
        labelList = pkl.load(f)

    resDictList = [
        [
            
                {
                    "reward" : {
                        "mean" : [],
                        "median" : [],
                        "std" : [],
                    },
                    "length" : {
                        "mean" : [],
                        "median" : [],
                        "std" : [],
                    }

                }
            for _ in policyIndiceList
        ] for _ in range(clusterNumber + 1)]

    modelNumberToIndice = {
        str(modelNumber) : i
        for i, modelNumber in enumerate(policyIndiceList)
    }

    for res in resList:
        labelIndice = labelList[res["trainingNumber"]]
        modelIndice = modelNumberToIndice[str(res["modelNumber"])]
        resDictList[labelIndice][modelIndice]["reward"]["mean"].append(res["reward"]["mean"])
        resDictList[labelIndice][modelIndice]["reward"]["median"].append(res["reward"]["median"])
        resDictList[labelIndice][modelIndice]["reward"]["std"].append(res["reward"]["std"])
        resDictList[labelIndice][modelIndice]["length"]["mean"].append(res["length"]["mean"])
        resDictList[labelIndice][modelIndice]["length"]["median"].append(res["length"]["median"])
        resDictList[labelIndice][modelIndice]["length"]["std"].append(res["length"]["std"])

    nTrainingsList = []
    for j, clusterId in enumerate(clusterList):
        print("Cluster " + str(clusterId))
        nTrainingsList.append(len(resDictList[clusterId][0]["reward"]["mean"]))
        print("Number of trainings : " + str(nTrainingsList[j]))
        for i, modelId in enumerate(policyIndiceList):
            print("Model id : " + str(modelId))
            print("Average of mean reward per episode : " + str(np.mean(resDictList[clusterId][i]["reward"]["mean"])))
            print("Std of mean reward per episode : " + str(np.std(resDictList[clusterId][i]["reward"]["mean"])))
            print("Confidence Interval  (95%) : " + str((1.96 * np.std(resDictList[clusterId][i]["reward"]["mean"])) / np.sqrt(nTrainingsList[j])))
            print("Average of mean length per episode : " + str(np.mean(resDictList[clusterId][i]["length"]["mean"])))
            print("Std of mean length per episode  : " + str(np.std(resDictList[clusterId][i]["length"]["mean"])))


    plt.subplot(211)
    for j, clusterId in enumerate(clusterList):
        rewardValue = np.array([np.mean(resDictList[clusterId][i]["reward"]["mean"]) for i in range(len(policyIndiceList))])
        stdRewardValue = np.array([(1.96 * np.std(resDictList[clusterId][i]["reward"]["mean"])) / np.sqrt(nTrainingsList[j]) for i in range(len(policyIndiceList))])
        plt.plot(absciss, rewardValue, label="Cluster " + str(clusterId) + " (" + str(nTrainingsList[j]) + "  trainings )")
        plt.fill_between(absciss, rewardValue + stdRewardValue, rewardValue - stdRewardValue, alpha=0.3)
    plt.title("Average Total Reward per episode over training per cluster")
    plt.ylim(-200,300)
    plt.ylabel("Total Reward per episode")
    plt.xlabel("Environment Timesteps")
    plt.legend()

    plt.subplot(212)
    for j, clusterId in enumerate(clusterList):
        lengthValue = np.array([np.mean(resDictList[clusterId][i]["length"]["mean"]) for i in range(len(policyIndiceList))])
        stdLengthValue = np.array([(1.96 * np.std(resDictList[clusterId][i]["length"]["mean"])) / np.sqrt(nTrainingsList[j]) for i in range(len(policyIndiceList))])
        plt.plot(absciss, lengthValue, label="Cluster " + str(clusterId) + " (" + str(nTrainingsList[j]) + "  trainings )")
        plt.fill_between(absciss, lengthValue + stdLengthValue, lengthValue - stdLengthValue, alpha=0.3)
    plt.title("Average Timesteps per episode over training per cluster")
    plt.ylim(0,1000)
    plt.ylabel("Timesteps per episode")
    plt.xlabel("Environment Timesteps")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
