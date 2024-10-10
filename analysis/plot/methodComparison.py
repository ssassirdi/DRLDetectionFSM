import os
os.environ["PYTHONWARNINGS"] = "ignore"

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


def main() -> None:
    # Arguments
    pathList = [
        "articleData/InferenceD3QNTestClustering/resList.pkl",
        "articleData/InferenceD3QNHeuristic/resList.pkl",
    ]

    pathNameList = [
        "D3QN",
        "D3QN with Heuristic"
    ]

    policyIndiceList = [1,2,3,4,5,6,7,8,9,10]

    absciss = [12_500 * modelId for modelId in policyIndiceList]

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
        ] for _ in range(len(pathList))]

    modelNumberToIndice = {
        str(modelId) : i
        for i, modelId in enumerate(policyIndiceList)
    }

    for i, path in enumerate(pathList):
        with open(path, "rb") as f:
            resList = pkl.load(f)

        for res in resList:
            modelIndice = modelNumberToIndice[str(res["modelNumber"])]
            resDictList[i][modelIndice]["reward"]["mean"].append(res["reward"]["mean"])
            resDictList[i][modelIndice]["reward"]["median"].append(res["reward"]["median"])
            resDictList[i][modelIndice]["reward"]["std"].append(res["reward"]["std"])
            resDictList[i][modelIndice]["length"]["mean"].append(res["length"]["mean"])
            resDictList[i][modelIndice]["length"]["median"].append(res["length"]["median"])
            resDictList[i][modelIndice]["length"]["std"].append(res["length"]["std"])

    nIndividuListe = []
    for j, pathName in enumerate(pathNameList):
        print("Method : " + pathName)
        nIndividuListe.append(len(resDictList[j][0]["reward"]["mean"]))
        print("Nombre individu : " + str(nIndividuListe[j]))
        for i, modelId in enumerate(policyIndiceList):
            print("Model id : " + str(modelId))
            print("Average of mean reward per episode : " + str(np.mean(resDictList[j][i]["reward"]["mean"])))
            print("Std of mean reward per episode : " + str(np.std(resDictList[j][i]["reward"]["mean"])))
            print("Confidence Interval  (95%) : " + str((1.96 * np.std(resDictList[j][i]["reward"]["mean"])) / np.sqrt(nIndividuListe[j])))
            print("Average of mean length per episode : " + str(np.mean(resDictList[j][i]["length"]["mean"])))
            print("Std of mean length per episode  : " + str(np.std(resDictList[j][i]["length"]["mean"])))



    plt.subplot(211)
    for j, pathName in enumerate(pathNameList):
        rewardValue = np.array([np.mean(resDictList[j][i]["reward"]["mean"]) for i in range(len(policyIndiceList))])
        stdRewardValue = np.array([(1.96 * np.std(resDictList[j][i]["reward"]["mean"])) / np.sqrt(nIndividuListe[j]) for i in range(len(policyIndiceList))])
        plt.plot(absciss, rewardValue, label = pathName)
        plt.fill_between(absciss, rewardValue + stdRewardValue, rewardValue - stdRewardValue, alpha=0.3)
    plt.title("Total Reward per episode over training")
    plt.ylabel("Total Reward per episode")
    plt.xlabel("Environment TimeSteps")
    plt.legend()

    plt.subplot(212)
    for j, pathName in enumerate(pathNameList):
        lengthValue = np.array([np.mean(resDictList[j][i]["length"]["mean"]) for i in range(len(policyIndiceList))])
        stdLengthValue = np.array([(1.96 * np.std(resDictList[j][i]["length"]["mean"])) / np.sqrt(nIndividuListe[j]) for i in range(len(policyIndiceList))])
        plt.plot(absciss, lengthValue, label = pathName)
        plt.fill_between(absciss, lengthValue + stdLengthValue, lengthValue - stdLengthValue, alpha=0.3)
    plt.title("Timesteps per episode over training")
    plt.ylabel("Timesteps per episode")
    plt.xlabel("Environment TimeSteps")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    main()
