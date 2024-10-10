import pickle as pkl

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

from utils.tools.func import preprocessData

def addLabelViolin(violinList, labelList):
    labels = []
    for violin, label in zip(violinList, labelList):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))
    plt.legend(*zip(*labels), loc=2)

def main():
    # Arguments
    pathTraining1 = "articleData/TrainingsD3QNTestClustering/resultData.pkl"
    pathTraining2 = "articleData/TrainingsD3QNHeuristic/resultData.pkl" 
    labelTrainings = ("D3QN", "D3QNModified")

    with open(pathTraining1 ,"rb") as f:
        dataRawTraining1 = pkl.load(f)

    XTraining1 = preprocessData(dataRawTraining1, 250)

    with open(pathTraining2 ,"rb") as f:
        dataRawTraining2 = pkl.load(f)

    XTraining2 = preprocessData(dataRawTraining2, 250)
    
    dataPreprocessedTest = torch.mean(torch.tensor(XTraining1).view(200, 4, 3906) * 1000, dim=2).T
    violinPartsTest = plt.violinplot(dataPreprocessedTest, [0.2, 2.2, 4.2, 6.2])

    dataPreprocessedExplo = torch.mean(torch.tensor(XTraining2).view(200, 4, 3906) * 1000, dim=2).T
    violinPartsExplo = plt.violinplot(dataPreprocessedExplo, [0.8, 2.8, 4.8, 6.8])

    xTicks = [0.5, 2.5, 4.5, 6.5]
    labelTicks = ["State " + str(i) for i in range(4)]

    plt.xticks(xTicks, labelTicks)
    plt.ylabel("Average Timesteps per episode")
    addLabelViolin((violinPartsTest, violinPartsExplo), labelTrainings)
    plt.show()

if __name__ == "__main__":
    main()