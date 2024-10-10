import pickle as pkl

import matplotlib.pyplot as plt
import torch

def main():
    # Arguments
    pathTraining = "articleData/TrainingsD3QNHeuristic/resultData.pkl" 

    with open(pathTraining ,"rb") as f:
        dataRawTraining = pkl.load(f)

    zetaValues = torch.tensor(dataRawTraining[-1])

    # Zeta min-max profile
    minTensor = torch.min(zetaValues, dim=0).values
    maxTensor = torch.max(zetaValues, dim=0).values
    meanTensor = torch.mean(zetaValues, dim=0)
    x = range(len(minTensor))

    fig, ax = plt.subplots()
    ax.fill_between(x, minTensor, maxTensor, alpha=0.2, color="b")
    ax.plot(meanTensor, color="b")
    plt.show()

    # Zeta 95%CI profile
    n = len(zetaValues)
    stdTensor = torch.std(zetaValues, dim=0)
    meanTensor = torch.mean(zetaValues, dim=0)
    CIshapeTensor = (1.96/n ** (1/2)) * stdTensor
    x = range(len(minTensor))

    fig, ax = plt.subplots()
    ax.fill_between(x, meanTensor - CIshapeTensor, meanTensor + CIshapeTensor, alpha=0.2, color="b")
    ax.plot(meanTensor, color="b")
    plt.show()

if __name__ == "__main__":
    main()