import numpy as np
import pandas as pd
import torch

def observationGymToTorchRL(observation, observationNormalization, device):
    observation = torch.tensor(observation, device=device)
    observationScaled = (
        observation - observationNormalization["loc"].to(device)
    ) / observationNormalization["scale"].to(device)
    return observationScaled


def observationNormalisation(observation, observationNormalization, device):
    observationScaled = (
        observation - observationNormalization["loc"].to(device)
    ) / observationNormalization["scale"].to(device)
    return observationScaled


def actionTorchRLToGym(action, indice):
    return torch.argmax(action[indice]).item()
    
def preprocessData(dataRaw, windowSize):
    nTraining = len(dataRaw[3])
    nState = len(dataRaw[2][0])
    nElement = len(dataRaw[2][0][0])
    episodeLength = dataRaw[3]
    episodeLengthWindowed = [
        torch.tensor(pd.Series.rolling(pd.Series(episodeLengthRun), windowSize, min_periods=1).mean())
        for episodeLengthRun in episodeLength
    ]

    stateProportion = dataRaw[2]
    stateTimeStepWindowed = [
        torch.cat([
            (torch.tensor(
                    pd.Series.rolling(pd.Series(proportion), windowSize, min_periods=1)
                    .mean()
                    .values
            ) * episodeLengthRun) / 1000
            for proportion in stateProportionRun
        ])
        for stateProportionRun, episodeLengthRun in zip(stateProportion, episodeLengthWindowed)
    ]

    return np.array(torch.cat(stateTimeStepWindowed).view(nTraining, nElement, nState))
