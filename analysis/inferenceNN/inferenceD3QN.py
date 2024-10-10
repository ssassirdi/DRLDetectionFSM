import os
os.environ["PYTHONWARNINGS"] = "ignore"
import argparse
import yaml
import tqdm
import datetime
import copy

import numpy as np
import gymnasium as gym
import pickle as pkl
import torch
import multiprocessing

from torch import nn

from torchrl.envs.utils import ExplorationType, set_exploration_type

from torchrl.modules import DuelingMlpDQNet, QValueActor

from utils.tools.func import observationNormalisation, actionTorchRLToGym
from utils.tools.Parser import inferenceDataArgGenerator

torch.set_num_threads(1)

def actor(args):
    dictGlob = args[0]
    specificData = copy.deepcopy(args[1])

    # Assigning variables
    # Policy Parameters
    commonLayersCellNumber = dictGlob["Policy Parameters"]["commonLayersCellNumber"]
    outputLayerCellNumber = dictGlob["Policy Parameters"]["outputLayerCellNumber"]
    policyPath = specificData["policyPath"]
    dataRecordedPath = specificData["dataRecordedPath"]
    # Run Parameters
    rolloutNumber = dictGlob["Run Parameters"]["rolloutNumber"]
    maxTimeStep = dictGlob["Run Parameters"]["maxTimeStep"]
    device = dictGlob["Run Parameters"]["device"]
    

    # Load Data Recorded
    with open(dataRecordedPath, "rb") as f:
        dataRecorded = pkl.load(f)


    environment = gym.make("LunarLander-v2")

    # Network Instantiation
    actionSpec = dataRecorded["actionSpec"][0]

    # Loading for a fake tensordict to initialise the neural network
    fakeTensordict = pkl.load(open("analysis/inferenceNN/fakeTensor.pkl", "rb"))

    # Networks Instantiation
    # Model Instantiation
    mlp_kwargs_features = {
    "depth" : len(commonLayersCellNumber),
    "num_cells" : commonLayersCellNumber,
    "activation_class" : nn.ReLU,
    }
    mlp_kwargs_outputs = {
        "depth" : 1,
        "num_cells" : outputLayerCellNumber,
        "activation_class" : nn.ReLU,
    }
    net = DuelingMlpDQNet(actionSpec.shape[-1], 1, mlp_kwargs_features, mlp_kwargs_outputs).to(device)

    actor = QValueActor(net, in_keys=["observation"], spec=actionSpec).to(device)

    actor(fakeTensordict)

    # Loading weights
    actor.load_state_dict(torch.load(policyPath))

    observationNorm = dataRecorded["observationNormalizationFactor"][0]
    observationNorm["loc"] = observationNorm["loc"].to(device)
    observationNorm["scale"] = observationNorm["scale"].to(device)

    # Data containers instantiation
    rolloutTotalReward = []
    rolloutTimeStep = []

    # Rollout Loop
    with set_exploration_type(ExplorationType.MODE), torch.no_grad():
        for i in range(rolloutNumber):
            # Rollout intialization
            t = 0
            totalReward = 0
            terminated, truncated = False, False
            observation, info = environment.reset()

            while not (terminated or truncated) and t < maxTimeStep:

                # Observation Noise
                observationTorchRL = torch.tensor(observation, device=device)

                # Observation Normalisation
                normNoisedObservation = observationNormalisation(
                    observationTorchRL, observationNorm, device
                )

                # Action Inference
                actionTorchRL = actor.forward(normNoisedObservation)

                # Action Noise
                actionGym = actionTorchRLToGym(actionTorchRL, 1)

                # Environment Step
                observation, reward, terminated, truncated, _ = environment.step(actionGym)

                # Data recording
                totalReward += reward
                t += 1

            rolloutTotalReward.append(totalReward)
            rolloutTimeStep.append(t)

    # Creating Plot
    # Reward Statistics values
    specificData["reward"] = {
        "mean" : float(np.mean(rolloutTotalReward)),
        "median" : float(np.median(rolloutTotalReward)),
        "std" : float(np.std(rolloutTotalReward))
    } 
    specificData["length"] = {
        "mean" : float(np.mean(rolloutTimeStep)),
        "median" : float(np.median(rolloutTimeStep)),
        "std" : float(np.std(rolloutTimeStep))
    } 

    return specificData

def main(args) -> None:
    # Loading of the config .yaml
    with open(args.p, "r") as f:
        parameters = yaml.load(f, Loader=yaml.loader.SafeLoader)

    # Folder creation
    currentTime = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    mainPath = "analysis/inferenceNN/Data/" + str(currentTime)
    os.mkdir(mainPath)
    mainPath += "/"

    workerNumber = parameters["Global Parameters"]["workerNumber"]
    mainDir = parameters["Global Parameters"]["mainDir"]
    policyIndiceList = parameters["Global Parameters"]["policyIndiceList"]

    argGenerator = inferenceDataArgGenerator(parameters, mainDir, policyIndiceList)

    with multiprocessing.Pool(workerNumber) as p:
        resList = list(tqdm.tqdm(p.imap_unordered(actor, argGenerator.generateArgs()), total=len(argGenerator)))

    with open(mainPath + "resList.pkl", "wb") as f:
        pkl.dump(resList, f)


if __name__ == "__main__":
    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        action="store",
        default="analysis/inferenceNN/config.yaml",
    )
    parser.add_argument("-nosave", action="store_true")
    args = parser.parse_args()

    main(args)
