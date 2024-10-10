import os
import argparse
import yaml
import sys
import copy
import time
import tqdm
import datetime
import pickle as pkl

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

from torch import nn

from torchrl.collectors import SyncDataCollector

from torchrl.envs import (
    EnvCreator,
    ExplorationType,
    ParallelEnv,
    RewardScaling,
    StepCounter,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import (
    Compose,
    ObservationNorm,
    TransformedEnv,
    RewardSum,
)

from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import RandomSampler

from torchrl.envs.utils import ExplorationType

import torch
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, MultiStep, TensorDictReplayBuffer
from torchrl.envs import (
    EnvCreator,
    ExplorationType,
    ParallelEnv,
    RewardScaling,
    StepCounter,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import (
    RewardSum,
    Compose,
    ObservationNorm,
    TransformedEnv,
)
from torchrl.modules import DuelingMlpDQNet, QValueActor

from torchrl.objectives import DQNLoss, SoftUpdate

from tensordict.nn import TensorDictSequential

from utils.modules.SMActorModule import ESMGreedyModule

from utils.tools.Parser import trainingDataArgGenerator
from utils.tools.Recorder import Recorder

from utils.state_machine.common.PoliticInstantiator import politicInstantiationTensordict

def envInstantiator(
    maxStep: int,
    workerNumber=1,
    obs_norm_sd=None,
    rewardLoc=0,
    rewardScale=1,
    device="cpu",
):
    if obs_norm_sd is None:
        obs_norm_sd = {"standard_normal": True}

    if workerNumber > 1:
        base_env = ParallelEnv(
            workerNumber,
            EnvCreator(
                lambda: GymEnv(
                    "LunarLander-v2",
                    device=device,
                )
            ),
        )
    else:
        base_env = GymEnv(
            "LunarLander-v2",
            device=device,
        )

    env = TransformedEnv(
        base_env,
        Compose(
            ObservationNorm(in_keys=["observation"], **obs_norm_sd),
            StepCounter(max_steps=maxStep),
            RewardSum(),
            RewardScaling(loc=rewardLoc, scale=rewardScale, standard_normal=True),
        ),
    )
    return env


def observationNormalizationFactorEstimation(
    maxStep: int,
    observationNormIndice: int,
):
    testEnv = envInstantiator(maxStep, device="cpu")
    testEnv.transform[observationNormIndice].init_stats(
        num_iter=10000, reduce_dim=0, cat_dim=0
    )
    obsservationNorm = testEnv.transform[observationNormIndice].state_dict()
    return obsservationNorm


def envDimension(
    maxStep: int,
    observationNormParam: dict,
):
    testEnv = envInstantiator(maxStep, 1, observationNormParam, device="cpu")
    observationSpec, actionSpec = testEnv.observation_spec, testEnv.action_spec
    return observationSpec, actionSpec, testEnv.fake_tensordict()


def main(args):

    # Loading of theconfigFilePath config .yaml
    with open(args.p, "r") as f:
        parameters = yaml.load(f, Loader=yaml.loader.SafeLoader)
    currentTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create the saving path
    try:
        os.mkdir("training/D3QN/Data")
    except:
        pass
    
    path = "training/D3QN/Data/" + str(currentTime)
    os.mkdir(path)
    path += "/"

    # Parser
    changingParameter = parameters["Changing Parameters"]["parameterName"]
    changingParameterValues = parameters["Changing Parameters"]["Values"]
    repetition = parameters["Changing Parameters"]["repetitionPerTuple"]
    grid = parameters["Changing Parameters"]["gridSearch"]

    argGenerator = trainingDataArgGenerator(
        parameters, changingParameter, changingParameterValues, repetition, grid
    )

    # Creating data for the multi train
    allDataRecorded = []
    repetitionResultList = []
    allMeanRolloutReward = []
    allValueUsed = []
    allStateProportion = []
    allEpisodeLength = []

    for changedValue, runParameters, endRepetition in argGenerator.argsGenerate(plotArg=True):

        # Hyperparameters parsing
        # Parallelization Parameters
        rolloutDevice = runParameters["Parallelization Parameters"]["rolloutDevice"]
        lossDevice = runParameters["Parallelization Parameters"]["lossDevice"]
        replayBufferDevice = runParameters["Parallelization Parameters"]["replayBufferDevice"]
        workerNumber = runParameters["Parallelization Parameters"]["workerNumber"]
        # Network Parameters
        commonLayersCellNumber = runParameters["Network Parameters"]["commonLayersCellNumber"]
        outputLayerCellNumber = runParameters["Network Parameters"]["outputLayerCellNumber"]
        initialBias = runParameters["Network Parameters"]["initialBias"]
        # Optimizer Parameters
        learningRate = runParameters["Optimizer Parameters"]["learningRate"]
        betas = runParameters["Optimizer Parameters"]["betas"]
        maxGradNorm = runParameters["Optimizer Parameters"]["maxGradNorm"]
        weightDecay = runParameters["Optimizer Parameters"]["weightDecay"]
        # Training Loop Parameters
        totalTimeStep = runParameters["Training Loop Parameters"]["totalTimeStep"]
        maxTimeStep = runParameters["Training Loop Parameters"]["maxTimeStep"]
        rolloutBatchTimeStep = runParameters["Training Loop Parameters"]["rolloutBatchTimeStep"]
        updateEpoch = runParameters["Training Loop Parameters"]["updateEpoch"]
        updateBatchTimeStep = runParameters["Training Loop Parameters"]["updateBatchTimeStep"]
        # Save and Test Parameters
        saveNumber = runParameters["Test Parameters"]["saveNumber"]
        # RL Parameters
        gamma = runParameters["RL Parameters"]["gamma"]
        nStep = runParameters["RL Parameters"]["nStep"]
        epsInit = runParameters["RL Parameters"]["epsInit"]
        epsEnd = runParameters["RL Parameters"]["epsEnd"]
        annealingEpsStep = parameters["RL Parameters"]["lastStepAnnealing"]
        tau = runParameters["RL Parameters"]["tau"]
        replayBufferSize = runParameters["RL Parameters"]["replayBufferSize"]
        initRandomStep = runParameters["RL Parameters"]["initRandomStep"]
        rewardLoc = runParameters["RL Parameters"]["rewardLoc"]
        rewardScale = runParameters["RL Parameters"]["rewardScale"]
        # State Machine Parameters
        smName = parameters["State Machine Parameters"]["smName"]  # FreeFall, Break, AntiRotate, Stabilise, Land
        stateMachineEps = parameters["State Machine Parameters"]["stateMachineEps"]
        # Save and Test Values
        updateStep = totalTimeStep // rolloutBatchTimeStep
        saveFrequency = updateStep // saveNumber

        # Environment parameters
        observationNorm = observationNormalizationFactorEstimation(maxTimeStep, 0)
        observationSpec, actionSpec, fakeTensordict = envDimension(maxTimeStep, observationNorm)

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
        net = DuelingMlpDQNet(actionSpec.shape[-1], 1, mlp_kwargs_features, mlp_kwargs_outputs).to(rolloutDevice)
        net.value[-1].bias.data.fill_(initialBias)

        actor = QValueActor(net, in_keys=["observation"], spec=actionSpec).to(rolloutDevice)

        actor(fakeTensordict.to(rolloutDevice))

        # SM Instantiation
        smPolitic = politicInstantiationTensordict(smName, observationNorm)

        exploration_module = ESMGreedyModule(
            spec=actionSpec,
            stateMachine=smPolitic,
            annealing_num_steps = annealingEpsStep,
            workerNumber=workerNumber,
            eps_init = epsInit,
            eps_end = epsEnd,
            stateMachineEps= stateMachineEps,
        )

        actorExplore = TensorDictSequential(actor, exploration_module)

        dataCollector = SyncDataCollector(
            create_env_fn=envInstantiator(
                maxTimeStep,
                workerNumber,
                observationNorm,
                rewardLoc,
                rewardScale,
                rolloutDevice,
            ),
            policy=actorExplore,
            frames_per_batch=rolloutBatchTimeStep,
            total_frames=totalTimeStep + initRandomStep,
            exploration_type=ExplorationType.RANDOM,
            device=rolloutDevice,
            storing_device=rolloutDevice,
            split_trajs=False,
            postproc=MultiStep(gamma=gamma, n_steps=nStep),
        )

        replayBuffer = TensorDictReplayBuffer(
            sampler=RandomSampler(),
            batch_size=updateBatchTimeStep,
            storage=LazyMemmapStorage(replayBufferSize, device=replayBufferDevice),
            prefetch=updateEpoch,
        )

        lossModule = DQNLoss(actor, delay_value=True)
        lossModule.make_value_estimator(gamma=gamma)
        targetUpdater = SoftUpdate(lossModule, eps=1-tau)

        optimizer = torch.optim.Adam(lossModule.parameters(), lr=learningRate, weight_decay=weightDecay, betas=betas)

        collectedFrames = 0

        # Recorder Instantiation
        recorder = Recorder(
            actor.to("cpu"), optimizer, path + str(changedValue), [args.p]
        )

        # Normalization record
        recorder.addData("observationNormalizationFactor", observationNorm)
        recorder.addData("observationSpec", observationSpec)
        recorder.addData("actionSpec", actionSpec)

        # Random frame collection
        print("Replay buffer initialization")
        for batch in dataCollector:
            batchDataReshape = batch.reshape(-1).to(replayBufferDevice)
            replayBuffer.extend(batchDataReshape)
            collectedFrames += rolloutBatchTimeStep
            if collectedFrames > initRandomStep:
                break

        # Progress bar Instantiation
        progressBar = tqdm.tqdm(total=totalTimeStep, leave=False, position=1)
        lastEpisodeRewardsMean = -300
        lastEpidodeLength = 200
        nbState = len(smPolitic.politicList)

        # Timer
        trainingTime = 0
        rolloutTime = 0
        movingLossModuleTime = 0
        startTime = time.time()
        startRolloutTime = time.time()

        # Training loop
        print("Training loop start")
        for i, batch in enumerate(dataCollector):
            endRolloutTime = time.time()
            rolloutTime += endRolloutTime - startRolloutTime

            # Batch Data record
            episodeRewards = batch["next", "episode_reward"][batch["next", "done"]]
            if len(episodeRewards) > 0:
                recorder.addData("trainEpisodeReward", episodeRewards.mean().item())
                recorder.addData("trainEpisodeLength", batch["next", "step_count"][batch["next", "done"]].to(torch.float32).mean().item())
                lastEpisodeRewardsMean = episodeRewards.mean().item()
                lastEpidodeLength = batch["next", "step_count"][batch["next", "done"]].to(torch.float32).mean().item()
            else:
                recorder.addData("trainEpisodeReward", lastEpisodeRewardsMean)
                recorder.addData("trainEpisodeLength", lastEpidodeLength)
            recorder.addData(
                "meanStepReward",
                torch.mean((batch["next", "reward"] * rewardScale + rewardLoc))
                .detach()
                .to("cpu")
                .item(),
            )

            # Add reward according to eps per state
            batch = batch.reshape(-1)

            recorder.addData("state", batch["state"])

            # Add batch to replay buffer
            batchDataReshape = batch.to(replayBufferDevice)
            replayBuffer.extend(batchDataReshape)

            # Changing device for training
            startMovingLossTime = time.time()

            lossModule = lossModule.to(lossDevice)

            batch = batch.to(lossDevice)

            for _ in tqdm.tqdm(range(updateEpoch), position=2, leave=False):
                batch = replayBuffer.sample()
                batchLossDevice = batch.to(lossDevice)

                movingLossModuleTime += time.time() - startMovingLossTime
                startTrainingTime = time.time()

                loss = lossModule.forward(batchLossDevice)

                # Backward Pass
                loss["loss"].backward()
                torch.nn.utils.clip_grad_norm_(lossModule.parameters(), maxGradNorm)

                # Network Update
                optimizer.step()
                optimizer.zero_grad()

                #Update priority
                replayBuffer.update_tensordict_priority(batchLossDevice)

                #Update target
                targetUpdater.step()

                trainingTime += time.time() - startTrainingTime
                startMovingLossTime = time.time()

            # Changing LossModule to rolloutDevice
            lossModule = lossModule.to(rolloutDevice)

            movingLossModuleTime += time.time() - startMovingLossTime

            # Update Policy
            dataCollector.update_policy_weights_()

            # Update Epsilon
            actorExplore[1].step(rolloutBatchTimeStep)

            # Progress Bar Update
            progressBar.update(rolloutBatchTimeStep)

            # Save
            if i % saveFrequency == 0:
                recorder.save()

            startRolloutTime = time.time()

        # Last save
        recorder.save()

        # Data collector release
        dataCollector.shutdown()

        # Time Taken
        totalTime = time.time() - startTime
        print("Total Time taken : " + str(totalTime))
        print("Rollout time : " + str(rolloutTime))
        print("Training Time : " + str(trainingTime))
        print("Moving Loss Time : " + str(movingLossModuleTime))

        windowSize = totalTimeStep // 500
        X = [i*rolloutBatchTimeStep for i in range(totalTimeStep//rolloutBatchTimeStep)]
        plt.plot(
            X,
            pd.Series.rolling(
                pd.Series(recorder.dataRecorded["meanStepReward"]), windowSize
            ).mean(),
            label="mean",
        )
        plt.legend()
        plt.title("Mean Step Reward over Training rollouts")
        plt.savefig(recorder.savePath + "/Figures/totalRewardRollout.png", dpi=100)
        plt.clf()

        # Training Rollouts' Total Reward Plot
        plt.plot(
            X,
            pd.Series.rolling(
                pd.Series(recorder.dataRecorded["trainEpisodeReward"]), windowSize
            ).mean(),
            label="mean",
        )
        plt.ylim(-500, 300)
        plt.title("Rollouts' Total Reward over Training")
        plt.xlabel("Environment Steps")
        plt.ylabel("Average Reward per episode")
        plt.savefig(recorder.savePath + "/Figures/trainRewardRollout.png", dpi=100)
        plt.clf()


        # State proportion plot
        stateProportion = torch.zeros(nbState, len(recorder.dataRecorded["state"]))
        for j in range(len(recorder.dataRecorded["state"])):
            batchData = recorder.dataRecorded["state"][j]
            for i in range(nbState):
                stateProportion[i][j] = torch.sum(batchData == i).item() / rolloutBatchTimeStep
        fig, ax = plt.subplots()
        stateProportionWindowed = []
        episodeLengthwWindowed = torch.tensor(pd.Series.rolling(pd.Series(recorder.dataRecorded["trainEpisodeLength"]), windowSize, min_periods=1).mean())
        for proportion in stateProportion:
            stateProportionWindowed.append(torch.tensor(
                    pd.Series.rolling(pd.Series(proportion), windowSize, min_periods=1)
                    .mean()
                    .values
                )
            )
        x = range(len(stateProportionWindowed[0]))
        lastProportion = torch.zeros(len(stateProportionWindowed[0]))
        for i in range(nbState):
            ax.fill_between(X, lastProportion, lastProportion + stateProportionWindowed[i], label="Etat " + str(i))
            lastProportion +=  stateProportionWindowed[i]
        plt.title("State distribution over training")
        plt.legend()
        plt.savefig(recorder.savePath + "/Figures/stateDist.png", dpi=100)
        plt.clf()

        fig, ax = plt.subplots()

        x = range(len(stateProportionWindowed[0]))
        lastProportion = torch.zeros(len(stateProportionWindowed[0]))
        for i in range(nbState):
            ax.fill_between(X, lastProportion * episodeLengthwWindowed, (lastProportion + stateProportionWindowed[i]) * episodeLengthwWindowed, label="Etat " + str(i))
            lastProportion +=  stateProportionWindowed[i]
        plt.title("Average timesteps per State per episode over training")
        plt.ylim(0,1000)
        plt.ylabel("Average Timesteps spent per state")
        plt.xlabel("Environment Steps")
        plt.legend()
        plt.savefig(recorder.savePath + "/Figures/stateDistEpisodeLength.png", dpi=100)
        plt.clf()

        allDataRecorded.append(recorder.dataRecorded)
        repetitionResultList.append(recorder.dataRecorded)
        allStateProportion.append(stateProportion)
        allEpisodeLength.append(copy.deepcopy(recorder.dataRecorded["trainEpisodeLength"]))


        if endRepetition:
            # Plot for all same hyperparameters train
            dataTrainRollout = [
                dataRecorded["trainEpisodeReward"]
                for dataRecorded in repetitionResultList
            ]

            # Rolling Average of mean step Reward on all repetition
            dataTensors = [
                dataRecorded["meanStepReward"] for dataRecorded in repetitionResultList
            ]
            rollingTensor = torch.stack(
                [
                    torch.tensor(
                        pd.Series.rolling(pd.Series(dataTensor), windowSize)
                        .mean()[windowSize - 1 :]
                        .values
                    )
                    for dataTensor in dataTensors
                ]
            )
            minTensor = torch.min(rollingTensor, dim=0).values
            maxTensor = torch.max(rollingTensor, dim=0).values
            meanTensor = torch.mean(rollingTensor, dim=0)
            x = range(len(minTensor))

            fig, ax = plt.subplots()
            ax.fill_between(X, minTensor, maxTensor, alpha=0.2, color="b")
            ax.plot(meanTensor, color="b")
            plt.savefig(
                path + "multiPlotMeanStep" + str(changedValue) + ".png", dpi=100
            )
            plt.clf()

            # Train Results on all repetition
            minDataMeanTrain = np.min(dataTrainRollout, axis=0)
            maxDataMeanTrain = np.max(dataTrainRollout, axis=0)
            meanDataMeanTrain = np.mean(dataTrainRollout, axis=0)
            x = range(len(minDataMeanTrain))

            fig, ax = plt.subplots()
            ax.fill_between(X, minDataMeanTrain, maxDataMeanTrain, alpha=0.2, color="b")
            ax.plot(meanDataMeanTrain, color="b")
            plt.ylim(-500, 300)
            plt.savefig(
                path + "multiPlotTrainRollout" + str(changedValue) + ".png", dpi=100
            )
            plt.clf()

            # Record Mean Training of repetition
            allMeanRolloutReward.append(meanDataMeanTrain)
            allValueUsed.append(str(changedValue))

            repetitionResultList = []

    #Loading Baseline
    for pathData, name in zip(runParameters["Plot Parameters"]["baseLineData"], runParameters["Plot Parameters"]["baseLineName"]):
        allValueUsed.append(name)
        _, dataLoaded = pkl.load(open(pathData, "rb"))
        allMeanRolloutReward += dataLoaded


    for value, plot in zip(allValueUsed, allMeanRolloutReward):
        plt.plot(pd.Series.rolling(pd.Series(plot), windowSize).mean(), label=value)
    plt.ylim(-500, 300)
    plt.title("Plot of the mean over all repetition of every training parameters")
    plt.xlabel("TimeSteps")
    plt.ylabel("Mean episode reward over all repetition")
    plt.legend()
    plt.savefig(path + "trainingEpisodeMeanOverAllParameters.png", dpi=100)

    with open(path + "resultData.pkl", "wb") as f:
        pkl.dump((allValueUsed, allMeanRolloutReward, allStateProportion, allEpisodeLength), f)


if __name__ == "__main__":
    # Config file path parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        action="store",
        default="training/D3QN/config.yaml",
    )
    args = parser.parse_args()

    main(args)