#!/usr/bin/python
import sys
import os
import time
import datetime
import pickle
import shutil
import copy

sys.path.append(os.getcwd())

import torch

from typing import Tuple


class Recorder:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        savePath: str,
        extraFileToSave: Tuple,
    ) -> None:
        self.initTime = time.time()

        self.model = model
        self.optimizer = optimizer
        self.stepCount = 0

        self.dataRecorded = {}

        self.dateTime = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        self.savePath = savePath + self.dateTime
        os.mkdir(self.savePath)
        for filePath in extraFileToSave:
            shutil.copy(filePath, self.savePath)
        os.mkdir(self.savePath + "/Models")
        os.mkdir(self.savePath + "/Optimizers")
        os.mkdir(self.savePath + "/Figures")

    def addData(
        self,
        key: str,
        value,
    ) -> None:
        if key in self.dataRecorded.keys():
            self.dataRecorded[key].append(value)
        else:
            self.dataRecorded[key] = [value]

    def addDataSequence(
        self,
        key: str,
        sequence,
    ) -> None:
        if key in self.dataRecorded.keys():
            self.dataRecorded[key] += sequence
        else:
            self.dataRecorded[key] = sequence

    def save(
        self,
        modelName=None,
    ) -> str:
        # Creating Paths
        if modelName == None:
            modelPath = (
                self.savePath + "/Models/modelSave" + str(self.stepCount) + ".pt"
            )
            optimizerPath = (
                self.savePath
                + "/Optimizers/optimizerSave"
                + str(self.stepCount)
                + ".pt"
            )
        else:
            modelPath = self.savePath + "/Models/modelSave" + modelName + ".pt"
            optimizerPath = (
                self.savePath + "/Optimizers/optimizerSave" + modelName + ".pt"
            )

        # Saving to Paths
        torch.save(copy.deepcopy(self.model).to("cpu").state_dict(), modelPath)
        torch.save(self.optimizer.state_dict(), optimizerPath)
        with open(self.savePath + "/dataRecordedDict.pkl", "wb") as f:
            pickle.dump(self.dataRecorded, f)

        if modelName == None:
            self.stepCount += 1

        return modelPath
