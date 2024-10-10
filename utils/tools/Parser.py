import os
from itertools import product


class trainingDataArgGenerator:
    def __init__(
        self,
        argDict,
        changingParameters,
        changingParemeterValues,
        repetitionPerValues,
        grid,
    ) -> None:
        self.argDict = argDict
        self.changingParameters = changingParameters
        self.changingParameterValues = changingParemeterValues
        self.repetitionPerValues = repetitionPerValues
        self.grid = grid

    def argsGenerate(self, plotArg=False):
        if self.grid:
            valueTupleGenerator = product(*self.changingParameterValues)
        else:
            valueTupleGenerator = zip(*self.changingParameterValues)

        # Loop for multirun
        for values in valueTupleGenerator:
            for changedParameter, valueParameter in zip(
                self.changingParameters, values
            ):
                self.argDict[changedParameter[0]][changedParameter[1]] = valueParameter
            for i in range(self.repetitionPerValues):
                if plotArg:
                    yield values, self.argDict, i + 1 == self.repetitionPerValues
                else:
                    yield values, self.argDict


class inferenceDataArgGenerator:
    def __init__(self, dictGlobal : dict, mainPath : str, modelList : list) -> None:
        self.listDir = [mainPath + "/" + folderPath for folderPath in os.listdir(mainPath) if (".png" not in folderPath) and (".pkl" not in folderPath)]
        self.modelList = modelList
        self.dictGlobal = dictGlobal

    def generateArgs(self):
        for trainingNumber, folderPath in enumerate(self.listDir):
            for modelNumber in self.modelList:
                specificDict = {
                    "policyPath" : folderPath + "/Models/modelSave" + str(modelNumber) + ".pt",
                    "dataRecordedPath" : folderPath + "/dataRecordedDict.pkl",
                    "modelNumber" : modelNumber,
                    "trainingNumber" : trainingNumber,
                }
                yield self.dictGlobal, specificDict
        
    def __len__(self):
        return len(self.listDir) * len(self.modelList)