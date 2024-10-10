import torch

from tensordict import TensorDict

from utils.state_machine.common.StateMachinePolitic import StateMachinePolitic

class SMPiTensorDict(StateMachinePolitic):
    def __init__(self, stateMachine, politicList, observationNorm, ray=True):
        super().__init__(stateMachine, politicList)
        
        self.observationNormScale = observationNorm["scale"].to("cpu")
        self.observationNormLoc = observationNorm["loc"].to("cpu")
        self.ray = ray
    
    def actAndTransit(self, observation):
        self.transition(None, observation)
        return self.act(observation)
    
    def forward(self, observation : TensorDict)-> TensorDict:
        if self.ray:
            observation = observation.to("cpu")
            observationsUnnormalized = (observation * self.observationNormScale) + self.observationNormLoc
            indices = self.actAndTransit(observationsUnnormalized)
            actions = torch.zeros(4)
            actions[indices] = 1
        else:
            observations = observation["observation"].to("cpu")
            observationsUnnormalized = (observations * self.observationNormScale) + self.observationNormLoc
            indices = [self.transitionAct(observation) for observation in observationsUnnormalized]
            actions = torch.zeros_like(observation["action"])
            actions[torch.arange(len(indices)),indices] = 1
        return actions