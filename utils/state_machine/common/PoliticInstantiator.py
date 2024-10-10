from utils.state_machine.common.StateMachinePolitic import StateMachinePolitic
from utils.state_machine.common.SMPiTensorDict import SMPiTensorDict

import utils.state_machine.instances.LunarSMPiHeur as SMPiHeur
    
def politicInstantiation(politicName: str, epsilon):
    if politicName == "Heur":
        politic = StateMachinePolitic(
            SMPiHeur.StateMachineLunarLander(0),
            [
                SMPiHeur.High(),
                SMPiHeur.Instable(),
                SMPiHeur.Low(),
                SMPiHeur.Grounded(),
            ]
        )
        return politic

def politicInstantiationTensordict(politicName: str, observationNorm):
    if politicName == "Heur":
        politic = SMPiTensorDict(
            SMPiHeur.StateMachineLunarLander(0),
            [
                SMPiHeur.High(),
                SMPiHeur.Instable(),
                SMPiHeur.Low(),
                SMPiHeur.Grounded(),
            ],
            observationNorm,
        )
        return politic