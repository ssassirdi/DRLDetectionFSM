import numpy as np
from typing import Dict, List

from utils.state_machine.common.StateMachine import StateMachine
from utils.state_machine.common.Politic import Politic


class StateMachinePolitic(Politic):
    def __init__(
        self,
        stateMachine: StateMachine,
        politicList: List[Politic],
    ):

        self.transitionNeed = True
        self.stateMachine = stateMachine
        self.politicList = politicList

    def act(
        self,
        observation: np.ndarray,
    ) -> int:
        return self.politicList[self.stateMachine.getCurrentState()].act(observation)

    def transition(
        self,
        action: np.ndarray,
        observation: np.ndarray,
    ) -> None:
        self.stateMachine.transition(action, observation)

    def reset(
        self,
        )->None:
        self.stateMachine.reset()