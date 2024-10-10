import numpy as np


class StateMachine:
    def __init__(
        self,
        initialState: int,
    ):
        self.startState = initialState
        self.currentState = initialState

    def getCurrentState(
        self,
    ) -> int:
        return self.currentState

    def transition(
        self,
        action: np.ndarray,
        observation: np.ndarray,
    ) -> None:
        raise NotImplementedError
    
    def reset(self,
              )-> None:
        self.currentState = self.startState
