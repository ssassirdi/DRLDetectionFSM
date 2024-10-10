import copy
from typing import Optional

import numpy as np
import torch

from tensordict.nn import (
    TensorDictModuleBase,
)
from tensordict.tensordict import TensorDictBase
from tensordict.utils import expand_as_right, NestedKey

from torchrl.data.tensor_specs import CompositeSpec, TensorSpec
from torchrl.envs.utils import exploration_type, ExplorationType

from utils.state_machine.common.SMPiTensorDict import SMPiTensorDict


    
class ESMGreedyModule(TensorDictModuleBase):
    def __init__(
        self,
        spec: TensorSpec,
        stateMachine : SMPiTensorDict,
        annealing_num_steps,
        workerNumber : int,
        eps_init: float = 1.0,
        eps_end: float = 0.1,
        stateMachineEps: float = 0,
        *,
        action_key: Optional[NestedKey] = "action",
        action_mask_key: Optional[NestedKey] = None,
    ):
        self.action_key = action_key
        self.action_mask_key = action_mask_key
        in_keys = [self.action_key]
        if self.action_mask_key is not None:
            in_keys.append(self.action_mask_key)
        self.in_keys = in_keys
        self.out_keys = [self.action_key]

        super().__init__()

        self.register_buffer("eps_init", torch.tensor([eps_init]))
        self.register_buffer("eps_end", torch.tensor([eps_end]))
        self.nbState = len(stateMachine.politicList)
        if self.eps_end > self.eps_init:
            raise RuntimeError("eps should decrease over time or be constant")
        self.eps = eps_init

        if spec is not None:
            if not isinstance(spec, CompositeSpec) and len(self.out_keys) >= 1:
                spec = CompositeSpec({action_key: spec}, shape=spec.shape[:-1])
        self._spec = spec
        self.stateMachineEps = stateMachineEps
        self.annealing_num_steps = annealing_num_steps
        self.stateMachineList = [copy.deepcopy(stateMachine) for _ in range(workerNumber)]
        self.workerNumber = workerNumber
        
    @property
    def spec(self):
        return self._spec

    def step(self, frames: int = 1) -> None:
        self.eps = max(self.eps_end.item(),(self.eps - (frames * (self.eps_init - self.eps_end) / self.annealing_num_steps)).item())



    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if exploration_type() == ExplorationType.RANDOM or exploration_type() is None:
            if isinstance(self.action_key, tuple) and len(self.action_key) > 1:
                action_tensordict = tensordict.get(self.action_key[:-1])
                action_key = self.action_key[-1]
            else:
                action_tensordict = tensordict
                action_key = self.action_key
            out = action_tensordict.get(action_key)
            outSMValue = torch.zeros_like(out)
            state = torch.zeros(self.workerNumber)
            for i in range(self.workerNumber):
                if tensordict["step_count"][i] == 0:
                    self.stateMachineList[i].reset()
                outSMValue[i] = self.stateMachineList[i].forward(tensordict["observation"][i].squeeze())
                currState = self.stateMachineList[i].stateMachine.getCurrentState()
                state[i] = currState
            tensordict["state"] = state
            rand = torch.rand(action_tensordict.shape, device=action_tensordict.device)
            condSM = ( rand < self.eps * self.stateMachineEps ).to(out.device)
            condSM = expand_as_right(condSM, out)
            condUni = torch.logical_and((rand < self.eps), rand > self.eps * self.stateMachineEps)
            condUni = expand_as_right(condUni, out)
            cond = (rand < self.eps).to(out.device)
            cond = torch.logical_not(expand_as_right(cond, out))
            spec = self.spec
            if spec is not None:
                if isinstance(spec, CompositeSpec):
                    spec = spec[self.action_key]
                if spec.shape != out.shape:
                    # In batched envs if the spec is passed unbatched, the rand() will not
                    # cover all batched dims
                    if (
                        not len(spec.shape)
                        or out.shape[-len(spec.shape) :] == spec.shape
                    ):
                        spec = spec.expand(out.shape)
                    else:
                        raise ValueError(
                            "Action spec shape does not match the action shape"
                        )
                if self.action_mask_key is not None:
                    action_mask = tensordict.get(self.action_mask_key, None)
                    if action_mask is None:
                        raise KeyError(
                            f"Action mask key {self.action_mask_key} not found in {tensordict}."
                        )
                    spec.update_mask(action_mask)

                out = condSM * outSMValue.to(out.device) + condUni * spec.rand().to(out.device) + cond * out
            else:
                raise RuntimeError("spec must be provided to the exploration wrapper.")
            action_tensordict.set(action_key, out)
        return tensordict
    
class ExploState2GreedyModule(TensorDictModuleBase):
    def __init__(
        self,
        spec: TensorSpec,
        workerNumber : int,
        windowAverage : int,
        alphaOubli : float,
        timeStepInit : int,
        timeStepMax : int,
        rolloutBatchTimeStep : int,
        valeurMax : float,
        recorder,
        *,
        action_key: Optional[NestedKey] = "action",
        action_mask_key: Optional[NestedKey] = None,
    ):
        self.action_key = action_key
        self.action_mask_key = action_mask_key
        in_keys = [self.action_key]
        if self.action_mask_key is not None:
            in_keys.append(self.action_mask_key)
        self.in_keys = in_keys
        self.out_keys = [self.action_key]

        super().__init__()

        self.recorder = recorder
        self.epsState = 0

        if spec is not None:
            if not isinstance(spec, CompositeSpec) and len(self.out_keys) >= 1:
                spec = CompositeSpec({action_key: spec}, shape=spec.shape[:-1])
        self._spec = spec

        self.workerNumber = workerNumber
        self.rolloutBatchTimeStep = rolloutBatchTimeStep
        self.windowAverage = windowAverage
        self.alphaOubli = alphaOubli
        self.timeStepInit = timeStepInit
        self.timeStepMax = timeStepMax
        self.valeurMax = valeurMax
        
    @property
    def spec(self):
        return self._spec

    def resetCount(self)->None:
        self.countState = [0 for _ in range(self.nbState)]

    def step(self, frames: int = 1) -> None:
        windowedMeanLength = np.mean(self.recorder.dataRecorded["trainEpisodeLength"][-self.windowAverage:])
        stateProportion = np.zeros(self.windowAverage)
        for j in range(min(self.windowAverage, len(self.recorder.dataRecorded["state"]))):
            batchData = self.recorder.dataRecorded["state"][-j]
            stateProportion[-j] = torch.sum(batchData == 2).item() / self.rolloutBatchTimeStep
        windowedStateProportion = np.mean(stateProportion)

        epsValueInstant = (self.valeurMax / (self.timeStepMax - self.timeStepInit)) * (windowedMeanLength * windowedStateProportion) - (self.valeurMax / (self.timeStepMax - self.timeStepInit)) * self.timeStepInit
        epsValueInstantClip = max(0., min(self.valeurMax, epsValueInstant))

        self.epsState = (1-self.alphaOubli) * self.epsState + self.alphaOubli * epsValueInstantClip




    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if exploration_type() == ExplorationType.RANDOM or exploration_type() is None:
            if isinstance(self.action_key, tuple) and len(self.action_key) > 1:
                action_tensordict = tensordict.get(self.action_key[:-1])
                action_key = self.action_key[-1]
            else:
                action_tensordict = tensordict
                action_key = self.action_key
            out = action_tensordict.get(action_key)
            # Action 0 conseillée pour l'état 2
            outSMValue = torch.zeros_like(out)
            outSMValue[:, 0] = 1
            rand = torch.rand(action_tensordict.shape, device=action_tensordict.device)

            condSM = (rand < self.epsState).to(out.device)
            condSM = expand_as_right(condSM, out)
            condState2 = tensordict["state"] == 2
            condState2 = expand_as_right(condState2, out)

            condSM = torch.logical_and(condSM, condState2)
            condSM = expand_as_right(condSM, out)

            spec = self.spec

            if spec is not None:
                if isinstance(spec, CompositeSpec):
                    spec = spec[self.action_key]
                if spec.shape != out.shape:
                    # In batched envs if the spec is passed unbatched, the rand() will not
                    # cover all batched dims
                    if (
                        not len(spec.shape)
                        or out.shape[-len(spec.shape) :] == spec.shape
                    ):
                        spec = spec.expand(out.shape)
                    else:
                        raise ValueError(
                            "Action spec shape does not match the action shape"
                        )
                if self.action_mask_key is not None:
                    action_mask = tensordict.get(self.action_mask_key, None)
                    if action_mask is None:
                        raise KeyError(
                            f"Action mask key {self.action_mask_key} not found in {tensordict}."
                        )
                    spec.update_mask(action_mask)
                out = condSM * outSMValue.to(out.device)+ torch.logical_not(condSM) * out
            else:
                raise RuntimeError("spec must be provided to the exploration wrapper.")
            action_tensordict.set(action_key, out)
        return tensordict