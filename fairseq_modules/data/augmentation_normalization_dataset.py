from augment import EffectChain

from fairseq.data import BaseWrapperDataset, FairseqDataset
import numpy as np
from typing import Dict, List, Tuple, Union
import torch
import torch.nn.functional as F

SAMPLING_RATE = 16000
ECHO_GAININ = 0.8
ECHO_GAINOUT = 0.9


class AugmentationNormalizationDataset(BaseWrapperDataset):

    def __init__(self,
    dataset: FairseqDataset,
    effects_info: Dict[str, Union[List, Dict]],
    p_augm: float,
    normalize: bool,
    is_train_split: bool) -> None:

        super(AugmentationNormalizationDataset, self).__init__(dataset)

        self.dataset = dataset
        self.effects_info = effects_info
        self.p_augm = p_augm
        self.normalize = normalize
        self.is_train_split = is_train_split

        self.sr = SAMPLING_RATE
        self.echo_gainin = ECHO_GAININ
        self.echo_gainout = ECHO_GAINOUT

        self.info = {"rate": self.sr}
        
        # add effects
        self.avai_effects = []
        if self.effects_info["tempo"][0] != self.effects_info["tempo"][1]:
            self.avai_effects.append("tempo")
        if self.effects_info["pitch"][0] != self.effects_info["pitch"][1]:
            self.avai_effects.append("pitch")
        cond_delay = self.effects_info["echo"]["delay"][0] != self.effects_info["echo"]["delay"][1]
        cond_decay = self.effects_info["echo"]["decay"][0] != self.effects_info["echo"]["decay"][1]
        if cond_delay and cond_decay:
            self.avai_effects.append("echo")

    def __getitem__(self, index: int) -> Tuple[str, torch.tensor, torch.tensor]:

        # (make sure input is a torch tensor)
        _id, input_tensor, target_tensor = self.dataset[index]

        # apply effects or keep the original audiowave
        if self.is_train_split and np.random.rand() < self.p_augm:
            input_tensor = self._augment(input_tensor)

        # normalize audiowave
        if self.normalize:
            input_tensor = self._normalize(input_tensor)

        return (_id, input_tensor, target_tensor)

    def _augment(self, input_tensor: torch.tensor) -> torch.tensor:

        # init empty chain
        effect_chain = EffectChain()

        if "pitch" in self.avai_effects:

            effect_chain = effect_chain.pitch(
                np.random.randint(*self.effects_info["pitch"])).rate(self.sr)

        if "tempo" in self.avai_effects:

            effect_chain = effect_chain.tempo(
                np.random.uniform(*self.effects_info["tempo"]))

        if "echo" in self.avai_effects:

            effect_chain = effect_chain.echo(
                self.echo_gainin,
                self.echo_gainout,
                np.random.randint(*self.effects_info["echo"]["delay"]),
                np.random.uniform(*self.effects_info["echo"]["decay"]))

        # apply effects and reduce channel dimension that is created by default
        input_tensor_augm = effect_chain.apply(input_tensor,
            src_info = self.info, target_info = self.info).squeeze(0)

        # sox might misbehave sometimes by giving nan/inf if sequences are too short (or silent)
        if torch.isnan(input_tensor_augm).any() or torch.isinf(input_tensor_augm).any():
            return input_tensor
        else:
            return input_tensor_augm

    def _normalize(self, input_tensor: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            input_tensor = F.layer_norm(input_tensor, input_tensor.shape)
        return input_tensor

    def worker_init_fn(worker_id: int) -> None:
        np.random.seed(np.random.get_state()[1][0] + worker_id)