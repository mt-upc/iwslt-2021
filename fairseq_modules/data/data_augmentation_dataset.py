from augment import EffectChain

from fairseq.data import BaseWrapperDataset, FairseqDataset
import numpy as np
from typing import Dict, List, Tuple, Union
import torch
import torch.nn.functional as F

SAMPLING_RATE = 16000
ECHO_GAININ = 0.9
ECHO_GAINOUT = 0.8


class DataAugmentationDataset(BaseWrapperDataset):

    def __init__(self,
    dataset: FairseqDataset,
    effects_info: Dict[str, Union[List, Dict]],
    p_augm: float,
    normalize: bool) -> None:

        super(DataAugmentationDataset, self).__init__(dataset)

        self.dataset = dataset
        self.effects_info = effects_info
        self.p_augm = p_augm
        self.normalize = normalize

        self.sr = SAMPLING_RATE
        self.echo_gainin = ECHO_GAININ
        self.echo_gainout = ECHO_GAINOUT

        self.info = {"rate": self.sr}
            
        self.avai_effects = []
        if self.effects_info["tempo"][0] != self.effects_info["tempo"][1]:
            self.avai_effects.append("tempo")
        if self.effects_info["pitch"][0] != self.effects_info["pitch"][1]:
            self.avai_effects.append("pitch")
        cond_delay = self.effects_info["echo"]["delay"][0] != self.effects_info["echo"]["delay"][1]
        cond_decay = self.effects_info["echo"]["decay"][0] != self.effects_info["echo"]["decay"][1]
        if cond_delay and cond_decay:
            self.avai_effects.append("echo")

        # register effect chain function
        self._create_effect_chain()


    def __getitem__(self, index: int) -> Tuple[str, torch.tensor, torch.tensor]:

        # (make sure input is a torch tensor)
        _id, input_tensor, target_tensor = self.dataset[index]

        # apply effects or keep the original audiowave
        if np.random.rand() < self.p_augm:
            input_tensor = self._apply_effects(input_tensor)

        # normalize audiowave
        if self.normalize:
            with torch.no_grad():
                input_tensor = F.layer_norm(input_tensor, input_tensor.shape)

        return (_id, input_tensor, target_tensor)

    def _apply_effects(self, input_tensor: torch.tensor) -> torch.tensor:

        # apply effects and reduce channel dimension that is created by default
        input_tensor_augm = self.effect_chain.apply(input_tensor,
            src_info = self.info, target_info = self.info).squeeze(0)

        # sox might misbehave sometimes by giving nan/inf if sequences are too short (or silent)
        if torch.isnan(input_tensor_augm).any() or torch.isinf(input_tensor_augm).any():
            return input_tensor
        else:
            return input_tensor_augm

    def _create_effect_chain(self) -> None:

        # init empty chain
        self.effect_chain = EffectChain()

        if "pitch" in self.avai_effects:

            def rnd_pitch_value() -> int:
                return np.random.randint(*self.effects_info["pitch"])

            self.effect_chain = self.effect_chain.pitch(rnd_pitch_value).rate(self.sr)

        if "tempo" in self.avai_effects:

            def rnd_tempo_factor() -> float:
                return np.random.uniform(*self.effects_info["tempo"])

            self.effect_chain = self.effect_chain.tempo(rnd_tempo_factor)

        if "echo" in self.avai_effects:

            def rnd_echo_delay() -> int:
                return np.random.randint(*self.effects_info["echo"]["delay"])

            def rnd_echo_decay() -> float:
                return np.random.uniform(*self.effects_info["echo"]["decay"])

            self.effect_chain = self.effect_chain.echo(
                self.echo_gainin, self.echo_gainout, rnd_echo_delay, rnd_echo_decay)