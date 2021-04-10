from augment import EffectChain

from fairseq.data import BaseWrapperDataset, FairseqDataset
import numpy as np
from typing import Dict, List, Tuple, Union
import torch

SAMPLING_RATE = 16000
ECHO_GAIN_IN = 0.8
ECHO_GAIN_OUT = 0.9


class DataAugmentationDataset(BaseWrapperDataset):

    def __init__(self,
    dataset: FairseqDataset,
    effects_params: Dict[str, Union[float, List[float]]],
    p_augm: float,
    segm_len: int) -> None:

        super(DataAugmentationDataset, self).__init__(dataset)

        self.dataset = dataset

        self.sr = SAMPLING_RATE
        self.info = {"rate": self.sr}
        
        self.p_augm = p_augm
        self.segm_len = segm_len

        self.effects_info = effects_params
        self.all_effects = self.effects_info.keys()

        self.echo_gain_in = ECHO_GAIN_IN
        self.echo_gain_out = ECHO_GAIN_OUT

        # set up random parameter generators
        rnd_pitch_value = lambda: np.random.randint(*self.effects_info["pitch"])
        rnd_tempo_factor = lambda: np.random.uniform(*self.effects_info["tempo"])
        rnd_echo_delay_value = lambda: np.random.randint(*self.effects_info["echo_delay"])
        rnd_echo_decay_factor = lambda: np.random.uniform(*self.effects_info["echo_decay"])

        # create general effect chain with randomizable parameters
        self.general_augm_chain = EffectChain().pitch(rnd_pitch_value()).rate(self.sr). \
            tempo(rnd_tempo_factor()).echo(self.echo_gain_in, self.echo_gain_out,
            rnd_echo_delay_value, rnd_echo_decay_factor)

        # create effect chain for segments
        self.segment_augm_chain = EffectChain().time_dropout(max_frames = self.effects_info["time_dropout"])


    def __getitem__(self, index: int) -> Tuple[str, torch.tensor, torch.tensor]:

        # (make sure input is a torch tensor)
        _id, input_tensor, target_tensor = self.dataset[index]

        # apply general effects on the whole example (or keep the original example)
        if np.random.rand() < self.p_augm:
            input_tensor = self._apply_general_effects(input_tensor)

        # apply effects on segments
        input_tensor = self._apply_segment_effects(input_tensor)

        return (_id, input_tensor, target_tensor)

    def _apply_general_effects(self, input_tensor: torch.tensor) -> torch.tensor:

        input_tensor_augm = self.general_augm_chain.apply(input_tensor,
            src_info = self.info, target_info = self.info).squeeze(0)

        # sox might misbehave sometimes by giving nan/inf if sequences are too short (or silent)
        if torch.isnan(input_tensor_augm).any() or torch.isinf(input_tensor_augm).any():
            return input_tensor
        else:
            return input_tensor_augm

    def _apply_segment_effects(self, input_tensor: torch.tensor) -> torch.tensor:

        n_frames = input_tensor.shape[0]

        if n_frames < self.segm_len:

            # adjust length of time dropout to the size of the small example
            reduced_segment_aug_chain = EffectChain().time_dropout(
                max_frames = n_frames / self.segm_len * self.effects_info["time_dropout"])

            input_tensor = reduced_segment_aug_chain.apply(input_tensor,
                src_info = self.info,
                target_info = self.info).squeeze(0)

        else:

            for i in range(0, n_frames - self.segm_len, self.segm_len):
                input_tensor[i: i + self.segm_len] = self.segment_augm_chain.apply(
                    input_tensor[i: i + self.segm_len],
                    src_info = self.info,
                    target_info = self.info).squeeze(0)

        return input_tensor