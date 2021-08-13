import numpy as np
from omegaconf import II
from dataclasses import dataclass, field
from typing import Optional

from fairseq.data import ConcatDataset, SubsampleDataset
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDataset
from fairseq.tasks import register_task
from fairseq.tasks.speech_to_text import (
    SpeechToTextTask,
    SpeechToTextTaskConfig,
)

from fairseq_modules.data.augmentation_normalization_dataset import AugmentationNormalizationDataset


@dataclass
class SpeechToTextModTaskConfig(SpeechToTextTaskConfig):
    sample_ratios: str = field(
        default="1",
        metadata={"help": "sample ratios of the train subsets"}
    )

    da_p_augm: float = field(
        default="1",
        metadata={"help": "The probability that data augmentation is applied to an example."}
    )

    da_tempo: str = field(
        default="1,1",
        metadata={"help": "The range from which to sample the tempo factor during data augmentation"}
    )

    da_pitch: str = field(
        default="0,0",
        metadata={"help": "The range from which to sample the pitch value during data augmentation. \
            Measured in cents (i.e. 100ths of a semitone)"}
    )

    da_echo_delay: str = field(
        default="0,0",
        metadata={"help": "The range from which to sample the echo delay value during data augmentation. \
            Measured in milliseconds"}
    )

    da_echo_decay: str = field(
        default="0,0",
        metadata={"help": "The range from which to sample the echo decay factor during data augmentation."}
    )

    normalize: bool = field(
        default=True,
        metadata={"help": "Whether to normalize the audiowave to zero mean and unit variance."}
    )

    interactive_tgt_lang: Optional[str] = field(
        default=None,
        metadata={"help": "Target language to be used with Fairseq's interactive mode."}
    )

    seed: int = II("common.seed")
    max_tokens: int = II("dataset.max_tokens")


@register_task("speech_to_text_iwslt21", dataclass=SpeechToTextModTaskConfig)
class SpeechToTextModTask(SpeechToTextTask):

    def __init__(self, cfg, tgt_dict):
        super().__init__(cfg, tgt_dict)

        # effect parameters for data augmentation
        self.da_effects_info = {
            "tempo": list(map(float, cfg.da_tempo.split(","))),
            "pitch": list(map(int, cfg.da_pitch.split(","))),
            "echo": {
                "delay": list(map(int, cfg.da_echo_delay.split(","))),
                "decay": list(map(float, cfg.da_echo_decay.split(",")))
            }
        }

        self.max_src_len = min(cfg.max_source_positions, cfg.max_tokens)
        self.tgt_dict = tgt_dict
        self.interactive_tgt_lang = cfg.interactive_tgt_lang

        assert len(set(self.da_effects_info["echo"]["delay"])) == \
            len(set(self.da_effects_info["echo"]["decay"])), \
            "Specify ranges for both parameters of echo (delay & decay) or for none"

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        if is_train_split:
            datasets = []
            splits = split.split(',')
            sample_ratios = \
                [float(r) for r in self.cfg.sample_ratios.split(',')]
            if sample_ratios == [1]:
                sample_ratios = sample_ratios * len(splits)
            assert len(splits) == len(sample_ratios), \
                "The nº of splits and sample_ratios must be equal."
            for s, r in zip(splits, sample_ratios):
                super().load_dataset(s, epoch, combine, **kwargs)
                if 0 < r < 1:
                    datasets.append(SubsampleDataset(self.datasets.pop(s), r))
                else:
                    datasets.append(self.datasets.pop(s))
            upsample_ratios = [int(r) if r > 1 else 1 for r in sample_ratios]
            self.datasets[split] = ConcatDataset(datasets, upsample_ratios)
        else:
            super().load_dataset(split, epoch, combine, **kwargs)

        self.datasets[split] = AugmentationNormalizationDataset(
            self.datasets[split], self.da_effects_info,
            self.cfg.da_p_augm, self.cfg.normalize,
            self.max_src_len, is_train_split)

    def begin_epoch(self, epoch, model):
        super().begin_epoch(epoch, model)
        np.random.seed(self.cfg.seed + epoch)
        if epoch == 1:
            return
        for split in list(self.datasets.keys()):
            if split.startswith("train"):
                # Perform a new subsampling at each epoch
                self.load_dataset(split, epoch)

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        assert self.interactive_tgt_lang is not None
        return SpeechToTextDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths,
            tgt_texts=([""] * len(src_tokens)),
            tgt_langs=([self.interactive_tgt_lang] * len(src_tokens)),
            tgt_dict=self.tgt_dict
        )
