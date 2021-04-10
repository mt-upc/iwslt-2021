from omegaconf import II
from dataclasses import dataclass, field

from fairseq.data import ConcatDataset
from fairseq.tasks import register_task
from fairseq.tasks.speech_to_text import (
    SpeechToTextTask,
    SpeechToTextTaskConfig,
)

from fairseq_modules.data.data_augmentation_dataset import DataAugmentationDataset


@dataclass
class SpeechToTextModTaskConfig(SpeechToTextTaskConfig):
    sample_ratios: str = field(
        default="1",
        metadata={"help": "sample ratios of the train subsets"}
    )

    da_time_drop: int = field(
        default=1600,
        metadata={"help": "The length of the the audio segments that are masked during data augmentation. \
            Measured in frames."}
    )

    da_segm_len: int = field(
        default=48000,
        metadata={"help": "The length of the segment to which the time drop effect is applied. \
            Measured in frames."}
    )

    da_p_augm: float = field(
        default="0.8",
        metadata={"help": "The probability that data augmentation is applied to an example."}
    )

    da_tempo: str = field(
        default="0.85,1.3",
        metadata={"help": "The range from which to sample the tempo factor during data augmentation"}
    )

    da_pitch: str = field(
        default="-300,300",
        metadata={"help": "The range from which to sample the pitch value during data augmentation. \
            Measured in cents (i.e. 100ths of a semitone)"}
    )

    da_echo_delay: str = field(
        default="20,200",
        metadata={"help": "The range from which to sample the echo delay value during data augmentation. \
            Measured in milliseconds"}
    )

    da_echo_decay: str = field(
        default="0.05,0.2",
        metadata={"help": "The range from which to sample the echo decay factor during data augmentation."}
    )


@register_task("speech_to_text_iwslt21", dataclass=SpeechToTextModTaskConfig)
class SpeechToTextModTask(SpeechToTextTask):

    def __init__(self, cfg, tgt_dict):
        super().__init__(cfg, tgt_dict)

        # effect parameters for data augmentation
        self.effects_params = {
            "tempo": list(map(float, cfg.da_tempo.split(","))),
            "pitch": list(map(int, cfg.da_pitch.split(","))),
            "echo_delay": list(map(int, cfg.da_echo_delay.split(","))),
            "echo_decay": list(map(float, cfg.da_echo_decay.split(",")))
        }

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        if split.startswith("train"):
            datasets = []
            splits = split.split(',')
            sample_ratios = \
                [int(r) for r in self.cfg.sample_ratios.split(',')]
            if sample_ratios == [1]:
                sample_ratios = sample_ratios * len(splits)
            assert len(splits) == len(sample_ratios), \
                "The nº of splits and sample_ratios must be equal."
            for s in splits:
                super().load_dataset(s, epoch, combine, **kwargs)
                datasets.append(self.datasets.pop(s))
            self.datasets[split] = ConcatDataset(datasets, sample_ratios)
            self.datasets[split] = DataAugmentationDataset(
                self.datasets[split], self.effects_params,
                self.cfg.da_p_augm, self.cfg.da_segm_len)
        else:
            super().load_dataset(split, epoch, combine, **kwargs)
