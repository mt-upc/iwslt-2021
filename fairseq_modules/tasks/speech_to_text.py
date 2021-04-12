from dataclasses import dataclass, field

from fairseq.data import ConcatDataset, SubsampleDataset
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

        assert len(set(self.da_effects_info["echo"]["delay"])) == \
            len(set(self.da_effects_info["echo"]["decay"])), \
            "Specify ranges for both parameters of echo (delay & decay) or for none"

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        if split.startswith("train"):
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
            self.datasets[split] = AugmentationNormalizationDataset(
                self.datasets[split], self.da_effects_info,
                self.cfg.da_p_augm, self.cfg.normalize)
        else:
            super().load_dataset(split, epoch, combine, **kwargs)

    def begin_epoch(self, epoch, model):
        super().begin_epoch(epoch, model)
        if epoch == 1:
            return
        for split in self.datasets.keys():
            if split.startswith("train"):
                # Perform a new subsampling at each epoch
                self.load_dataset(split, epoch)
