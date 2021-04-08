from omegaconf import II
from dataclasses import dataclass, field

from fairseq.data import ConcatDataset
from fairseq.tasks import register_task
from fairseq.tasks.speech_to_text import (
    SpeechToTextTask,
    SpeechToTextTaskConfig,
)


@dataclass
class SpeechToTextModTaskConfig(SpeechToTextTaskConfig):
    sample_ratios: str = field(
        default="1",
        metadata={"help": "sample ratios of the train subsets"}
    )


@register_task("speech_to_text_iwslt21", dataclass=SpeechToTextModTaskConfig)
class SpeechToTextModTask(SpeechToTextTask):

    def __init__(self, cfg, tgt_dict):
        super().__init__(cfg, tgt_dict)

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
        else:
            super().load_dataset(split, epoch, combine, **kwargs)
