import re
import copy
import logging
from typing import Optional
from omegaconf import DictConfig, open_dict
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils
from fairseq.tasks import FairseqTask
from fairseq.modules import (
    LayerNorm,
    FairseqDropout,
)
from fairseq.models import register_model
from fairseq.models.transformer import TransformerDecoder
from fairseq.models.wav2vec import (
    Wav2Vec2Seq2SeqConfig,
    Wav2Vec2Seq2SeqModel,
    Wav2VecEncoder,
    Embedding,
)
from fairseq.models.speech_to_text import (
    lengths_to_padding_mask,
    Conv1dSubsampler,
)

logger = logging.getLogger(__name__)

BLOCKS2REGEX = {
    "encoder.feat_extr": r"encoder.*\.feature_extractor\..*|"
                         r"encoder.*\.post_extract_proj\..*|"
                         r"encoder.*\.pos_conv\..*",
    "encoder.self_attn": r"encoder.*\.self_attn\..*",
    "encoder.layer_norm": r"encoder.*layer_norm.*",
    "encoder.ffn": r"encoder.*\.fc[1-2]\..*",
    "adapter": r"encoder\.adapter.*",
    "len_adaptor": r"encoder\.len_adaptor.*",
    "decoder.embedding": r"decoder\.embed_tokens.*|"
                         r"decoder\.embed_positions.*|"
                         r"decoder\.layernorm_embedding.*",
    "decoder.self_attn": r"decoder.*\.self_attn\..*",
    "decoder.layer_norm": r"decoder.*layer_norm.*",
    "decoder.encoder_attn": r"decoder.*\.encoder_attn\..*",
    "decoder.ffn": r"decoder.*\.fc[1-2]\..*",
}


@dataclass
class Wav2Vec2Seq2SeqModConfig(Wav2Vec2Seq2SeqConfig):
    freeze_layers: str = field(
        default="",
        metadata={"help": "finetune only LayerNorm and Attention (LNA) layers"}
    )
    adapter_dim: Optional[int] = field(
        default=None,
        metadata={"help": "projection size of the Adapter"}
    )
    adapter_post: bool = field(
        default=False,
        metadata={"help": "if true, the Adapter is placed after the "
                          "Length Adaptor"}
    )
    adapter_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability for the encoder-decoder "
                          "attention weights (if it's not specified, the "
                          "decoder_attention_dropout is used)"}
    )
    len_adaptor_kernel_sizes: str = field(
        default="3,3",
        metadata={"help": "kernel sizes of the Length Adaptor (Conv1d)"}
    )
    len_adaptor_channels: int = field(
        default=1024,
        metadata={"help": "# of channels in the Length Adaptor (Conv1d)"}
    )
    load_pretrained_decoder_from: Optional[str] = field(
        default=None,
        metadata={"help": "model to take decoder weights from"}
    )
    decoder_output_dim: int = field(
        default=768,
        metadata={"help": "decoder output dimension (extra linear layer "
                          "if different from decoder embed dim)"}
    )
    decoder_enc_attention_dropout: Optional[float] = field(
        default=None,
        metadata={"help": "dropout probability for the encoder-decoder "
                          "attention weights (if it's not specified, the "
                          "decoder_attention_dropout is used)"}
    )


@register_model("wav2vec_seq2seq_iwslt21", dataclass=Wav2Vec2Seq2SeqModConfig)
class Wav2Vec2Seq2SeqModModel(Wav2Vec2Seq2SeqModel):
    """
    Modified version of the wav2vec_seq2seq model.

    It adds these functionalities:
      - Use with the speech_to_text pipeline
      - Loading pretrained decoder
      - Finetuning only LNA layers
      - Using adapter and length_adaptor modules
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, cfg: Wav2Vec2Seq2SeqModConfig, task: FairseqTask):
        """Build a new model instance."""

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(task.tgt_dict, cfg.decoder_embed_dim)

        encoder = cls.build_encoder(cfg)
        decoder = cls.build_decoder(cfg, task.tgt_dict, decoder_embed_tokens)

        model = Wav2Vec2Seq2SeqModModel(encoder, decoder)
        model.freeze_blocks(cfg)
        return model

    @classmethod
    def build_encoder(cls, cfg: Wav2Vec2Seq2SeqModConfig):
        return Wav2VecEncoderMod(cfg)

    @classmethod
    def build_decoder(cls, cfg: Wav2Vec2Seq2SeqModConfig, tgt_dict, embed_tokens):
        decoder = TransformerDecoderMod(cfg, tgt_dict, embed_tokens)
        if getattr(cfg, "load_pretrained_decoder_from", None):
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=cfg.load_pretrained_decoder_from
            )
            logger.info(
                f"loaded pretrained decoder from: "
                f"{cfg.load_pretrained_decoder_from}"
            )
        return decoder

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, **kwargs
        )
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return decoder_out

    def freeze_blocks(self, cfg: Wav2Vec2Seq2SeqModConfig):
        regex_to_freeze = re.compile(
            "|".join([BLOCKS2REGEX[b] for b in cfg.freeze_layers.split(',')])
        )
        for n, p in self.named_parameters():
            if re.match(regex_to_freeze, n):
                p.requires_grad = False

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def load_state_dict(self, state_dict, strict=True, model_cfg=None, args=None):
        # Allow loading state dicts without Adapter
        state_has_adapter = any([k for k in state_dict.keys()
                                 if k.startswith("encoder.adapter.")])
        if self.encoder.adapter and not state_has_adapter:
            state_dict["encoder.adapter.layer_norm.weight"] = \
                self.encoder.adapter.layer_norm.weight
            state_dict["encoder.adapter.layer_norm.bias"] = \
                self.encoder.adapter.layer_norm.bias
            state_dict["encoder.adapter.down_proj.weight"] = \
                self.encoder.adapter.down_proj.weight
            state_dict["encoder.adapter.down_proj.bias"] = \
                self.encoder.adapter.down_proj.bias
            state_dict["encoder.adapter.up_proj.weight"] = \
                self.encoder.adapter.up_proj.weight
            state_dict["encoder.adapter.up_proj.bias"] = \
                self.encoder.adapter.up_proj.bias

        super().load_state_dict(state_dict, strict, model_cfg, args)


class Wav2VecEncoderMod(Wav2VecEncoder):
    """
    Modification of the Wav2VecEncoder

    It modifies it to work with the speech_to_text pipeline.
    Moreover, it includes the adapter and length adaptor modules.
    """
    def __init__(self, cfg: Wav2Vec2Seq2SeqModConfig, tgt_dict=None):
        super().__init__(cfg, tgt_dict)
        self.adapter = Adapter(
            cfg.decoder_embed_dim,
            cfg.adapter_dim,
            cfg.adapter_dropout
        ) if cfg.adapter_dim else None

        self.len_adaptor = Conv1dSubsampler(
            cfg.decoder_embed_dim,
            cfg.len_adaptor_channels,
            cfg.decoder_embed_dim,
            [int(k) for k in cfg.len_adaptor_kernel_sizes.split(",")],
        )
        self.adapter_post = cfg.adapter_post

    def forward(self, src_tokens, src_lengths, **kwargs):
        encoder_out = super().forward(
            source=src_tokens,
            padding_mask=lengths_to_padding_mask(src_lengths),
            tbc=False,
            **kwargs
        )
        encoder_out["encoder_padding_mask"] = encoder_out.pop("padding_mask")
        if self.adapter and not self.adapter_post:
            encoder_out["encoder_out"] = \
                self.adapter(
                    encoder_out["encoder_out"]
                )
        encoder_out["encoder_out"], lengths = self.len_adaptor(
            encoder_out["encoder_out"],
            (~encoder_out["encoder_padding_mask"]).sum(dim=1)
        )
        if self.adapter and self.adapter_post:
            encoder_out["encoder_out"] = \
                self.adapter(
                    encoder_out["encoder_out"]
                )
        encoder_out["encoder_padding_mask"] = lengths_to_padding_mask(lengths)
        return {k: [v] for k, v in encoder_out.items()}

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out, new_order):
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
        }


class TransformerDecoderMod(TransformerDecoder):
    """
    Modification of the TransformerDecoder

    It is adapted to the argument names defined in Wav2Vec2Seq2SeqModConfig.
    """
    def __init__(self, cfg, dictionary, embed_tokens, no_encoder_attn=False):
        transformer_cfg = copy.deepcopy(cfg)
        with open_dict(transformer_cfg):
            transformer_cfg.dropout = transformer_cfg.decoder_dropout
            transformer_cfg.attention_dropout = (
                transformer_cfg.decoder_attention_dropout
            )
            transformer_cfg.activation_dropout = (
                transformer_cfg.decoder_activation_dropout
            )
            transformer_cfg.layernorm_embedding = True
            transformer_cfg.adaptive_input = False
            transformer_cfg.no_scale_embedding = False
            transformer_cfg.quant_noise_pq = 0.0
            transformer_cfg.adaptive_softmax_cutoff = None
        super().__init__(transformer_cfg, dictionary, embed_tokens, no_encoder_attn)
        if cfg.decoder_enc_attention_dropout is not None:
            for layer in self.layers:
                layer.encoder_attn.dropout_module.p = \
                    cfg.decoder_enc_attention_dropout

    def load_state_dict(self, state_dict, strict=True):
        state_dict["output_projection.weight"] = state_dict["embed_tokens.weight"]
        super().load_state_dict(state_dict, strict)


class Adapter(nn.Module):
    """
    Adapter for model finetuning, as described in:
    https://arxiv.org/pdf/1909.08478.pdf
    """
    def __init__(self, embed_dim, proj_dim, dropout):
        super().__init__()
        self.layer_norm = LayerNorm(embed_dim)
        self.down_proj = nn.Linear(embed_dim, proj_dim)
        self.up_proj = nn.Linear(proj_dim, embed_dim)
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.down_proj(x)
        x = F.relu(x)
        x = self.up_proj(x)
        x = self.dropout_module(x)
        x += residual
        return x
