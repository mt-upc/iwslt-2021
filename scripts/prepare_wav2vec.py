#!/usr/bin/env

import torch
import argparse
from fairseq.checkpoint_utils import (
    load_checkpoint_to_cpu,
    torch_persistent_save,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Wav2Vec checkpoint to be prepared")
    args = parser.parse_args()

    ckpt = load_checkpoint_to_cpu(args.checkpoint)

    ckpt['args'] = None
    ckpt['cfg'] = ckpt['cfg']['model']['w2v_args']

    for key in list(ckpt['model'].keys()):
        w = ckpt['model'].pop(key)
        if key.startswith('w2v_encoder.w2v_model.'):
            new_key = key.replace('w2v_encoder.w2v_model.', '')
            ckpt['model'][new_key] = w

    # These are Wav2Vec2 parameters which will be removed in Wav2VecEncoder
    if 'small' in args.checkpoint.split('/')[-1]:
        ckpt['model']['quantizer.vars'] = torch.randn(1, 640, 128)
        ckpt['model']['quantizer.weight_proj.weight'] = torch.randn(640, 512)
        ckpt['model']['quantizer.weight_proj.bias'] = torch.randn(640)
        ckpt['model']['project_q.weight'] = torch.randn(256, 256)
        ckpt['model']['project_q.bias'] = torch.randn(256)
        ckpt['model']['final_proj.weight'] = torch.randn(256, 768)
        ckpt['model']['final_proj.bias'] = torch.randn(256)
    else:
        ckpt['model']['quantizer.vars'] = torch.randn(1, 640, 384)
        ckpt['model']['quantizer.weight_proj.weight'] = torch.randn(640, 512)
        ckpt['model']['quantizer.weight_proj.bias'] = torch.randn(640)
        ckpt['model']['project_q.weight'] = torch.randn(768, 768)
        ckpt['model']['project_q.bias'] = torch.randn(768)
        ckpt['model']['final_proj.weight'] = torch.randn(768, 1024)
        ckpt['model']['final_proj.bias'] = torch.randn(768)

    torch_persistent_save(ckpt, args.checkpoint)


if __name__ == "__main__":
    main()
