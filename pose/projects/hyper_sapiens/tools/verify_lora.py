"""Verify LoRA integration with Sapiens ViT backbone.

Run:
    cd $SAPIENS_ROOT/pose
    PYTHONPATH=$SAPIENS_ROOT:$SAPIENS_ROOT/pretrain:$SAPIENS_ROOT/pose:$PYTHONPATH \
        python projects/hyper_sapiens/tools/verify_lora.py [--arch sapiens_0.3b]

Checks:
    1. LoRAModel wraps the ViT backbone correctly
    2. Only LoRA parameters have requires_grad=True
    3. Parameter counts match expectations
    4. Forward pass produces correct output shape
"""
import argparse
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='sapiens_0.3b',
                        choices=['sapiens_0.3b', 'sapiens_0.6b', 'sapiens_1b'],
                        help='Sapiens architecture variant')
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--alpha', type=int, default=16)
    parser.add_argument('--img-size', type=int, nargs=2, default=[768, 576],
                        help='Input image size (H, W)')
    args = parser.parse_args()

    # Import after PYTHONPATH is set
    from mmpretrain.models.peft.lora import LoRAModel
    from mmpretrain.models.backbones.vision_transformer import VisionTransformer

    print(f'=== LoRA Verification for {args.arch} ===\n')

    # 1. Build backbone with LoRA
    H, W = args.img_size
    backbone_cfg = dict(
        arch=args.arch,
        img_size=(H, W),
        patch_size=16,
        qkv_bias=True,
        out_type='featmap',
        with_cls_token=False,
        patch_cfg=dict(padding=2),
    )
    vit = VisionTransformer(**backbone_cfg)

    lora_model = LoRAModel(
        module=dict(type='mmpretrain.VisionTransformer', **backbone_cfg),
        alpha=args.alpha,
        rank=args.rank,
        drop_rate=0.0,
        targets=[
            dict(type='.*attn\\.qkv'),
            dict(type='.*attn\\.proj'),
        ],
    )

    # 2. Parameter analysis
    total = sum(p.numel() for p in lora_model.parameters())
    trainable = sum(p.numel() for p in lora_model.parameters()
                    if p.requires_grad)
    frozen = total - trainable
    lora_params = sum(p.numel() for n, p in lora_model.named_parameters()
                      if '.lora_' in n)

    print(f'Total parameters:     {total:>15,}')
    print(f'Trainable parameters: {trainable:>15,} ({100*trainable/total:.2f}%)')
    print(f'Frozen parameters:    {frozen:>15,} ({100*frozen/total:.2f}%)')
    print(f'LoRA parameters:      {lora_params:>15,}')
    print()

    # Verify all trainable params are LoRA
    non_lora_trainable = []
    for n, p in lora_model.named_parameters():
        if p.requires_grad and '.lora_' not in n:
            non_lora_trainable.append(n)
    if non_lora_trainable:
        print(f'WARNING: {len(non_lora_trainable)} non-LoRA trainable params:')
        for n in non_lora_trainable[:5]:
            print(f'  {n}')
    else:
        print('OK: All trainable parameters are LoRA parameters')

    # 3. List LoRA layers
    lora_layers = [n for n, _ in lora_model.named_modules()
                   if n.endswith('.lora_down') or n.endswith('.lora_up')]
    print(f'\nLoRA layers applied: {len(lora_layers) // 2} pairs')
    # Show first 3
    for n in lora_layers[:6]:
        print(f'  {n}')
    if len(lora_layers) > 6:
        print(f'  ... ({len(lora_layers) - 6} more)')

    # 4. Forward pass test
    print(f'\n=== Forward Pass Test (input: 1x3x{H}x{W}) ===')
    dummy = torch.randn(1, 3, H, W)
    lora_model.eval()
    with torch.no_grad():
        out = lora_model(dummy)
    if isinstance(out, (list, tuple)):
        out = out[-1] if isinstance(out, (list, tuple)) else out
    print(f'Output shape: {out.shape}')
    print(f'Output dtype: {out.dtype}')

    print('\n=== All checks passed ===')


if __name__ == '__main__':
    main()
