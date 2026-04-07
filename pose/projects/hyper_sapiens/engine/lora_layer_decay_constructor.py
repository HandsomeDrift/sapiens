"""Layer-decay optimizer constructor with LoRA parameter support.

When the backbone is wrapped by LoRAModel, parameter names change from
``backbone.layers.{N}.attn.qkv.weight`` to
``backbone.module.layers.{N}.attn.qkv.original_layer.weight`` (frozen) and
``backbone.module.layers.{N}.attn.qkv.lora_down.weight`` (trainable).

This constructor:
1. Strips the ``module.`` prefix to correctly resolve the ViT layer index.
2. Assigns LoRA parameters to their corresponding layer group with an
   optional learning rate multiplier (``lora_lr_scale``).
"""

import re

from mmengine.dist.utils import get_dist_info
from mmengine.optim import DefaultOptimWrapperConstructor
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS


def get_num_layer_for_vit_lora(var_name, num_max_layer):
    """Resolve ViT layer index from parameter name, handling LoRA wrapping.

    Handles both normal paths (``backbone.layers.3.attn.qkv.weight``)
    and LoRA-wrapped paths (``backbone.module.layers.3.attn.qkv.lora_down.weight``).
    """
    # Strip 'backbone.' prefix if present
    name = var_name
    if name.startswith('backbone.'):
        name = name[len('backbone.'):]

    # Strip 'module.' prefix added by LoRAModel wrapping
    if name.startswith('module.'):
        name = name[len('module.'):]

    if name in ('cls_token', 'mask_token', 'pos_embed'):
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('layers'):
        # layers.{N}.xxx
        layer_id = int(name.split('.')[1])
        return layer_id + 1
    else:
        # head parameters go to the last group
        return num_max_layer - 1


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class LoRALayerDecayOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    """Optimizer constructor supporting LoRA parameters with layer decay.

    Compared to ``LayerDecayOptimWrapperConstructor``, this constructor:
    - Correctly parses layer indices when backbone is wrapped by LoRAModel
    - Applies a separate ``lora_lr_scale`` multiplier to LoRA parameters
    - Treats LoRA parameters as 'decay' group (they are 2D weight matrices)

    paramwise_cfg keys:
        num_layers (int): Number of transformer layers in the ViT.
        layer_decay_rate (float): Layer-wise LR decay factor.
        lora_lr_scale (float): Extra LR multiplier for LoRA parameters
            on top of the layer-decayed LR. Default: 1.0.
    """

    def __init__(self, optim_wrapper_cfg, paramwise_cfg=None):
        super().__init__(optim_wrapper_cfg, paramwise_cfg=paramwise_cfg)
        self.layer_decay_rate = paramwise_cfg.get('layer_decay_rate', 0.5)

    def add_params(self, params, module, prefix='', lr=None):
        parameter_groups = {}
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        layer_decay_rate = self.paramwise_cfg.get('layer_decay_rate')
        lora_lr_scale = self.paramwise_cfg.get('lora_lr_scale', 1.0)
        weight_decay = self.base_wd

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue

            is_lora = '.lora_' in name

            # Weight decay: no decay for 1-D params (bias, norm), pos_embed
            if (len(param.shape) == 1 or name.endswith('.bias')
                    or 'pos_embed' in name):
                group_name = 'no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay

            # Resolve layer index
            layer_id = get_num_layer_for_vit_lora(name, num_layers)

            if is_lora:
                group_name = f'layer_{layer_id}_lora_{group_name}'
            else:
                group_name = f'layer_{layer_id}_{group_name}'

            if group_name not in parameter_groups:
                scale = layer_decay_rate ** (num_layers - layer_id - 1)
                if is_lora:
                    scale *= lora_lr_scale

                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'param_names': [],
                    'lr_scale': scale,
                    'group_name': group_name,
                    'lr': scale * self.base_lr,
                }

            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)

        rank, _ = get_dist_info()
        if rank == 0:
            # Summary logging
            total_params = sum(
                p.numel() for p in module.parameters())
            trainable_params = sum(
                p.numel() for p in module.parameters() if p.requires_grad)
            lora_params = sum(
                p.numel() for n, p in module.named_parameters()
                if '.lora_' in n)
            print(f'[LoRALayerDecay] Total params: {total_params:,}')
            print(f'[LoRALayerDecay] Trainable params: {trainable_params:,} '
                  f'({100 * trainable_params / total_params:.2f}%)')
            print(f'[LoRALayerDecay] LoRA params: {lora_params:,}')
            for key in sorted(parameter_groups.keys()):
                grp = parameter_groups[key]
                n_params = sum(p.numel() for p in grp['params'])
                print(f'  {key}: {len(grp["params"])} tensors, '
                      f'{n_params:,} params, '
                      f'lr_scale={grp["lr_scale"]:.4f}, '
                      f'lr={grp["lr"]:.2e}')

        params.extend(parameter_groups.values())
