"""Stage 1: Synthetic warmup with LoRA on SyRIP + MINI-RGBD.

Supervised training on synthetic data to adapt Sapiens to infant anatomy.
LoRA rank=8 applied to all attention layers (QKV + proj), ~0.2% trainable.

Usage:
    cd $SAPIENS_ROOT/pose
    PYTHONPATH=$SAPIENS_ROOT:$SAPIENS_ROOT/pretrain:$SAPIENS_ROOT/pose:$PYTHONPATH \
        python tools/train.py projects/hyper_sapiens/configs/stage1_lora_syrip_mini.py \
        --work-dir work_dirs/hyper_stage1_lora
"""

_base_ = ['../../../configs/_base_/default_runtime.py']

# ---------- Architecture ----------
model_name = 'sapiens_1b'
embed_dim = 1536
num_layers = 40
num_keypoints = 17

# ---------- Paths (105 server) ----------
pretrained_checkpoint = '/data/xxt/sapiens_lite_host/pretrain/checkpoints/sapiens_1b/sapiens_1b_epoch_173_clean.pth'
data_root_syrip = '/data/xxt/sapiens_data/syrip_coco17'
data_root_mini = '/data/xxt/sapiens_data/mini_rgbd_coco17'

# ---------- Training ----------
image_size = [768, 1024]  # width x height
sigma = 6
scale = 4
patch_size = 16
num_epochs = 30
evaluate_every_n_epochs = 5
vis_every_iters = 100

train_cfg = dict(max_epochs=num_epochs, val_interval=evaluate_every_n_epochs)

# ---------- Custom imports ----------
custom_imports = dict(
    imports=[
        'projects.hyper_sapiens.engine.lora_layer_decay_constructor',
    ],
    allow_failed_imports=False,
)

# ---------- Optimizer (LoRA-aware layer decay) ----------
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.1),
    paramwise_cfg=dict(
        num_layers=num_layers,
        layer_decay_rate=0.85,
        lora_lr_scale=10.0,  # LoRA params get 10x the layer-decayed LR
    ),
    constructor='LoRALayerDecayOptimWrapperConstructor',
    clip_grad=dict(max_norm=1., norm_type=2),
)

# ---------- LR scheduler ----------
param_scheduler = [
    dict(type='LinearLR', begin=0, end=500, start_factor=0.001,
         by_epoch=False),
    dict(type='MultiStepLR', begin=0, end=num_epochs,
         milestones=[20, 27], gamma=0.1, by_epoch=True),
]

auto_scale_lr = dict(base_batch_size=512)

# ---------- Hooks ----------
default_hooks = dict(
    checkpoint=dict(save_best='coco/AP', rule='greater', max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=10),
)

# ---------- Codec ----------
codec = dict(
    type='UDPHeatmap',
    input_size=(image_size[0], image_size[1]),
    heatmap_size=(image_size[0] // scale, image_size[1] // scale),
    sigma=sigma,
)

# ---------- Model: LoRA backbone + HeatmapHead ----------
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type='mmpretrain.LoRAModel',
        module=dict(
            type='mmpretrain.VisionTransformer',
            arch=model_name,
            img_size=(image_size[1], image_size[0]),
            patch_size=patch_size,
            qkv_bias=True,
            final_norm=True,
            drop_path_rate=0.0,
            with_cls_token=False,
            out_type='featmap',
            patch_cfg=dict(padding=2),
            with_cp=True,
            init_cfg=dict(type='Pretrained', checkpoint=pretrained_checkpoint),
        ),
        alpha=16,
        rank=8,
        drop_rate=0.05,
        targets=[
            dict(type='.*attn\\.qkv'),
            dict(type='.*attn\\.proj'),
        ],
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=embed_dim,
        out_channels=num_keypoints,
        deconv_out_channels=(768, 768),
        deconv_kernel_sizes=(4, 4),
        conv_out_channels=(768, 768),
        conv_kernel_sizes=(1, 1),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec,
    ),
    test_cfg=dict(flip_test=True, flip_mode='heatmap', shift_heatmap=False),
)

# ---------- Pipelines ----------
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='PhotometricDistortion'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(type='CoarseDropout',
                 max_holes=1, max_height=0.4, max_width=0.4,
                 min_holes=1, min_height=0.2, min_width=0.2, p=1.0),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='PackPoseInputs'),
]

# ---------- Datasets: SyRIP + MINI-RGBD combined ----------
dataset_syrip = dict(
    type='CocoDataset',
    data_root=data_root_syrip,
    data_mode='topdown',
    ann_file='annotations/person_keypoints_syrip_train2017.json',
    data_prefix=dict(img='syrip_train2017/'),
)

dataset_mini_rgbd = dict(
    type='CocoDataset',
    data_root=data_root_mini,
    data_mode='topdown',
    ann_file='annotations/person_keypoints_mini_rgbd_train2017.json',
    data_prefix=dict(img='mini_rgbd_train2017/'),
)

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/coco.py'),
        datasets=[dataset_syrip, dataset_mini_rgbd],
        pipeline=train_pipeline,
    ),
)

# ---------- Validation: SyRIP val ----------
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root_syrip,
        data_mode='topdown',
        ann_file='annotations/person_keypoints_syrip_val2017.json',
        bbox_file=f'{data_root_syrip}/person_detection_results/syrip_val_detections_AP_H_70_person.json',
        data_prefix=dict(img='syrip_val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=f'{data_root_syrip}/annotations/person_keypoints_syrip_val2017.json',
)
test_evaluator = val_evaluator
