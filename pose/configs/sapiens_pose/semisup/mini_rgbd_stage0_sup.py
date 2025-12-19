# ==== 公共导入（自动解析仓库根路径） ====
import os
import sys
from pathlib import Path

_CFG_FILE = Path(__file__).resolve()
_DEFAULT_ROOT = _CFG_FILE.parents[4]
SAPIENS_ROOT = Path(os.environ.get('SAPIENS_ROOT', str(_DEFAULT_ROOT))).resolve()

sys.path.insert(0, str(SAPIENS_ROOT / 'pretrain'))
sys.path.insert(0, str(SAPIENS_ROOT))

default_scope = 'mmpose'
custom_imports = dict(
    imports=[
        'mmpose.datasets',
        'mmpose.datasets.transforms',
        'mmpose.models',
        'mmpose.engine',
        'mmpretrain.models.backbones.vision_transformer',
        'pose.semisup',
    ],
    allow_failed_imports=False,
)

# ==== 数据根与输入尺寸（请按实际路径修改） ====
DATA_ROOT = '/path/to/mini_rgbd_coco'
image_size = (640, 480)  # (W, H)，MINI-RGBD 默认 640x480

# ==== 模型（学生原生，有监督热身阶段） ====
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch='sapiens_0.3b',
        img_size=(image_size[1], image_size[0]),
        patch_size=16,
        qkv_bias=True,
        final_norm=True,
        drop_path_rate=0.0,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/abs/path/to/sapiens_0.3b_checkpoint.pth'),  # ← 请替换
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=1024,
        out_channels=25,  # MINI-RGBD 提供 25 个 SMIL 关键点
        deconv_out_channels=(512, 512),
        deconv_kernel_sizes=(4, 4),
        conv_out_channels=(512, 512),
        conv_kernel_sizes=(1, 1),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=dict(type='MSRAHeatmap',
                     input_size=image_size,
                     heatmap_size=(image_size[0] // 4, image_size[1] // 4),
                     sigma=2)
    ),
    test_cfg=dict(flip_test=True, flip_mode='heatmap', shift_heatmap=False)
)

# ==== 训练 / 验证数据 ====
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=DATA_ROOT,
        data_mode='topdown',
        data_prefix=dict(img='.'),
        ann_file='annotations/mini_rgbd_train_keypoints.json',
        pipeline=[
            dict(type='LoadImage'),
            dict(type='TopdownGetBboxCenterScale', padding=1.25),
            dict(type='TopdownRandomFlip', flip_prob=0.5),
            dict(type='TopdownAffine', input_size=image_size),
            dict(type='PackPoseInputs',
                 meta_keys=('img_id','img_path','bbox_center','bbox_scale','ori_shape'))
        ]
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=DATA_ROOT,
        data_mode='topdown',
        data_prefix=dict(img='.'),
        ann_file='annotations/mini_rgbd_val_keypoints.json',
        pipeline=[
            dict(type='LoadImage'),
            dict(type='TopdownGetBboxCenterScale', padding=1.25),
            dict(type='TopdownAffine', input_size=image_size),
            dict(type='PackPoseInputs',
                 meta_keys=('img_id','img_path','bbox_center','bbox_scale','ori_shape'))
        ]
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=f'{DATA_ROOT}/annotations/mini_rgbd_val_keypoints.json',
    metric='keypoint',
    format_only=False,
    use_area=True,
)

test_dataloader = val_dataloader
test_evaluator = val_evaluator

# ==== 优化器 / 训练循环 ====
optim_wrapper = dict(optimizer=dict(type='AdamW', lr=5e-5, weight_decay=0.05))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)
val_cfg = dict()
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=2),
    dict(type='MultiStepLR', milestones=[15, 18], gamma=0.1, by_epoch=True),
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50),
)

# ==== 可选：从上一次训练继续 ====
load_from = None
