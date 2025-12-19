# ==== 公共导入（见上） ====
import sys
sys.path.insert(0, '/home/xiangxiantong/sapiens/pretrain')
sys.path.insert(0, '/home/xiangxiantong/sapiens')

default_scope = 'mmpose'
custom_imports = dict(
    imports=[
        'mmpose.datasets', 'mmpose.datasets.transforms',
        'mmpose.models', 'mmpose.engine',
        'mmpretrain.models.backbones.vision_transformer',
        'pose.semisup',
    ],
    allow_failed_imports=False,
)

# ==== 数据根与输入尺寸 ====
DATA_ROOT = '/data-nxs/xiangxiantong/stand_data'  # 改成你的带标注小集合根目录
image_size = (1024, 768)  # (W,H)

# ==== 模型（学生原生） ====
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch='sapiens_1b',
        img_size=(image_size[1], image_size[0]),
        patch_size=16,
        qkv_bias=True,
        final_norm=True,
        drop_path_rate=0.0,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
        init_cfg=dict(type='Pretrained', checkpoint='/ABS/PATH/TO/ViT_BACKBONE.pth'),  # ← 仅初始化 backbone
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=1536,             # sapiens_1b 的 embed dim（按你仓库定义为准）
        out_channels=17,
        deconv_out_channels=(768, 768),
        deconv_kernel_sizes=(4, 4),
        conv_out_channels=(768, 768),
        conv_kernel_sizes=(1, 1),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=dict(type='MSRAHeatmap', input_size=image_size, heatmap_size=(image_size[0]//4, image_size[1]//4), sigma=2)
    ),
    test_cfg=dict(flip_test=True, flip_mode='heatmap', shift_heatmap=False)
)

# ==== 训练数据（小标注集 L） ====
train_dataloader = dict(
    batch_size=16, num_workers=4, persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset', data_root=DATA_ROOT, data_mode='topdown',
        data_prefix=dict(img='train2017/'),  # 改成你的 L 集路径
        ann_file=f'{DATA_ROOT}/annotations/person_keypoints_train2017.json',
        pipeline=[
            dict(type='LoadImage'),
            dict(type='TopdownGetBboxCenterScale', padding=1.25),
            dict(type='TopdownAffine', input_size=image_size),
            dict(type='PackPoseInputs',
                 meta_keys=('img_id','img_path','bbox_center','bbox_scale','ori_shape'))
        ]
    )
)

val_dataloader = None; val_evaluator = None; val_cfg = None  # 快速热身可不设验证
test_dataloader = None; test_evaluator = None

# ==== 优化器/训练循环 ====
optim_wrapper = dict(optimizer=dict(type='AdamW', lr=5e-5, weight_decay=0.05))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_interval=0)
param_scheduler = [dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=2)]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2),
    logger=dict(type='LoggerHook', interval=50),
)
