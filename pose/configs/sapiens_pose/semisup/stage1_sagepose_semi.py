# ==== 公共导入 ====
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
        'mmpose.datasets', 'mmpose.datasets.transforms',
        'mmpose.models', 'mmpose.engine',
        'mmpretrain.models.backbones.vision_transformer',
        'pose.semisup', 'pose.semisup.data.pipelines.fix_center_scale',
    ],
    allow_failed_imports=False,
)

# ==== 数据根与路径（按你的实际路径） ====
DATA_ROOT = '/data-nxs/xiangxiantong/stand_data'
DET_JSON  = f'{DATA_ROOT}/person_detection_results/stand_detections_person.json'
ID_MAP    = f'{DATA_ROOT}/person_detection_results/stand_detections_person_image_id_map.json'
IMG_FOLDER= 'stand_6fps_together'
image_size = (1024, 768)  # (W,H)
# image_size=(896, 672)
model_name = 'sapiens_0.3b'; embed_dim=1024; num_layers=24
# model_name = 'sapiens_0.6b'; embed_dim=1280; num_layers=32
# model_name = 'sapiens_1b'; embed_dim=1536; num_layers=40

# ==== 模型（SAGE-Pose 封装器） ====
model = dict(
    type='DualTeacherWrapper',
    num_keypoints=17,
    student=dict(
        type='TopdownPoseEstimator',
        data_preprocessor=dict(
            type='PoseDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True),
        backbone=dict(
            type='mmpretrain.VisionTransformer',
            arch=model_name,
            img_size=(image_size[1], image_size[0]),
            patch_size=16,
            qkv_bias=True,
            final_norm=True,
            drop_path_rate=0.0,
            with_cls_token=False,
            out_type='featmap',
            patch_cfg=dict(padding=2),
            # with_cp=True,
            # 只初始化 backbone；如果你要从 Stage0 的学生整体继续，可把 stage0 的 ckpt 放到 load_from（见下）
            # init_cfg=dict(type='Pretrained', checkpoint='/data/xiangxiantong/sapiens_lite_host/torchscript/pretrain/checkpoints/sapiens_0.3b/sapiens_0.3b_epoch_1600_clean.pth'),
            # init_cfg=dict(type='Pretrained', checkpoint='/data/xiangxiantong/sapiens_lite_host/torchscript/pretrain/checkpoints/sapiens_0.6b/sapiens_0.6b_epoch_1600_clean.pth'),
            # init_cfg=dict(type='Pretrained', checkpoint='/data/xiangxiantong/sapiens_lite_host/torchscript/pretrain/checkpoints/sapiens_1b/sapiens_1b_epoch_173_clean.pth'),
            init_cfg=dict(type='Pretrained', checkpoint=f'/data/xiangxiantong/sapiens_lite_host/torchscript/pose/checkpoints/sapiens_0.3b/sapiens_0.3b_coco_best_coco_AP_796.pth', prefix='backbone.'),
        ),
        head=dict(
            type='HeatmapHead',
            in_channels=embed_dim,
            out_channels=17,
            deconv_out_channels=(768, 768),
            # deconv_out_channels=(512, 512),
            deconv_kernel_sizes=(4, 4),
            conv_out_channels=(768, 768),
            # conv_out_channels=(512, 512),
            conv_kernel_sizes=(1, 1),
            loss=dict(type='KeypointMSELoss', use_target_weight=True),
            decoder=dict(type='MSRAHeatmap', input_size=image_size, heatmap_size=(image_size[0]//4, image_size[1]//4), sigma=2),
            init_cfg=dict(type='Pretrained', checkpoint=f'/data/xiangxiantong/sapiens_lite_host/torchscript/pose/checkpoints/sapiens_0.3b/sapiens_0.3b_coco_best_coco_AP_796.pth', prefix='head.'),
        ),
        test_cfg=dict(flip_test=True, flip_mode='heatmap', shift_heatmap=False)
    ),
    semi_cfg=dict(
        unsup_only=True,                # ← 跳过监督，纯 U；若想混合监督，这里可设 False 并提供 L 的 dataloader
        teacher_param_dtype='fp32',     # 教师半精度
        student_param_dtype='fp32',     # 学生用 BF16 更稳、更省显存（也可 fp32）
        ema_momentum=0.999,
        lambda_u=5.0,
        lambda_topo=0.05, lambda_bone=0.05, lambda_angle=0.02,
        temperature=1.0, beta=1.0,
        consistency_warmup_iters=0,
        min_keep_ratio=0.3,
        percentile=None,
        num_strong_views=1,
        weak_max_deg=10, strong_max_deg=15,
        accum_steps=4,
        stats_warmup_updates=1000,
        consistency_warmup_updates=800,
        lambda_u_ramp_updates=5000,
        spatial_tau=0.1,
        debug_log_interval=20,
        # 17点 COCO 的骨架边与角（示例；你可按需求调整）
        edges=[(5,7),(7,9),(6,8),(8,10),(5,6),(11,12),(5,11),(6,12),(11,13),(13,15),(12,14),(14,16)],
        angle_triplets=[(5,7,9),(6,8,10),(11,13,15),(12,14,16)],
        ref_bone_lengths=[1.0]*12,
        instance_thresh=0.70
    )
)

# ==== 训练数据（无标签 U） ====
train_dataloader = dict(
    batch_size=4, num_workers=4, persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='UnlabeledCocoTopDownDataset',
        data_root=DATA_ROOT,
        img_folder=IMG_FOLDER,
        det_json=DET_JSON,
        image_id_map_json=ID_MAP,
        bbox_score_thr=0.0,
        pipeline=[
            dict(type='LoadImage'),
            # 你的 mmpose 版本若没有该名，请改成 'GetBBoxCenterScale'
            dict(type='GetBBoxCenterScale', padding=1.25),
            dict(type='EnsureBatchedCenterScale'),
            dict(type='TopdownAffine', input_size=image_size),
            dict(type='PackPoseInputs',
                 meta_keys=('img_id','img_path','ori_shape','bbox_center','bbox_scale'))
        ]
    )
)
val_dataloader = None; val_evaluator = None; val_cfg = None
test_dataloader = None; test_evaluator = None

# ==== 优化器 / AMP / 训练循环 ====
optim_wrapper = dict(
    type='AmpOptimWrapper', dtype='bfloat16',  # 与 student_param_dtype 对齐
    optimizer=dict(type='AdamW', lr=5e-5, weight_decay=0.05),
    accumulative_counts=4,   # 例如 4×(batch_size=8)≈32 的有效 batch
)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=0)
param_scheduler = [dict(type='LinearLR', start_factor=0.5, by_epoch=True, begin=0, end=3)]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50),
)

# ==== 从 Stage 0 继续（可选） ====
# 如果你希望在 Stage0 基础上继续（优先使用），把下面这一行填成 stage0 的学生 ckpt：
# load_from = "/data/xiangxiantong/sapiens_lite_host/torchscript/pose/checkpoints/sapiens_0.3b/sapiens_0.3b_coco_best_coco_AP_796.pth"  # '/ABS/PATH/TO/stage0_student.pth'
load_from = None
