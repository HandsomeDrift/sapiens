# ==== 公共导入：自动推断仓库根路径 ====
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

# ==== 数据路径（请按实际输出目录修改） ====
DATA_ROOT = '/path/to/mini_rgbd_coco'
DET_JSON = f'{DATA_ROOT}/detections/mini_rgbd_unsup_person_dets.json'
ID_MAP = f'{DATA_ROOT}/image_id_maps/mini_rgbd_unsup_image_ids.json'
IMG_FOLDER = '.'  # file_name 已包含 seq/xxx.png，相对 DATA_ROOT

image_size = (640, 480)  # (W, H)
model_name = 'sapiens_0.3b'
embed_dim = 1024

# ==== 模型（SAGE-Pose 双教师封装） ====
model = dict(
    type='DualTeacherWrapper',
    num_keypoints=25,
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
            init_cfg=dict(
                type='Pretrained',
                checkpoint='/abs/path/to/sapiens_0.3b_checkpoint.pth'),  # ← 请替换
        ),
        head=dict(
            type='HeatmapHead',
            in_channels=embed_dim,
            out_channels=25,  # 25 个 SMIL 关键点
            deconv_out_channels=(512, 512),
            deconv_kernel_sizes=(4, 4),
            conv_out_channels=(512, 512),
            conv_kernel_sizes=(1, 1),
            loss=dict(type='KeypointMSELoss', use_target_weight=True),
            decoder=dict(
                type='MSRAHeatmap',
                input_size=image_size,
                heatmap_size=(image_size[0] // 4, image_size[1] // 4),
                sigma=2),
        ),
        test_cfg=dict(flip_test=True, flip_mode='heatmap', shift_heatmap=False)
    ),
    semi_cfg=dict(
        unsup_only=True,
        teacher_param_dtype='fp32',
        student_param_dtype='fp32',
        ema_momentum=0.999,
        lambda_u=5.0,
        lambda_topo=0.05,
        lambda_bone=0.05,
        lambda_angle=0.02,
        temperature=1.0,
        beta=1.0,
        consistency_warmup_iters=0,
        min_keep_ratio=0.3,
        percentile=None,
        num_strong_views=1,
        weak_max_deg=10,
        strong_max_deg=15,
        accum_steps=4,
        stats_warmup_updates=1000,
        consistency_warmup_updates=800,
        lambda_u_ramp_updates=5000,
        spatial_tau=0.1,
        debug_log_interval=20,
        edges=[(0,1),(1,4),(4,7),(7,10),
               (0,2),(2,5),(5,8),(8,11),
               (0,3),(3,6),(6,9),(9,12),
               (12,13),(13,16),(16,18),(18,20),(20,22),
               (12,14),(14,17),(17,19),(19,21),(21,23),
               (12,15),(15,24)],
        angle_triplets=[(0,1,4),(1,4,7),(4,7,10),
                        (0,2,5),(2,5,8),(5,8,11),
                        (0,3,6),(3,6,9),(6,9,12),(9,12,15),
                        (13,12,14),(12,13,16),(13,16,18),(16,18,20),(18,20,22),
                        (12,14,17),(14,17,19),(17,19,21),(19,21,23)],
        ref_bone_lengths=[1.0]*22,
        instance_thresh=0.70,
    )
)

# ==== 无标签数据（用于半监督 Stage1） ====
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=False,
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
            dict(type='GetBBoxCenterScale', padding=1.25),
            dict(type='EnsureBatchedCenterScale'),
            dict(type='TopdownAffine', input_size=image_size),
            dict(type='PackPoseInputs',
                 meta_keys=('img_id','img_path','ori_shape','bbox_center','bbox_scale'))
        ]
    )
)
val_dataloader = None
val_evaluator = None
val_cfg = None
test_dataloader = None
test_evaluator = None

# ==== 训练配置 ====
optim_wrapper = dict(
    type='AmpOptimWrapper', dtype='bfloat16',
    optimizer=dict(type='AdamW', lr=5e-5, weight_decay=0.05),
    accumulative_counts=4,
)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=0)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.5, by_epoch=True, begin=0, end=3)
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50),
)

load_from = None  # 若已完成 Stage0，可在此填写监督热身的 ckpt
