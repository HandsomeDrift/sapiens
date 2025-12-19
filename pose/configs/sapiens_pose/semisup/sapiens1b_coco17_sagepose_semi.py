# 让 OpenMMLab 的作用域、生效的注册树可用
default_scope = 'mmpose'

# 在构图之前，显式导入需要的模块（触发注册/补丁）
custom_imports = dict(
  imports=[
    'mmpose.datasets', 'mmpose.datasets.transforms',
    'mmpose.models', 'mmpose.engine',
    'mmpretrain.models.backbones.vision_transformer',
    'pose.semisup', 'pose.semisup.data.pipelines.fix_center_scale',  # ← 新增
  ],
  allow_failed_imports=False,
)




# 说明：该配置基于官方 Sapiens-1B 的 COCO-17 训练/评测模板改造。
# 你需要先把原始 1B 学生模型 config 粘贴为 student_cfg，再在此基础上包装 DualTeacherWrapper。

default_scope = 'mmpose'

model_name = 'sapiens_1b'; embed_dim=1536; num_layers=40

pretrained_checkpoint="/data/xiangxiantong/sapiens_lite_host/torchscript/pretrain/checkpoints/sapiens_1b/sapiens_1b_epoch_173_clean.pth"

evaluate_every_n_epochs = 1

vis_every_iters=100
image_size = [1024, 768] ## width x height
sigma = 6 ## sigma is 2 for 256
scale = 4
patch_size=16
num_keypoints=17
num_epochs=210

# codec settings
codec = dict(
    type='UDPHeatmap', input_size=(image_size[0], image_size[1]), heatmap_size=(int(image_size[0]/scale), int(image_size[1]/scale)), sigma=sigma) ## sigma is 2 for 256

# ===== 学生模型（请粘贴官方 Sapiens-1B 的 backbone/neck/head） =====
# student = dict(
#     type='TopdownPoseEstimator',
#     # 省略：backbone/neck/keypoint_head 与官方一致
# )
student = dict(
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
        patch_size=patch_size,
        qkv_bias=True,
        final_norm=True,
        drop_path_rate=0.0,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained_checkpoint),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=embed_dim,
        out_channels=num_keypoints,
        deconv_out_channels=(768, 768), ## this will 2x at each step. so total is 4x
        deconv_kernel_sizes=(4, 4),
        conv_out_channels=(768, 768),
        conv_kernel_sizes=(1, 1),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=False,
    ))

model = dict(
    type='DualTeacherWrapper',
    student=student,
    num_keypoints=17,
    semi_cfg=dict(
        teacher_param_dtype='fp16',   # 可选: 'fp16' / 'bf16' / 'fp32'
        student_param_dtype='fp32',   # 可选: 想更省显存可试 'bf16'（更稳于 fp16）
        unsup_only=True,
        lambda_u=1.0, lambda_topo=0.05, lambda_bone=0.05, lambda_angle=0.02,
        temperature=1.0, beta=0.7, num_strong_views=2, instance_thresh=0.70,
        edges=[(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,13),(13,15),(12,14),(14,16)],
        angle_triplets=[(5,7,9),(6,8,10),(11,13,15),(12,14,16)],
        ref_bone_lengths=[100]*11
    )
)

# ===== Data =====
# L（有标注）复用官方 COCO-17 训练配置；U（无标注）使用 UnlabeledCocoTopDownDataset

## 纯无监督
# train_dataloader = [
#     dict(
#         batch_size=16, num_workers=4, persistent_workers=False, sampler=dict(type='DefaultSampler', shuffle=True),
#         dataset=dict(
#             type='CocoDataset',  # 与官方一致
#             data_root='${DATA_ROOT}',
#             data_mode='topdown', data_prefix=dict(img='train2017/'),
#             ann_file='${DATA_ROOT}/annotations/person_keypoints_train2017.json',
#             pipeline=[  # 贴官方训练增广
#                 dict(type='RandomFlip', prob=0.5, direction='horizontal'),
#                 dict(type='RandomHalfBody'),
#                 dict(type='PackPoseInputs')
#             ]
#         )
#     ),
#     dict(
#         batch_size=64, num_workers=4, persistent_workers=False, sampler=dict(type='DefaultSampler', shuffle=True),
#         dataset=dict(
#             type='UnlabeledCocoTopDownDataset',
#             data_root='${DATA_ROOT}', img_folder='stand_6fps_together',
#             det_json='${DATA_ROOT}/person_detection_results/stand_detections_person.json',
#             pipeline=[
#                 # 教师弱增（几何/外观）与学生强增在 collate 时生成；此处仍需基本加载与裁剪
#                 dict(type='PackPoseInputs')
#             ]
#         )
#     )
# ]

train_dataloader = dict(
    batch_size=64, num_workers=4, persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='UnlabeledCocoTopDownDataset',
        data_root='/data-nxs/xiangxiantong/stand_data',
        img_folder='stand_6fps_together',
        # 须提供训练集的人检 JSON（与 val 同 schema）：
        det_json='/data-nxs/xiangxiantong/stand_data/person_detection_results/stand_detections_person.json',
        image_id_map_json='/data-nxs/xiangxiantong/stand_data/person_detection_results/stand_detections_person_image_id_map.json',  # ← 关键
        bbox_score_thr=0.0,  # 需要的话可调，比如 0.3
        pipeline=[
            dict(type='LoadImage'),                                   # 读图 -> results['img']
            dict(type='GetBBoxCenterScale', padding=1.25),     # 由 bbox 得 center/scale
            dict(type='EnsureBatchedCenterScale'),  
            dict(type='TopdownAffine', input_size=image_size),        # 仿射裁剪到模型输入
            dict(type='PackPoseInputs',
                 meta_keys=('img_id','img_path','ori_shape','bbox_center','bbox_scale'))
        ]
    )
)

## 暂时关闭验证
# val_dataloader = dict(
#     batch_size=1, num_workers=2, persistent_workers=False,
#     sampler=dict(type='DefaultSampler', shuffle=False, round_up=True),
#     dataset=dict(
#         type='CocoDataset', data_root='${DATA_ROOT}', data_mode='topdown',
#         data_prefix=dict(img='val2017/'),
#         ann_file='${DATA_ROOT}/annotations/person_keypoints_val2017.json',
#         pipeline=[dict(type='PackPoseInputs')]
#     )
# )

# test_dataloader = val_dataloader

# ===== Hooks =====
hooks = [
    dict(type='TwoStreamIterHook'),
    dict(type='UnsupIterHook'), 
    dict(type='DualEMAHook', momentum=0.999)
]

# ===== Runtime =====
# optim_wrapper = dict(optimizer=dict(type='AdamW', lr=5e-5, weight_decay=0.05))
optim_wrapper = dict(
  type='AmpOptimWrapper',  # 取代原来的 OptimWrapper
  dtype='bfloat16',        # 若你的 GPU (A100/H100/4090等) 支持 BF16；否则用 'float16'
  optimizer=dict(type='AdamW', lr=5e-5, weight_decay=0.05)
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=60, val_interval=5)
## 暂时关闭验证
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')
param_scheduler = [dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=1000)]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50)
)
