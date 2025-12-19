
from mmengine.registry import TRANSFORMS

__all__ = ['build_weak_geom_pipeline', 'build_weak_app_pipeline']

# 这里返回的是 mmcv/MMEngine 的 Transform 配置列表（供 config 直接引用）

def build_weak_geom_pipeline():
    return [
        dict(type='Resize', scale=(1024, 768)),
        dict(type='RandomFlip', prob=0.5, direction='horizontal'),
        dict(type='RandomRotation', degree=15),
        dict(type='PackPoseInputs')
    ]

def build_weak_app_pipeline():
    return [
        dict(type='Resize', scale=(1024, 768)),
        dict(type='RandomFlip', prob=0.5, direction='horizontal'),
        dict(type='PhotoMetricDistortion', brightness_delta=16, contrast_range=(0.8,1.2), saturation_range=(0.8,1.2), hue_delta=8),
        dict(type='PackPoseInputs')
    ]
