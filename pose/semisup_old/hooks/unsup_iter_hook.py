# pose/semisup/hooks/unsup_iter_hook.py
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class UnsupIterHook(Hook):
    """把唯一的 train batch 重命名为 U 批次，兼容 DualTeacherWrapper.loss() 的取数逻辑。"""
    def before_train_iter(self, runner, batch_idx: int, data_batch=None):
        # data_batch 是 U 的原始 batch；复制引用到 data_batch['U']
        if isinstance(data_batch, dict) and 'U' not in data_batch:
            data_batch['U'] = data_batch
            # 同时屏蔽监督键，避免学生 loss 被误触发
            data_batch.pop('inputs', None)
            data_batch.pop('data_samples', None)
