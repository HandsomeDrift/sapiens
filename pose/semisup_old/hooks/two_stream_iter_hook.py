
from typing import Iterator
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class TwoStreamIterHook(Hook):
    """让 Runner 在每步同时从有标注(L)与无标注(U)两路取 batch，拼成一个 data_batch。
    要求 runner.train_loop 仍只接收一个 dataloader；本 Hook 在 before_train_iter 内部
    从 runner._train_dataloaders[1] 取 U 批次，并注入到 data_batch['U']。"""
    def before_train_iter(self, runner, batch_idx: int, data_batch=None):
        if not hasattr(runner, '_train_dataloaders'):
            return
        loaders = runner._train_dataloaders  # mmengine >=0.9 支持 list dataloaders
        if not isinstance(loaders, (list, tuple)) or len(loaders) < 2:
            return
        # data_batch 是 L 批次；我们再从第二个 loader 取 U 批次
        if not hasattr(self, '_unsup_iter'):
            self._unsup_iter = iter(loaders[1])
        try:
            u_batch = next(self._unsup_iter)
        except StopIteration:
            self._unsup_iter = iter(loaders[1])
            u_batch = next(self._unsup_iter)
        data_batch['U'] = u_batch
