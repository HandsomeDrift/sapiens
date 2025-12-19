
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class DualEMAHook(Hook):
    """维护两位教师(geom/app)对学生的 EMA，同步在每个 iteration 后更新。"""
    def __init__(self, momentum: float = 0.999):
        self.momentum = momentum

    def before_train(self, runner):
        model = runner.model.module if hasattr(runner.model,'module') else runner.model
        if hasattr(model, 'geom_teacher'):
            model.geom_teacher.load_state_dict(model.student.state_dict(), strict=False)
            model.app_teacher.load_state_dict(model.student.state_dict(), strict=False)


    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        if not hasattr(model, 'geom_teacher'):
            return
        for t in [model.geom_teacher, model.app_teacher]:
            if t is None: continue
            m = self.momentum
            for (name_s, p_s), (name_t, p_t) in zip(model.student.state_dict().items(), t.state_dict().items()):
                if p_t.shape != p_s.shape:
                    continue
                p_t.copy_(p_t * m + p_s.to(dtype=p_t.dtype) * (1.0 - m))
