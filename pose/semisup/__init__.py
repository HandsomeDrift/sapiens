# 确保子模块被 import，从而完成 mmengine/mmpose 注册
from .data import unlabeled_coco_topdown  # noqa: F401
from .data.pipelines import fix_center_scale  # noqa: F401
from .models import dual_teacher_wrapper  # noqa: F401
from .models.losses import structural_priors  # noqa: F401
from .utils import geometry, pseudo_label  # noqa: F401
