from mmengine.registry import TRANSFORMS

@TRANSFORMS.register_module()
class EnsureBatchedCenterScale:
    """把 {bbox_center,bbox_scale} 从 (2,) 改成 (1,2)。"""
    def __call__(self, results):
        for k in ('bbox_center', 'bbox_scale'):
            if k in results:
                v = results[k]
                # v 可能是 numpy.ndarray 或 torch.Tensor；都支持 v.ndim
                if getattr(v, 'ndim', 1) == 1:
                    results[k] = v[None, ...]
        return results
