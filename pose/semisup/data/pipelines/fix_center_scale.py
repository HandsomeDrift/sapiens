from mmengine.registry import TRANSFORMS

@TRANSFORMS.register_module()
class EnsureBatchedCenterScale:
    """把 {bbox_center,bbox_scale} 从 (2,) 改成 (1,2)，并同步别名 {center,scale}（可选）"""
    def __call__(self, results):
        for k in ('bbox_center', 'bbox_scale'):
            if k in results:
                v = results[k]
                if getattr(v, 'ndim', 1) == 1:
                    v = v[None, ...]
                results[k] = v
        if 'bbox_center' in results:
            results['center'] = results['bbox_center']
        if 'bbox_scale' in results:
            results['scale'] = results['bbox_scale']
        return results
