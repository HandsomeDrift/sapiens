from typing import List, Optional, Dict
import os, json, warnings
import numpy as np

from mmengine.dataset import BaseDataset
from mmengine.registry import DATASETS

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def _walk_basename_to_path(root: str) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for r, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                ap = os.path.abspath(os.path.join(r, f))
                if f in idx and idx[f] != ap:
                    raise RuntimeError(f'发现重名文件: {f}\n {idx[f]}\n {ap}')
                idx[f] = ap
    return idx

@DATASETS.register_module()
class UnlabeledCocoTopDownDataset(BaseDataset):
    """无标注 Top-Down 数据集（U 路）。
    - 需要 person detection json（与测试时 bbox_file 同结构）。
    - 支持 image_id_map_json: basename->id，则 id->basename 还原真实文件名；
      若不给映射，则回落到 COCO 12位数字名 f"{id:012d}.jpg"。
    - pipeline: LoadImage -> TopdownGetBboxCenterScale/GetBBoxCenterScale -> TopdownAffine -> PackPoseInputs
    """
    METAINFO = dict(from_file='coco')

    def __init__(self,
                 data_root: str,
                 img_folder: str,
                 det_json: str,
                 image_id_map_json: Optional[str] = None,
                 bbox_score_thr: float = 0.0,
                 pipeline: Optional[List[dict]] = None,
                 test_mode: bool = False):
        self.data_root = data_root
        self.img_dir = os.path.join(data_root, img_folder)

        assert os.path.isfile(det_json), f'det_json 不存在: {det_json}'
        with open(det_json, 'r') as f:
            raw = json.load(f)
        dets = [d for d in raw if d.get('category_id', 1) == 1 and d.get('score', 1.0) >= bbox_score_thr]

        id2name = None
        if image_id_map_json is not None:
            assert os.path.isfile(image_id_map_json), f'image_id_map_json 不存在: {image_id_map_json}'
            with open(image_id_map_json, 'r') as f:
                name2id = json.load(f)
            inv = {}
            for k, v in name2id.items():
                if not isinstance(v, int):
                    raise ValueError(f'映射 id 必须为 int: {k}->{v}')
                if v in inv and inv[v] != k:
                    raise ValueError(f'重复 id={v}: {inv[v]} / {k}')
                inv[v] = k
            id2name = inv

        name2path = _walk_basename_to_path(self.img_dir)
        if not name2path:
            raise FileNotFoundError(f'{self.img_dir} 下未扫描到图像')

        self.samples: List[Dict] = []
        miss_map = miss_img = 0
        for d in dets:
            img_id = int(d['image_id'])
            x, y, w, h = d['bbox']          # COCO: xywh
            x2, y2 = x + max(0.0, w), y + max(0.0, h)
            bbox_xyxy = np.asarray([[x, y, x2, y2]], dtype=np.float32)  # (1,4)
            score_arr = np.asarray([float(d.get('score', 1.0))], dtype=np.float32)  # (1,)

            if id2name is not None:
                basename = id2name.get(img_id, None)
                if basename is None:
                    miss_map += 1
                    continue
            else:
                basename = f'{img_id:012d}.jpg'

            full_path = name2path.get(basename, os.path.join(self.img_dir, basename))
            if not os.path.isfile(full_path):
                miss_img += 1
                continue

            self.samples.append(dict(
                img_id=img_id, img_path=full_path,
                bbox=bbox_xyxy, bbox_score=score_arr
            ))

        if miss_map > 0:
            warnings.warn(f'[U-Data] 有 {miss_map} 个 image_id 在映射中找不到，已跳过。')
        if miss_img > 0:
            warnings.warn(f'[U-Data] 有 {miss_img} 个文件路径不存在，已跳过。')

        super().__init__(ann_file=None,
                         data_root=data_root,
                         data_prefix=dict(img=self.img_dir),
                         pipeline=pipeline,
                         test_mode=test_mode)

    def load_data_list(self):
        data_list = []
        for s in self.samples:
            data_list.append(dict(
                img_path=s['img_path'],
                bbox=s['bbox'],                 # (1,4)
                bbox_score=s['bbox_score'],     # (1,)
                img_id=s['img_id'],
            ))
        return data_list
