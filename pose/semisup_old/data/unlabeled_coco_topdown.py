from typing import List, Optional, Dict
import os
import numpy as np
import json
import warnings
from collections import defaultdict

from mmengine.dataset import BaseDataset
from mmengine.registry import DATASETS


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _walk_basename_to_path(root: str) -> Dict[str, str]:
    """递归扫描 root，建立 basename -> full_path 的索引。
    若发现同名文件（不同子目录）会直接报错，避免歧义。
    """
    idx: Dict[str, str] = {}
    for r, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                if f in idx and os.path.abspath(os.path.join(r, f)) != idx[f]:
                    raise RuntimeError(
                        f'发现重名文件: {f}\n  {idx[f]}\n  {os.path.abspath(os.path.join(r, f))}\n'
                        '请去重，或改用“相对路径→id”的映射。'
                    )
                idx[f] = os.path.abspath(os.path.join(r, f))
    return idx


@DATASETS.register_module()
class UnlabeledCocoTopDownDataset(BaseDataset):
    """无标注 Top-Down 数据集（U 路）。
    - 需要 person detection json（与测试集 bbox_file 相同结构）。
    - 支持通过 image_id_map_json( basename->id ) 将 id 还原成真实文件名；
      若不给映射，则退回 COCO 12位数字文件名规则：f"{id:012d}.jpg"。
    - pipeline 内至少应包含 LoadImage 或等价读取步骤；强/弱增在模型/Hook侧处理。
    """

    METAINFO = dict(from_file='coco')

    def __init__(self,
                 data_root: str,
                 img_folder: str = 'val2017',
                 det_json: Optional[str] = None,
                 image_id_map_json: Optional[str] = None,  # 新增：basename->id 映射
                 bbox_score_thr: float = 0.0,               # 新增：分数阈值
                 pipeline: Optional[List[dict]] = None,
                 test_mode: bool = False):
        self.data_root = data_root
        self.img_dir = os.path.join(data_root, img_folder)

        # 读取检测结果
        assert det_json is not None and os.path.isfile(det_json), \
            f'det_json (bbox_file) 缺失或不存在: {det_json}'
        with open(det_json, 'r') as f:
            raw = json.load(f)
        # 只取 person 类，过滤低分
        dets = [d for d in raw if d.get('category_id', 1) == 1 and d.get('score', 1.0) >= bbox_score_thr]

        # 建立 id -> basename 的映射（如果提供了 image_id_map_json）
        id2name: Optional[Dict[int, str]] = None
        if image_id_map_json is not None:
            assert os.path.isfile(image_id_map_json), f'image_id_map_json 不存在: {image_id_map_json}'
            with open(image_id_map_json, 'r') as f:
                name2id = json.load(f)  # 期望: {basename(str): id(int)}
            # 自检：id 唯一
            inv: Dict[int, str] = {}
            for k, v in name2id.items():
                if not isinstance(v, int):
                    raise ValueError(f'映射中的 id 必须是 int：{k} -> {v}')
                if v in inv and inv[v] != k:
                    raise ValueError(f'映射中出现重复 id={v}: {inv[v]} / {k}')
                inv[v] = k
            id2name = inv

        # 建立 basename -> full_path 的索引（支持子目录）
        name2path = _walk_basename_to_path(self.img_dir)
        if len(name2path) == 0:
            raise FileNotFoundError(f'在 {self.img_dir} 下未扫描到任何图像')

        # 组装样本
        self.samples: List[Dict] = []
        miss_img, miss_map = 0, 0
        for d in dets:
            if d.get('category_id', 1) != 1:
                continue
            img_id = int(d['image_id'])
            x, y, w, h = d['bbox']            # COCO: xywh
            x2, y2 = x + max(0.0, w), y + max(0.0, h)
            bbox_xyxy = np.asarray([[x, y, x2, y2]], dtype=np.float32)  # (1,4) ← 重要
            score_arr  = np.asarray([float(d.get('score', 1.0))], dtype=np.float32)  # (1,) ← 匹配长度

            # 通过映射还原 basename；否则按 COCO 12位数规则
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

            self.samples.append({
                'img_id': img_id,
                'img_path': full_path,
                'bbox': bbox_xyxy,
                'bbox_score': score_arr, # ← 跟随实例数的 (1,)
            })

        if miss_map > 0:
            warnings.warn(f'[UnlabeledCocoTopDownDataset] 有 {miss_map} 个 image_id 在映射中找不到，已跳过。')
        if miss_img > 0:
            warnings.warn(f'[UnlabeledCocoTopDownDataset] 有 {miss_img} 个文件路径不存在，已跳过。')

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
                bbox=s['bbox'],            # ← 已是 np.ndarray(xyxy, float32)
                bbox_score=s.get('bbox_score'), # (1,), 可无则不传
                img_id=s['img_id'],
                # det_score=s.get('score', 1.0),
            ))
        return data_list
