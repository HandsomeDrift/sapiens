#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用 MMDetection 检测器生成 COCO 风格的 person 检测结果 JSON。
- 兼容 mmdet 2.x/3.x 的 inference 输出
- 可选读取 COCO 标注文件以获得 image_id（强烈推荐）
- 否则按文件顺序分配自增 image_id，并另存 mapping 方便后续复用

用法示例（见文件末尾注释）：
python tools/gen_person_dets.py \
  --det-config /path/to/your_mmdet_cfg.py \
  --det-checkpoint /path/to/your_detector.pth \
  --input-dir /path/to/images_or_subdirs \
  --out-json /path/to/YourDataset_detections_person.json \
  --coco-ann /path/to/your_annotations.json \
  --det-cat-id 0 --bbox-thr 0.3 --nms-thr 0.5 --device cuda:0
"""
import os, re, json, argparse, glob, warnings
from typing import Dict, List, Tuple

import numpy as np
import torch

# ====== mmdet 初始化 ======
try:
    from mmdet.apis import init_detector, inference_detector
    HAS_MMDET = True
except Exception as e:
    HAS_MMDET = False
    _INIT_ERR = e

# ====== NMS 实现（优先用 torchvision；否则纯 PyTorch 实现）======
def nms_torch(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """纯 PyTorch NMS（若可用 torchvision.ops.nms 会自动替代）"""
    # boxes: (N, 4) [x1,y1,x2,y2], scores: (N,)
    # 返回保留索引
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long)

try:
    from torchvision.ops import nms as tv_nms  # type: ignore
    def do_nms(boxes, scores, thr):
        return tv_nms(boxes, scores, thr)
except Exception:
    def do_nms(boxes, scores, thr):
        return nms_torch(boxes, scores, thr)

# ====== 工具函数 ======
_IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def list_images(input_dir_or_txt: str) -> List[str]:
    if os.path.isdir(input_dir_or_txt):
        paths = []
        for ext in _IMG_EXT:
            paths += glob.glob(os.path.join(input_dir_or_txt, f'**/*{ext}'), recursive=True)
        paths = sorted(paths)
        return paths
    elif os.path.isfile(input_dir_or_txt) and input_dir_or_txt.lower().endswith('.txt'):
        with open(input_dir_or_txt, 'r') as f:
            lines = [x.strip() for x in f.readlines() if x.strip()]
        return lines
    else:
        raise FileNotFoundError(f'input {input_dir_or_txt} is neither a dir nor a .txt list')

def build_image_id_map(coco_ann: str, image_paths: List[str]) -> Dict[str, int]:
    """
    从 COCO 标注文件建立 file_name -> image_id 的映射。
    匹配策略：只用 basename（不含路径），区分大小写。
    """
    with open(coco_ann, 'r') as f:
        ann = json.load(f)
    name_to_id = {os.path.basename(im['file_name']): int(im['id']) for im in ann['images']}
    mapping = {}
    miss = 0
    for p in image_paths:
        bn = os.path.basename(p)
        if bn in name_to_id:
            mapping[p] = name_to_id[bn]
        else:
            miss += 1
    if miss > 0:
        warnings.warn(f'{miss} images were not found in COCO ann "images.file_name". They will be skipped.')
    return mapping

def assign_sequential_ids(image_paths: List[str]) -> Dict[str, int]:
    """没有 COCO 标注时，按顺序分配 image_id（从 1 开始），并返回映射。"""
    return {p: i+1 for i, p in enumerate(image_paths)}

def det_result_to_xywh(bboxes_xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = np.split(bboxes_xyxy, 4, axis=1)
    w = np.maximum(0.0, x2 - x1)
    h = np.maximum(0.0, y2 - y1)
    return np.concatenate([x1, y1, w, h], axis=1)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det-config', required=True, help='MMDetection config (.py)')
    ap.add_argument('--det-checkpoint', required=True, help='Detector checkpoint (.pth)')
    ap.add_argument('--input', required=True, help='Image dir or a .txt list')
    ap.add_argument('--out-json', required=True, help='Output detection json (COCO style)')
    ap.add_argument('--coco-ann', default=None, help='Optional COCO annotation JSON to fetch image_id')
    ap.add_argument('--det-cat-id', type=int, default=0, help='Class index for person in detector output (often 0)')
    ap.add_argument('--bbox-thr', type=float, default=0.30, help='Score threshold before NMS')
    ap.add_argument('--nms-thr', type=float, default=0.50, help='IoU threshold for NMS')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--max-per-image', type=int, default=300, help='Optional cap after NMS')
    return ap.parse_args()

def main():
    if not HAS_MMDET:
        raise RuntimeError(f'mmdet not available: {_INIT_ERR}')
    args = parse_args()
    imgs = list_images(args.input)
    if len(imgs) == 0:
        raise RuntimeError('No images found')

    # image_id 映射
    if args.coco_ann and os.path.isfile(args.coco_ann):
        imgid_map = build_image_id_map(args.coco_ann, imgs)
        imgs = [p for p in imgs if p in imgid_map]  # 只保留能匹配到 ann 的图片
    else:
        imgid_map = assign_sequential_ids(imgs)
        # 顺手保存一份映射，方便后续数据集适配
        map_path = os.path.splitext(args.out_json)[0] + '_image_id_map.json'
        with open(map_path, 'w') as f:
            json.dump({os.path.basename(k): v for k, v in imgid_map.items()}, f, indent=2)
        print(f'[Info] Saved image_id mapping to {map_path}')

    # 初始化检测器
    model = init_detector(args.det_config, args.det_checkpoint, device=args.device)

    results = []
    for idx, img_path in enumerate(imgs):
        if idx % 50 == 0:
            print(f'[{idx}/{len(imgs)}] {img_path}')
        det = inference_detector(model, img_path)

        # 兼容 mmdet 3.x (DetDataSample) 与 2.x (list of arrays)
        bboxes_xyxy, scores, labels = None, None, None
        if hasattr(det, 'pred_instances'):
            inst = det.pred_instances
            # Tensor -> numpy
            bboxes_xyxy = inst.bboxes.detach().cpu().numpy()
            scores = inst.scores.detach().cpu().numpy()
            labels = inst.labels.detach().cpu().numpy()
        elif isinstance(det, (list, tuple)):
            # list[cls] of ndarray (N_i, 5), last dim: x1,y1,x2,y2,score
            all_b = det
            if args.det_cat_id >= len(all_b):
                warnings.warn(f'det-cat-id {args.det_cat_id} out of range (num_classes={len(all_b)}), skip')
                continue
            arr = all_b[args.det_cat_id]
            if arr is None or len(arr) == 0:
                arr = np.zeros((0, 5), dtype=np.float32)
            bboxes_xyxy = arr[:, :4]
            scores = arr[:, 4]
            labels = np.full((len(scores),), args.det_cat_id, dtype=np.int64)
        else:
            warnings.warn('Unknown detection result type; skip')
            continue

        # 只保留 person 类 + 分数阈值
        m = (labels == args.det_cat_id) & (scores >= args.bbox_thr)
        if m.sum() == 0:
            continue
        b = torch.from_numpy(bboxes_xyxy[m]).float()
        s = torch.from_numpy(scores[m]).float()

        # NMS
        if len(s) > 0 and args.nms_thr is not None and args.nms_thr < 1.0:
            keep = do_nms(b, s, args.nms_thr).cpu().numpy()
            b = b[keep]
            s = s[keep]

        # 限制每图最多框数（可选）
        if args.max_per_image and len(s) > args.max_per_image:
            topk = torch.topk(s, args.max_per_image).indices
            b = b[topk]
            s = s[topk]

        # 转 COCO xywh
        xywh = det_result_to_xywh(b.numpy())
        image_id = int(imgid_map[img_path])
        for j in range(xywh.shape[0]):
            results.append({
                'image_id': image_id,
                'category_id': 1,  # COCO person
                'bbox': [float(x) for x in xywh[j].tolist()],
                'score': float(s[j].item())
            })

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(results, f)
    print(f'[OK] Wrote {len(results)} detections to {args.out_json}')

if __name__ == '__main__':
    main()
