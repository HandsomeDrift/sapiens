#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MINI-RGBD → COCO Keypoints + detection JSON 预处理脚本。

该脚本会遍历 MINI-RGBD 数据集的 12 个序列，将 2D 关节点文件
(`syn_joints_2Ddep_XXXXX.txt`) 转换为 COCO 风格的关键点标注，并同时
生成半监督阶段所需的人体检测 JSON（利用关键点外接框近似）。

主要功能：
1. 将指定序列划分写入多个 split（如 train/val/unsup）；
2. 输出 `annotations/*.json` 与 `detections/*.json`；
3. 保存 `image_id` 映射，方便后续复用；
4. 默认将 25 个 SMIL 关节点映射到 17 个 COCO 关键点。

示例用法：
python pose/scripts/datasets/mini_rgbd/prepare_mini_rgbd.py \
  --source-root /path/to/MINI-RGBD \
  --output-root /path/to/mini_rgbd_coco \
  --split train=01-09 val=10-12 unsup=01-12

运行前确保安装 Pillow (PIL)：
  pip install Pillow
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

# === 常量定义 ===
MINI_KEYPOINT_NAMES = [
    "global",
    "leftThigh",
    "rightThigh",
    "spine",
    "leftCalf",
    "rightCalf",
    "spine1",
    "leftFoot",
    "rightFoot",
    "spine2",
    "leftToes",
    "rightToes",
    "neck",
    "leftShoulder",
    "rightShoulder",
    "head",
    "leftUpperArm",
    "rightUpperArm",
    "leftForeArm",
    "rightForeArm",
    "leftHand",
    "rightHand",
    "leftFingers",
    "rightFingers",
    "noseVertex",
]

# 0-based joint indices; convert to 1-based when exporting skeleton
SKELETON_EDGES = [
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12),
    (12, 13), (13, 16), (16, 18), (18, 20), (20, 22),
    (12, 14), (14, 17), (17, 19), (19, 21), (21, 23),
    (12, 15), (15, 24),
]

PAD_RATIO = 0.1  # bounding box padding
PAD_PIXELS = 20.0


@dataclass
class FrameSample:
    image_relpath: str
    image_size: Tuple[int, int]  # (width, height)
    keypoints: List[float]  # len=17*3
    num_keypoints: int
    bbox_xywh: List[float]
    area: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MINI-RGBD to COCO format.")
    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help="MINI-RGBD 原始数据根目录（包含 01, 02, ... 序列）。",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="输出目录，将写入 annotations/、detections/、image_id_maps/ 等文件夹。",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        required=True,
        help="指定数据划分，例如 train=01-08,09 val=09-10 unsup=01-12。",
    )
    parser.add_argument("--img-ext", default=".png", help="RGB 图像扩展名（例如 '.png'）。")
    parser.add_argument("--rgb-dir-name", default="rgb", help="序列内 RGB 图像子目录名。")
    parser.add_argument("--joint-dir-name", default="joints_2Ddep", help="序列内 2D 关节标注子目录名。")
    parser.add_argument("--mask-dir-name", default="fg_mask", help="序列内前景掩码子目录名（可选）。")
    parser.add_argument(
        "--bbox-min-side",
        type=float,
        default=40.0,
        help="若关键点生成的 bbox 太小，为其设置的最小边长度（像素）。",
    )
    parser.add_argument(
        "--score",
        type=float,
        default=1.0,
        help="生成检测 JSON 时使用的置信度分数。",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="若某帧缺少图像或关节文件则直接跳过（默认遇到缺失会报错）。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印更多调试信息。",
    )
    return parser.parse_args()


def parse_split_arg(values: Sequence[str]) -> Dict[str, List[str]]:
    """
    支持如下格式：
      train=01-08,09
      val=10-12
    返回一个 dict，value 为零填充的序列编号列表（例如 "01"）。
    """
    splits: Dict[str, List[str]] = {}
    for token in values:
        if "=" not in token:
            raise ValueError(f'Invalid split spec "{token}", expected name=seqs')
        name, seq_spec = token.split("=", 1)
        seq_ids: List[str] = []
        for part in seq_spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                start = int(a)
                end = int(b)
                if end < start:
                    raise ValueError(f"Invalid range {part}")
                seq_ids.extend(f"{i:02d}" for i in range(start, end + 1))
            else:
                if part.isdigit():
                    seq_ids.append(f"{int(part):02d}")
                else:
                    seq_ids.append(part)
        if not seq_ids:
            raise ValueError(f'Split "{name}" is empty.')
        splits[name] = sorted(set(seq_ids))
    return splits


def build_joint_index() -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(MINI_KEYPOINT_NAMES)}


def read_joint_file(path: Path) -> Dict[int, Tuple[float, float, float]]:
    data: Dict[int, Tuple[float, float, float]] = {}
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            x, y, depth, jid = parts[:4]
            try:
                jid_i = int(jid)
            except ValueError:
                continue
            data[jid_i] = (float(x), float(y), float(depth))
    return data


def keypoints_from_mini(
    joint_lookup: Dict[str, int],
    mini_joints: Dict[int, Tuple[float, float, float]],
    image_size: Tuple[int, int],
) -> Tuple[List[float], int]:
    width, height = image_size
    keypoints: List[float] = []
    valid_count = 0
    for kp_name in MINI_KEYPOINT_NAMES:
        idx = joint_lookup.get(kp_name, -1)
        if idx < 0 or idx not in mini_joints:
            keypoints.extend([0.0, 0.0, 0.0])
            continue
        x, y, depth = mini_joints[idx]
        visible = 2
        if math.isnan(x) or math.isnan(y) or depth <= 0:
            visible = 0
        if not (0 <= x < width and 0 <= y < height):
            visible = 0
        if visible == 0:
            keypoints.extend([0.0, 0.0, 0.0])
        else:
            keypoints.extend([float(x), float(y), float(visible)])
            valid_count += 1
    return keypoints, valid_count


def bbox_from_keypoints(
    keypoints: Sequence[float],
    image_size: Tuple[int, int],
    min_side: float,
) -> Tuple[List[float], float]:
    width, height = image_size
    xs = []
    ys = []
    for i in range(0, len(keypoints), 3):
        v = keypoints[i + 2]
        if v <= 0:
            continue
        xs.append(keypoints[i])
        ys.append(keypoints[i + 1])
    if not xs or not ys:
        return [0.0, 0.0, 0.0, 0.0], 0.0

    x_min = max(0.0, min(xs))
    x_max = min(width - 1.0, max(xs))
    y_min = max(0.0, min(ys))
    y_max = min(height - 1.0, max(ys))

    w = x_max - x_min
    h = y_max - y_min
    pad_w = max(PAD_PIXELS, PAD_RATIO * w)
    pad_h = max(PAD_PIXELS, PAD_RATIO * h)
    x1 = max(0.0, x_min - pad_w)
    y1 = max(0.0, y_min - pad_h)
    x2 = min(width - 1.0, x_max + pad_w)
    y2 = min(height - 1.0, y_max + pad_h)
    w = max(min_side, x2 - x1)
    h = max(min_side, y2 - y1)
    area = w * h
    return [x1, y1, w, h], area


def process_split(
    split_name: str,
    seq_ids: Sequence[str],
    args: argparse.Namespace,
    joint_lookup: Dict[str, int],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    source_root: Path = args.source_root
    img_ext = args.img_ext.lower()
    skip_missing = args.skip_missing

    images_json: List[Dict] = []
    annotations_json: List[Dict] = []
    detections_json: List[Dict] = []

    ann_id = 1
    image_id = 1
    seen_images = set()

    for seq_id in seq_ids:
        seq_dir = source_root / seq_id
        if not seq_dir.is_dir():
            msg = f'Sequence folder "{seq_id}" not found under {source_root}'
            if skip_missing:
                print(f"[WARN] {msg}, skip.")
                continue
            raise FileNotFoundError(msg)

    joint_dir = seq_dir / args.joint_dir_name
    if not joint_dir.is_dir():
        msg = f'Joint directory "{joint_dir}" not found.'
        if skip_missing:
            print(f"[WARN] {msg} skip sequence.")
            return [], [], []
        raise FileNotFoundError(msg)

    joint_files = sorted(joint_dir.glob("syn_joints_2Ddep_*.txt"))
        if args.verbose:
            print(f"[INFO] Split {split_name}, seq {seq_id}: {len(joint_files)} frames")
        for joint_file in joint_files:
            stem = joint_file.stem.replace("syn_joints_2Ddep_", "")
            img_file = seq_dir / args.rgb_dir_name / f"syn_{stem}{img_ext}"
            if not img_file.is_file():
                msg = f"Image file missing: {img_file}"
                if skip_missing:
                    print(f"[WARN] {msg}, skip frame.")
                    continue
                raise FileNotFoundError(msg)

            joints = read_joint_file(joint_file)
            with Image.open(img_file) as img:
                width, height = img.size

            keypoints, num_kpt = keypoints_from_mini(joint_lookup, joints, (width, height))
            bbox_xywh, area = bbox_from_keypoints(keypoints, (width, height), args.bbox_min_side)
            if num_kpt == 0 or area <= 0:
                if args.verbose:
                    print(f"[WARN] No valid keypoints in {joint_file}, skip.")
                continue

            image_rel = f"{seq_id}/{args.rgb_dir_name}/{img_file.name}"
            if image_rel in seen_images:
                # MINI-RGBD 理论上不会重复，但以防万一
                raise RuntimeError(f"Duplicate image relative path detected: {image_rel}")
            seen_images.add(image_rel)

            images_json.append(
                dict(
                    id=image_id,
                    file_name=image_rel,
                    width=width,
                    height=height,
                )
            )
            annotations_json.append(
                dict(
                    id=ann_id,
                    image_id=image_id,
                    category_id=1,
                    keypoints=[round(float(x), 3) for x in keypoints],
                    num_keypoints=num_kpt,
                    bbox=[round(float(x), 3) for x in bbox_xywh],
                    area=round(float(area), 3),
                    iscrowd=0,
                )
            )
            detections_json.append(
                dict(
                    image_id=image_id,
                    category_id=1,
                    bbox=[round(float(x), 3) for x in bbox_xywh],
                    score=float(args.score),
                )
            )
            ann_id += 1
            image_id += 1

    return images_json, annotations_json, detections_json


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Saved {path}")


def main() -> None:
    args = parse_args()
    splits = parse_split_arg(args.split)
    joint_lookup = build_joint_index()

    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    categories = [
        dict(
            id=1,
            name="person",
            supercategory="person",
            keypoints=MINI_KEYPOINT_NAMES,
            skeleton=[[a + 1, b + 1] for a, b in SKELETON_EDGES],
        )
    ]

    for split_name, seq_ids in splits.items():
        images, annotations, detections = process_split(
            split_name, seq_ids, args, joint_lookup
        )

        if not images:
            print(f"[WARN] Split {split_name} has no images after processing, skip saving.")
            continue

        coco_dict = dict(
            images=images,
            annotations=annotations,
            categories=categories,
        )
        det_list = detections
        image_map = {img["file_name"]: img["id"] for img in images}

        ann_path = output_root / "annotations" / f"mini_rgbd_{split_name}_keypoints.json"
        det_path = output_root / "detections" / f"mini_rgbd_{split_name}_person_dets.json"
        map_path = output_root / "image_id_maps" / f"mini_rgbd_{split_name}_image_ids.json"

        save_json(coco_dict, ann_path)
        save_json(det_list, det_path)
        save_json(image_map, map_path)


if __name__ == "__main__":
    main()
