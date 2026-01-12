#!/usr/bin/env python3
"""
Convert MINI-RGBD to COCO17 keypoints with train/val/test splits.

Example:
  python pose/scripts/convert/mini_rgbd_to_coco17.py \
    --source-root /data/MINI-RGBD \
    --output-root /data/mini_rgbd_coco17 \
    --split train2017=01-08 val2017=09-10 test2017=11-12 \
    --output-prefix mini_rgbd \
    --copy-mode symlink
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


COCO17_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO17_SKELETON = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
]

COCO_TO_MINI = {
    "nose": "noseVertex",
    "left_shoulder": "leftUpperArm",
    "right_shoulder": "rightUpperArm",
    "left_elbow": "leftForeArm",
    "right_elbow": "rightForeArm",
    "left_wrist": "leftHand",
    "right_wrist": "rightHand",
    "left_hip": "leftThigh",
    "right_hip": "rightThigh",
    "left_knee": "leftCalf",
    "right_knee": "rightCalf",
    "left_ankle": "leftFoot",
    "right_ankle": "rightFoot",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MINI-RGBD to COCO17 keypoints."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--split",
        nargs="+",
        default=["train2017=01-08", "val2017=09-10", "test2017=11-12"],
        help="Split spec, e.g. train2017=01-08 val2017=09-10 test2017=11-12",
    )
    parser.add_argument(
        "--output-prefix",
        default="mini_rgbd",
        help="Prefix for split dirs and annotation filenames (set empty to disable).",
    )
    parser.add_argument(
        "--jointlist",
        type=Path,
        default=None,
        help="Path to jointlist.txt (defaults to <source-root>/jointlist.txt).",
    )
    parser.add_argument("--img-ext", default=".png")
    parser.add_argument("--rgb-dir-name", default="rgb")
    parser.add_argument("--joint-dir-name", default="joints_2Ddep")
    parser.add_argument(
        "--copy-mode",
        choices=["symlink", "copy", "move"],
        default="symlink",
        help="How to place images into train2017/val2017/test2017.",
    )
    parser.add_argument(
        "--flat-images",
        action="store_true",
        help="Store images flat with numbered names in each split.",
    )
    parser.add_argument(
        "--bbox-min-side",
        type=float,
        default=40.0,
        help="Minimum side length for bbox if keypoints are too tight.",
    )
    parser.add_argument("--pad-ratio", type=float, default=0.1)
    parser.add_argument("--pad-pixels", type=float, default=20.0)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def parse_split_arg(values: Sequence[str]) -> Dict[str, List[str]]:
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
                start_s, end_s = part.split("-", 1)
                start = int(start_s)
                end = int(end_s)
                if end < start:
                    raise ValueError(f"Invalid range {part}")
                seq_ids.extend(f"{i:02d}" for i in range(start, end + 1))
            else:
                seq_ids.append(f"{int(part):02d}" if part.isdigit() else part)
        if not seq_ids:
            raise ValueError(f'Split "{name}" is empty.')
        splits[name] = sorted(set(seq_ids))
    return splits


def read_jointlist(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Joint list not found: {path}")
    names: List[str] = []
    with path.open("r") as f:
        for line in f:
            name = line.strip()
            if name:
                names.append(name)
    if not names:
        raise ValueError(f"Joint list is empty: {path}")
    return names


def prefixed_name(prefix: str, name: str) -> str:
    return f"{prefix}_{name}" if prefix else name


def detect_joint_id_base(joint_ids: Sequence[int], joint_count: int) -> int:
    if not joint_ids:
        return 0
    min_id = min(joint_ids)
    max_id = max(joint_ids)
    if min_id == 0 and max_id <= joint_count - 1:
        return 0
    if min_id == 1 and max_id <= joint_count:
        return 1
    if min_id == 1:
        return 1
    return 0


def read_joint_file(
    path: Path,
    joint_count: int,
    joint_id_base: Optional[int],
) -> Tuple[Dict[int, Tuple[float, float, float]], int]:
    raw: Dict[int, Tuple[float, float, float]] = {}
    joint_ids: List[int] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            x_str, y_str, depth_str, jid_str = parts[:4]
            try:
                jid = int(float(jid_str))
            except ValueError:
                continue
            joint_ids.append(jid)
            raw[jid] = (float(x_str), float(y_str), float(depth_str))
    base = joint_id_base
    if base is None:
        base = detect_joint_id_base(joint_ids, joint_count)
    data: Dict[int, Tuple[float, float, float]] = {}
    for jid, value in raw.items():
        idx = jid - base
        if 0 <= idx < joint_count:
            data[idx] = value
    return data, base


def resolve_dir(seq_dir: Path, preferred: str) -> Path:
    if preferred:
        candidate = seq_dir / preferred
        if candidate.is_dir():
            return candidate
    return seq_dir


def find_image_file(
    rgb_dir: Path,
    stem: str,
    img_ext: str,
) -> Optional[Path]:
    candidate = rgb_dir / f"syn_{stem}{img_ext}"
    if candidate.is_file():
        return candidate
    for alt in rgb_dir.glob(f"syn_{stem}.*"):
        if alt.is_file():
            return alt
    return None


def keypoints_from_mini(
    mini_joints: Dict[int, Tuple[float, float, float]],
    mini_name_to_idx: Dict[str, int],
    image_size: Tuple[int, int],
) -> Tuple[List[float], int]:
    width, height = image_size
    keypoints: List[float] = []
    valid_count = 0
    for coco_name in COCO17_KEYPOINTS:
        mini_name = COCO_TO_MINI.get(coco_name)
        if not mini_name:
            keypoints.extend([0.0, 0.0, 0.0])
            continue
        mini_idx = mini_name_to_idx.get(mini_name)
        if mini_idx is None or mini_idx not in mini_joints:
            keypoints.extend([0.0, 0.0, 0.0])
            continue
        x, y, depth = mini_joints[mini_idx]
        visible = 2
        if math.isnan(x) or math.isnan(y) or depth <= 0:
            visible = 0
        if not (0.0 <= x < width and 0.0 <= y < height):
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
    pad_ratio: float,
    pad_pixels: float,
) -> Tuple[List[float], float]:
    width, height = image_size
    xs: List[float] = []
    ys: List[float] = []
    for i in range(0, len(keypoints), 3):
        if keypoints[i + 2] <= 0:
            continue
        xs.append(float(keypoints[i]))
        ys.append(float(keypoints[i + 1]))
    if not xs or not ys:
        return [0.0, 0.0, 0.0, 0.0], 0.0

    x_min = max(0.0, min(xs))
    x_max = min(width - 1.0, max(xs))
    y_min = max(0.0, min(ys))
    y_max = min(height - 1.0, max(ys))

    w = x_max - x_min
    h = y_max - y_min
    pad_w = max(pad_pixels, pad_ratio * w)
    pad_h = max(pad_pixels, pad_ratio * h)
    x1 = max(0.0, x_min - pad_w)
    y1 = max(0.0, y_min - pad_h)
    x2 = min(width - 1.0, x_max + pad_w)
    y2 = min(height - 1.0, y_max + pad_h)
    w = max(min_side, x2 - x1)
    h = max(min_side, y2 - y1)
    area = w * h
    return [x1, y1, w, h], area


def place_image(src: Path, dst: Path, copy_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy_mode == "copy":
        shutil.copy2(src, dst)
    elif copy_mode == "move":
        shutil.move(src, dst)
    else:
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copy2(src, dst)


def build_coco(
    split_name: str,
    seq_ids: Sequence[str],
    args: argparse.Namespace,
    mini_name_to_idx: Dict[str, int],
) -> Tuple[Dict, List[Dict]]:
    source_root: Path = args.source_root
    img_ext = args.img_ext
    skip_missing = args.skip_missing

    images_json: List[Dict] = []
    annotations_json: List[Dict] = []
    image_map: List[Dict] = []

    ann_id = 1
    image_id = 1
    seen_images = set()
    joint_id_base: Optional[int] = None

    split_dir_name = prefixed_name(args.output_prefix, split_name)
    split_dir = args.output_root / split_dir_name

    for seq_id in seq_ids:
        seq_dir = source_root / seq_id
        if not seq_dir.is_dir():
            msg = f'Sequence folder "{seq_id}" not found under {source_root}'
            if skip_missing:
                print(f"[WARN] {msg}, skip.")
                continue
            raise FileNotFoundError(msg)

        joint_dir = resolve_dir(seq_dir, args.joint_dir_name)
        rgb_dir = resolve_dir(seq_dir, args.rgb_dir_name)

        joint_files = sorted(joint_dir.glob("syn_joints_2Ddep_*.txt"))
        if not joint_files:
            msg = f"No joint files found under {joint_dir}"
            if skip_missing:
                print(f"[WARN] {msg}, skip sequence {seq_id}.")
                continue
            raise FileNotFoundError(msg)

        if args.verbose:
            print(f"[INFO] Split {split_name}, seq {seq_id}: {len(joint_files)} frames")

        for joint_file in joint_files:
            stem = joint_file.stem.replace("syn_joints_2Ddep_", "")
            img_file = find_image_file(rgb_dir, stem, img_ext)
            if img_file is None or not img_file.is_file():
                msg = f"Image file missing for frame {stem} under {rgb_dir}"
                if skip_missing:
                    print(f"[WARN] {msg}, skip frame.")
                    continue
                raise FileNotFoundError(msg)

            joints, joint_id_base = read_joint_file(
                joint_file, len(mini_name_to_idx), joint_id_base
            )
            with Image.open(img_file) as img:
                width, height = img.size

            keypoints, num_kpt = keypoints_from_mini(
                joints, mini_name_to_idx, (width, height)
            )
            bbox_xywh, area = bbox_from_keypoints(
                keypoints, (width, height), args.bbox_min_side, args.pad_ratio, args.pad_pixels
            )
            if num_kpt == 0 or area <= 0:
                if args.verbose:
                    print(f"[WARN] No valid keypoints in {joint_file}, skip.")
                continue

            if args.flat_images:
                file_name = f"{image_id:012d}{img_file.suffix.lower()}"
                target_path = split_dir / file_name
                image_map.append(
                    {
                        "id": image_id,
                        "file_name": file_name,
                        "source_path": f"{seq_id}/{img_file.name}",
                    }
                )
            else:
                rel_path = Path(seq_id) / img_file.name
                file_name = rel_path.as_posix()
                target_path = split_dir / rel_path

            if file_name in seen_images:
                raise RuntimeError(f"Duplicate image path detected: {file_name}")
            seen_images.add(file_name)
            place_image(img_file, target_path, args.copy_mode)

            images_json.append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                }
            )
            annotations_json.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "keypoints": [round(float(x), 3) for x in keypoints],
                    "num_keypoints": num_kpt,
                    "bbox": [round(float(x), 3) for x in bbox_xywh],
                    "area": round(float(area), 3),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
            image_id += 1

    coco_dict = {
        "images": images_json,
        "annotations": annotations_json,
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": COCO17_KEYPOINTS,
                "skeleton": COCO17_SKELETON,
            }
        ],
    }
    return coco_dict, image_map


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Saved {path}")


def main() -> None:
    args = parse_args()
    splits = parse_split_arg(args.split)
    jointlist_path = args.jointlist or (args.source_root / "jointlist.txt")
    mini_names = read_jointlist(jointlist_path)
    mini_name_to_idx = {name: idx for idx, name in enumerate(mini_names)}

    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    for split_name, seq_ids in splits.items():
        coco_dict, image_map = build_coco(
            split_name, seq_ids, args, mini_name_to_idx
        )
        if not coco_dict["images"]:
            print(f"[WARN] Split {split_name} has no images, skip saving.")
            continue

        prefixed_split = prefixed_name(args.output_prefix, split_name)
        ann_path = (
            output_root
            / "annotations"
            / f"person_keypoints_{prefixed_split}.json"
        )
        save_json(coco_dict, ann_path)
        if args.flat_images and image_map:
            map_path = (
                output_root
                / "annotations"
                / f"{prefixed_split}_image_map.json"
            )
            save_json(image_map, map_path)


if __name__ == "__main__":
    main()
