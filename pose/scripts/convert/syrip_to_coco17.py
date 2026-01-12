#!/usr/bin/env python3
"""
Convert SyRIP to COCO17 with train/val/test splits.

Example:
python pose/scripts/convert/syrip_to_coco17.py \
  --source-root /data/xxt/sapiens_data/SyRIP \
  --output-root /data/xxt/sapiens_data/syrip_coco17 \
  --output-prefix syrip \
  --val-ratio 0.1 --seed 42 \
  --copy-mode copy
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SyRIP to COCO17.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--train-ann",
        type=Path,
        default=Path("annotations/200R_1000S/person_keypoints_train_infant.json"),
    )
    parser.add_argument(
        "--test-ann",
        type=Path,
        default=Path("annotations/validate500/person_keypoints_validate_infant.json"),
    )
    parser.add_argument(
        "--train-images-dir",
        type=Path,
        default=Path("images/train_infant"),
    )
    parser.add_argument(
        "--test-images-dir",
        type=Path,
        default=Path("images/validate_infant"),
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-prefix",
        default="syrip",
        help="Prefix for split dirs and annotation filenames (set empty to disable).",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["symlink", "copy", "move"],
        default="symlink",
    )
    parser.add_argument(
        "--flat-images",
        action="store_true",
        help="Store images flat with numbered names in each split.",
    )
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Saved {path}")


def prefixed_name(prefix: str, name: str) -> str:
    return f"{prefix}_{name}" if prefix else name


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


def ensure_size(img_info: Dict, img_path: Path) -> Tuple[int, int]:
    width = img_info.get("width")
    height = img_info.get("height")
    if width and height:
        return int(width), int(height)
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required when width/height are missing.") from exc
    with Image.open(img_path) as img:
        return img.size


def split_ids(
    ids: List[int], val_ratio: float, seed: int
) -> Tuple[Sequence[int], Sequence[int]]:
    if val_ratio < 0 or val_ratio >= 1:
        raise ValueError("val_ratio must be in [0, 1).")
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    if val_ratio == 0:
        return shuffled, []
    val_count = max(1, int(len(shuffled) * val_ratio))
    if val_count >= len(shuffled):
        val_count = len(shuffled) - 1
    return shuffled[val_count:], shuffled[:val_count]


def build_split(
    split_name: str,
    image_ids: Sequence[int],
    images_by_id: Dict[int, Dict],
    anns_by_image: Dict[int, List[Dict]],
    src_images_dir: Path,
    output_root: Path,
    copy_mode: str,
    flat_images: bool,
    skip_missing: bool,
    verbose: bool,
) -> Tuple[Dict, List[Dict]]:
    images_out: List[Dict] = []
    annotations_out: List[Dict] = []
    image_map: List[Dict] = []

    ann_id = 1
    split_dir = output_root / split_name
    for idx, image_id in enumerate(image_ids, start=1):
        img_info = images_by_id[image_id]
        file_name = img_info.get("file_name")
        if not file_name:
            if verbose:
                print(f"[WARN] Missing file_name for image id {image_id}, skip.")
            continue
        src_path = src_images_dir / file_name
        if not src_path.is_file():
            msg = f"Image file missing: {src_path}"
            if skip_missing:
                print(f"[WARN] {msg}")
                continue
            raise FileNotFoundError(msg)

        if flat_images:
            target_name = f"{idx:012d}{src_path.suffix.lower()}"
            target_path = split_dir / target_name
            image_map.append(
                {
                    "id": idx,
                    "file_name": target_name,
                    "source_file_name": file_name,
                }
            )
        else:
            target_name = file_name
            target_path = split_dir / file_name

        width, height = ensure_size(img_info, src_path)
        place_image(src_path, target_path, copy_mode)

        images_out.append(
            {
                "id": idx,
                "file_name": target_name,
                "width": width,
                "height": height,
            }
        )
        for ann in anns_by_image.get(image_id, []):
            bbox = ann.get("bbox", [0.0, 0.0, 0.0, 0.0])
            area = ann.get("area")
            if area is None:
                area = float(bbox[2]) * float(bbox[3])
            keypoints = ann.get("keypoints", [])
            if not keypoints or len(keypoints) < len(COCO17_KEYPOINTS) * 3:
                if verbose:
                    print(
                        f"[WARN] Invalid keypoints for image id {image_id}, skip ann."
                    )
                continue
            num_kpts = ann.get("num_keypoints")
            if num_kpts is None:
                num_kpts = int(sum(1 for v in keypoints[2::3] if v > 0))
            annotations_out.append(
                {
                    "id": ann_id,
                    "image_id": idx,
                    "category_id": int(ann.get("category_id", 1)),
                    "bbox": [float(x) for x in bbox],
                    "area": float(area),
                    "iscrowd": int(ann.get("iscrowd", 0)),
                    "keypoints": [float(x) for x in keypoints[: len(COCO17_KEYPOINTS) * 3]],
                    "num_keypoints": int(num_kpts),
                }
            )
            ann_id += 1

    coco_dict = {
        "images": images_out,
        "annotations": annotations_out,
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


def main() -> None:
    args = parse_args()
    source_root = args.source_root

    train_ann_path = source_root / args.train_ann
    test_ann_path = source_root / args.test_ann
    train_images_dir = source_root / args.train_images_dir
    test_images_dir = source_root / args.test_images_dir

    train_data = load_json(train_ann_path)
    test_data = load_json(test_ann_path)

    train_images_by_id = {img["id"]: img for img in train_data.get("images", [])}
    train_anns_by_image: Dict[int, List[Dict]] = defaultdict(list)
    for ann in train_data.get("annotations", []):
        train_anns_by_image[ann["image_id"]].append(ann)

    train_image_ids = [
        img_id
        for img_id, img in train_images_by_id.items()
        if img_id in train_anns_by_image and img.get("is_labeled", True)
    ]
    if not train_image_ids:
        raise RuntimeError("No labeled training images found.")

    train_ids, val_ids = split_ids(train_image_ids, args.val_ratio, args.seed)
    test_images_by_id = {img["id"]: img for img in test_data.get("images", [])}
    test_anns_by_image: Dict[int, List[Dict]] = defaultdict(list)
    for ann in test_data.get("annotations", []):
        test_anns_by_image[ann["image_id"]].append(ann)
    test_image_ids = [
        img_id
        for img_id, img in test_images_by_id.items()
        if img_id in test_anns_by_image and img.get("is_labeled", True)
    ]
    if not test_image_ids:
        raise RuntimeError("No labeled test images found.")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    train_split = prefixed_name(args.output_prefix, "train2017")
    val_split = prefixed_name(args.output_prefix, "val2017")
    test_split = prefixed_name(args.output_prefix, "test2017")

    train_coco, train_map = build_split(
        train_split,
        train_ids,
        train_images_by_id,
        train_anns_by_image,
        train_images_dir,
        output_root,
        args.copy_mode,
        args.flat_images,
        args.skip_missing,
        args.verbose,
    )
    val_coco, val_map = build_split(
        val_split,
        val_ids,
        train_images_by_id,
        train_anns_by_image,
        train_images_dir,
        output_root,
        args.copy_mode,
        args.flat_images,
        args.skip_missing,
        args.verbose,
    )
    test_coco, test_map = build_split(
        test_split,
        test_image_ids,
        test_images_by_id,
        test_anns_by_image,
        test_images_dir,
        output_root,
        args.copy_mode,
        args.flat_images,
        args.skip_missing,
        args.verbose,
    )

    save_json(
        train_coco,
        output_root / "annotations" / f"person_keypoints_{train_split}.json",
    )
    if val_coco["images"]:
        save_json(
            val_coco,
            output_root / "annotations" / f"person_keypoints_{val_split}.json",
        )
    if test_coco["images"]:
        save_json(
            test_coco,
            output_root / "annotations" / f"person_keypoints_{test_split}.json",
        )

    if args.flat_images:
        if train_map:
            save_json(
                train_map,
                output_root / "annotations" / f"{train_split}_image_map.json",
            )
        if val_map:
            save_json(
                val_map,
                output_root / "annotations" / f"{val_split}_image_map.json",
            )
        if test_map:
            save_json(
                test_map,
                output_root / "annotations" / f"{test_split}_image_map.json",
            )


if __name__ == "__main__":
    main()
