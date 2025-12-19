import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


COCO17_ORDER = [
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


def parse_labelme_file(path: Path) -> Tuple[Dict, List[float]]:
    try:
        with path.open("r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return {}, []

    image_width = data.get("imageWidth")
    image_height = data.get("imageHeight")
    image_path = data.get("imagePath", path.with_suffix(".png").name)

    keypoints = [0.0] * (len(COCO17_ORDER) * 3)
    valid_points: List[Tuple[float, float]] = []

    for shape in data.get("shapes", []):
        label = shape.get("label")
        if label not in COCO17_ORDER:
            continue
        idx = COCO17_ORDER.index(label)
        points = shape.get("points", [])
        if not points:
            continue
        x, y = points[0]
        keypoints[idx * 3] = float(x)
        keypoints[idx * 3 + 1] = float(y)
        keypoints[idx * 3 + 2] = 2.0
        valid_points.append((float(x), float(y)))

    if not valid_points:
        return {}, []

    xs, ys = zip(*valid_points)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w, h = x_max - x_min, y_max - y_min
    pad = 0.1 * max(w, h)
    x_min = max(0.0, x_min - pad)
    y_min = max(0.0, y_min - pad)
    x_max = x_max + pad
    y_max = y_max + pad
    if image_width:
        x_max = min(float(image_width), x_max)
    if image_height:
        y_max = min(float(image_height), y_max)
    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)

    ann = {
        "bbox": [x_min, y_min, w, h],
        "area": w * h,
        "keypoints": keypoints,
        "num_keypoints": int(sum(1 for v in keypoints[2::3] if v > 0)),
        "iscrowd": 0,
        "category_id": 1,
        "image_path": image_path,
        "image_width": image_width,
        "image_height": image_height,
    }
    return ann, keypoints


def resolve_image_path(json_path: Path, image_path: str) -> Path:
    candidate = (json_path.parent / image_path).resolve()
    if candidate.exists():
        return candidate
    fallback = json_path.with_suffix(".png").resolve()
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Image not found for {json_path}")


def collect_samples(root: Path, strict: bool) -> List[Tuple[Path, Dict, Path]]:
    samples = []
    for json_path in root.rglob("*.json"):
        if "_labelme_init_coco17" in json_path.name:
            continue
        ann, _ = parse_labelme_file(json_path)
        if not ann:
            if strict:
                raise RuntimeError(f"Invalid JSON: {json_path}")
            continue
        img_path = resolve_image_path(json_path, ann["image_path"])
        samples.append((json_path, ann, img_path))
    return samples


def save_coco(
    samples: List[Tuple[Path, Dict, Path]],
    root: Path,
    out_path: Path,
    image_root: Path,
    split_name: str,
    copy_mode: str,
    flat_images: bool,
) -> None:
    images = []
    annotations = []
    image_map = []
    for idx, (json_path, ann_info, img_path) in enumerate(samples):
        rel_path = img_path.relative_to(root)
        if flat_images:
            new_name = f"{idx + 1:08d}{img_path.suffix.lower()}"
            target_path = image_root / new_name
            file_name = new_name
        else:
            target_path = image_root / rel_path
            file_name = rel_path.as_posix()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            continue
        if copy_mode == "copy":
            shutil.copy2(img_path, target_path)
        elif copy_mode == "move":
            shutil.move(img_path, target_path)
        else:
            try:
                target_path.symlink_to(img_path)
            except OSError:
                shutil.copy2(img_path, target_path)

        image_id = idx + 1
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": ann_info["image_width"],
                "height": ann_info["image_height"],
            }
        )
        if flat_images:
            image_map.append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "source_path": rel_path.as_posix(),
                }
            )
        annotations.append(
            {
                "id": idx + 1,
                "image_id": image_id,
                "bbox": ann_info["bbox"],
                "area": ann_info["area"],
                "iscrowd": 0,
                "category_id": 1,
                "num_keypoints": ann_info["num_keypoints"],
                "keypoints": ann_info["keypoints"],
            }
        )

    categories = [
        {
            "id": 1,
            "name": "person",
            "supercategory": "person",
            "keypoints": COCO17_ORDER,
            "skeleton": [],
        }
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(
            {
                "images": images,
                "annotations": annotations,
                "categories": categories,
            },
            f,
        )
    print(f"Saved {len(images)} {split_name} samples to {out_path}")
    if flat_images and image_map:
        map_path = out_path.with_name(f"{split_name}_image_map.json")
        with map_path.open("w") as f:
            json.dump(image_map, f, indent=2)
        print(f"Saved {split_name} image map to {map_path}")


def split_samples(
    samples: List[Tuple[Path, Dict, Path]], train_ratio: float, seed: int
):
    random.seed(seed)
    random.shuffle(samples)
    split = int(len(samples) * train_ratio)
    return samples[:split], samples[split:]


def main():
    parser = argparse.ArgumentParser(
        description="Convert LabelMe COCO17 keypoints to COCO format"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Input root containing nested folders with JSON/PNG files",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output root to create train2017/val2017 and annotations/",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["symlink", "copy", "move"],
        default="symlink",
        help="How to place images into train2017/val2017",
    )
    parser.add_argument(
        "--flat-images",
        action="store_true",
        help="Place all images directly under train2017/val2017 with flat names",
    )
    parser.add_argument(
        "--strict-json",
        action="store_true",
        help="Fail on invalid JSON instead of skipping",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Train split ratio",
    )
    parser.add_argument(
        "--train-name",
        type=str,
        default="xt_train",
        help='Folder name for training images, e.g. "xt_train"',
    )
    parser.add_argument(
        "--val-name",
        type=str,
        default="xt_val",
        help='Folder name for validation images, e.g. "xt_val"',
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    args = parser.parse_args()

    samples = collect_samples(args.input_root, args.strict_json)
    if not samples:
        raise RuntimeError(f"No valid annotations found under {args.input_root}")

    train, val = split_samples(samples, args.train_ratio, args.seed)
    if not train or not val:
        raise RuntimeError("Split resulted in empty train or val set")

    train_images = args.output_root / args.train_name
    val_images = args.output_root / args.val_name
    ann_dir = args.output_root / "annotations"

    save_coco(
        train,
        args.input_root,
        ann_dir / f"person_keypoints_{args.train_name}.json",
        train_images,
        args.train_name,
        args.copy_mode,
        args.flat_images,
    )
    save_coco(
        val,
        args.input_root,
        ann_dir / f"person_keypoints_{args.val_name}.json",
        val_images,
        args.val_name,
        args.copy_mode,
        args.flat_images,
    )


if __name__ == "__main__":
    main()
