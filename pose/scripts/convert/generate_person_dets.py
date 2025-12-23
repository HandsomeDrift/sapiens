import argparse
import json
from pathlib import Path
from typing import List

from mmdet.apis import inference_detector, init_detector

'''
python pose/scripts/convert/generate_person_dets.py \
    --ann-file /data/xxt/sapiens_data/annotations/person_keypoints_xt_val.json \
    --data-root /data/xxt/sapiens_data \
    --image-prefix xt_val \
    --det-config pose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person_no_nms.py \
    --det-checkpoint /data/xxt/sapiens_lite_host/detector/checkpoints/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
    --out /data/xxt/sapiens_data/person_detection_results/xt_val_detections_AP_H_70_person.json \
    --score-thr 0.3 \
    --device cuda:1
'''


def load_images(ann_file: Path, data_root: Path, image_prefix: str) -> List[dict]:
    with ann_file.open("r") as f:
        coco = json.load(f)
    images = []
    for img in coco.get("images", []):
        file_name = img["file_name"]
        if Path(file_name).is_absolute():
            img_path = Path(file_name)
        elif image_prefix:
            img_path = data_root / image_prefix / file_name
        else:
            img_path = data_root / file_name
        images.append(
            {
                "id": img["id"],
                "path": str(img_path.resolve()),
            }
        )
    return images


def main():
    parser = argparse.ArgumentParser(
        description="Run person detector to produce COCO-style bbox_file JSON"
    )
    parser.add_argument(
        "--ann-file",
        type=Path,
        required=True,
        help="COCO annotation file whose images will be processed (e.g., val.json)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root folder that contains the images referenced in ann-file",
    )
    parser.add_argument(
        "--image-prefix",
        type=str,
        default="xt_val",
        help='Subfolder under data-root for images, e.g. "val2017". Use empty string for none.',
    )
    parser.add_argument(
        "--det-config",
        type=Path,
        required=True,
        help="MMDetection config file (e.g., rtmdet_m_640-8xb32_coco-person_no_nms.py)",
    )
    parser.add_argument(
        "--det-checkpoint",
        type=Path,
        required=True,
        help="Detector checkpoint path",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON path (e.g., person_detection_results/val_det.json)",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.3,
        help="Score threshold to filter detections",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help='Device string, e.g. "cuda" or "cuda:0" or "cpu"'
    )
    args = parser.parse_args()

    images = load_images(args.ann_file, args.data_root, args.image_prefix)
    if not images:
        raise RuntimeError(f"No images found in {args.ann_file}")

    if not args.det_config.exists():
        raise FileNotFoundError(f"det-config not found: {args.det_config}")
    if not args.det_checkpoint.exists():
        raise FileNotFoundError(f"det-checkpoint not found: {args.det_checkpoint}")

    model = init_detector(
        config=str(args.det_config),
        checkpoint=str(args.det_checkpoint),
        device=args.device,
    )

    results = []
    for idx, img_info in enumerate(images, 1):
        if idx % 50 == 0 or idx == len(images):
            print(f"[{idx}/{len(images)}] {img_info['path']}")
        pred = inference_detector(model, img_info["path"])
        # mmdet returns list per class; person is class 0 for COCO
        bboxes = pred.pred_instances.bboxes
        scores = pred.pred_instances.scores
        labels = pred.pred_instances.labels

        for bbox, score, label in zip(bboxes, scores, labels):
            if int(label) != 0:  # class 0 is person
                continue
            if float(score) < args.score_thr:
                continue
            x1, y1, x2, y2 = bbox.tolist()
            results.append(
                {
                    "image_id": img_info["id"],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                    "category_id": 1,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(results, f)
    print(f"Saved {len(results)} detections to {args.out}")


if __name__ == "__main__":
    main()
