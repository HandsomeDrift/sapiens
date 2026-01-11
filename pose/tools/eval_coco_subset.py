#!/usr/bin/env python3

import argparse
import copy
import json
import os
import runpy
from typing import Dict, List

import numpy as np
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

from mmpose.evaluation.functional import oks_nms, soft_oks_nms


def parse_keep_indices(value: str) -> List[int]:
    return [int(item) for item in value.split(",") if item.strip() != ""]


def load_sigmas(sigmas_file: str) -> List[float]:
    if not os.path.exists(sigmas_file):
        raise FileNotFoundError(f"Sigmas file not found: {sigmas_file}")
    data = runpy.run_path(sigmas_file)
    dataset_info = data.get("dataset_info")
    if dataset_info is None or "sigmas" not in dataset_info:
        raise KeyError(f"sigmas not found in: {sigmas_file}")
    return dataset_info["sigmas"]


def filter_keypoints_flat(keypoints: List[float], keep_indices: List[int]) -> np.ndarray:
    keypoints_arr = np.array(keypoints, dtype=np.float32).reshape(-1, 3)
    return keypoints_arr[keep_indices]


def filter_categories(categories: List[Dict], keep_indices: List[int]) -> List[Dict]:
    if not categories:
        return categories
    kept = []
    index_map = {old: new for new, old in enumerate(keep_indices)}
    for cat in categories:
        cat = copy.deepcopy(cat)
        if "keypoints" in cat:
            cat["keypoints"] = [cat["keypoints"][i] for i in keep_indices]
            cat["num_keypoints"] = len(cat["keypoints"])
        if "skeleton" in cat:
            new_skeleton = []
            for edge in cat["skeleton"]:
                src = edge[0] - 1
                dst = edge[1] - 1
                if src in index_map and dst in index_map:
                    new_skeleton.append([index_map[src] + 1, index_map[dst] + 1])
            cat["skeleton"] = new_skeleton
        kept.append(cat)
    return kept


def filter_annotations(annotations: List[Dict], keep_indices: List[int]) -> List[Dict]:
    filtered = []
    for ann in annotations:
        ann = copy.deepcopy(ann)
        if "keypoints" in ann:
            keypoints = filter_keypoints_flat(ann["keypoints"], keep_indices)
            ann["keypoints"] = keypoints.reshape(-1).tolist()
            ann["num_keypoints"] = int(np.sum(keypoints[:, 2] > 0))
        filtered.append(ann)
    return filtered


def filter_predictions(preds: List[Dict], keep_indices: List[int]) -> List[Dict]:
    filtered = []
    for pred in preds:
        pred = copy.deepcopy(pred)
        if "keypoints" in pred:
            keypoints = filter_keypoints_flat(pred["keypoints"], keep_indices)
            pred["keypoints"] = keypoints.reshape(-1).tolist()
        filtered.append(pred)
    return filtered


def compute_area(keypoints: np.ndarray) -> float:
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    return float((np.max(x) - np.min(x)) * (np.max(y) - np.min(y)))


def compute_score(
    bbox_score: float,
    keypoint_scores: np.ndarray,
    score_mode: str,
    keypoint_score_thr: float,
) -> float:
    if score_mode == "bbox":
        return float(bbox_score)
    if score_mode == "keypoint":
        return float(np.mean(keypoint_scores))
    if score_mode == "bbox_keypoint":
        valid = keypoint_scores[keypoint_scores > keypoint_score_thr]
        if valid.size == 0:
            return 0.0
        return float(bbox_score * np.mean(valid))
    raise ValueError(f"Unsupported score_mode: {score_mode}")


def apply_nms(
    preds: List[Dict],
    sigmas: np.ndarray,
    nms_mode: str,
    nms_thr: float,
) -> List[Dict]:
    if nms_mode == "none":
        return preds
    nms_fn = oks_nms if nms_mode == "oks_nms" else soft_oks_nms
    grouped = {}
    for pred in preds:
        grouped.setdefault(pred["image_id"], []).append(pred)
    filtered = []
    for image_id, items in grouped.items():
        instances = []
        for item in items:
            keypoints = np.array(item["keypoints"], dtype=np.float32).reshape(-1, 3)
            instances.append(
                {
                    "keypoints": keypoints,
                    "area": item["area"],
                    "score": item["score"],
                }
            )
        keep = nms_fn(instances, nms_thr, sigmas=sigmas)
        filtered.extend([items[i] for i in keep])
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate COCO keypoints metrics on a keypoint subset.")
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--pred-file", required=True)
    parser.add_argument("--keep-indices", required=True)
    parser.add_argument("--sigmas-file", required=True)
    parser.add_argument("--output-metrics", default="")
    parser.add_argument("--score-mode", default="bbox_keypoint")
    parser.add_argument("--pred-score-mode", default="bbox")
    parser.add_argument("--keypoint-score-thr", type=float, default=0.2)
    parser.add_argument("--nms-mode", default="oks_nms")
    parser.add_argument("--nms-thr", type=float, default=0.9)
    parser.add_argument("--use-area", action="store_true", default=True)
    args = parser.parse_args()

    keep_indices = parse_keep_indices(args.keep_indices)
    if not keep_indices:
        raise ValueError("keep-indices is empty")

    sigmas = load_sigmas(args.sigmas_file)
    if max(keep_indices) >= len(sigmas):
        raise ValueError("keep-indices contains out-of-range keypoints")

    with open(args.ann_file, "r") as f:
        gt = json.load(f)
    with open(args.pred_file, "r") as f:
        preds = json.load(f)

    if "annotations" not in gt:
        raise KeyError("Missing annotations in ann-file")

    gt = copy.deepcopy(gt)
    gt["annotations"] = filter_annotations(gt["annotations"], keep_indices)
    if "categories" in gt:
        gt["categories"] = filter_categories(gt["categories"], keep_indices)

    preds = filter_predictions(preds, keep_indices)

    tmp_dir = os.path.dirname(os.path.abspath(args.pred_file))
    gt_subset_file = os.path.join(tmp_dir, "gt_subset.json")
    pred_subset_file = os.path.join(tmp_dir, "pred_subset.json")
    with open(gt_subset_file, "w") as f:
        json.dump(gt, f)
    with open(pred_subset_file, "w") as f:
        json.dump(preds, f)

    sigma_arr = np.array(sigmas, dtype=np.float32)[keep_indices]

    rescored = []
    for pred in preds:
        keypoints = np.array(pred["keypoints"], dtype=np.float32).reshape(-1, 3)
        bbox_score = float(pred["score"]) if args.pred_score_mode == "bbox" else float(pred["score"])
        score = compute_score(
            bbox_score=bbox_score,
            keypoint_scores=keypoints[:, 2],
            score_mode=args.score_mode,
            keypoint_score_thr=args.keypoint_score_thr,
        )
        area = compute_area(keypoints)
        pred = copy.deepcopy(pred)
        pred["score"] = score
        pred["area"] = area
        rescored.append(pred)

    rescored = apply_nms(rescored, sigma_arr, args.nms_mode, args.nms_thr)

    pred_rescored_file = os.path.join(tmp_dir, "pred_subset_rescored.json")
    with open(pred_rescored_file, "w") as f:
        json.dump(rescored, f)

    coco_gt = COCO(gt_subset_file)
    coco_dt = coco_gt.loadRes(pred_rescored_file)

    coco_eval = COCOeval(coco_gt, coco_dt, "keypoints", sigma_arr, args.use_area)
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats_names = [
        "AP",
        "AP .5",
        "AP .75",
        "AP (M)",
        "AP (L)",
        "AR",
        "AR .5",
        "AR .75",
        "AR (M)",
        "AR (L)",
    ]
    metrics = {name: float(value) for name, value in zip(stats_names, coco_eval.stats)}
    if args.output_metrics:
        with open(args.output_metrics, "w") as f:
            json.dump(metrics, f, indent=2)
    print("metrics_file:", args.output_metrics or "n/a")


if __name__ == "__main__":
    main()
