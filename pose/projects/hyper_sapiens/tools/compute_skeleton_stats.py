"""Compute skeleton statistics from COCO-format keypoint annotations.

Outputs a skeleton_stats.json file containing:
  - bone_length_means / bone_length_stds: per-edge bone length statistics
  - angle_min / angle_max: per-triplet joint angle ROM (5th/95th percentile)
  - edges, angle_triplets, symmetric_pairs: skeleton topology

This file is required by PoseRewardFunction (GRPO) and geometric consistency losses.

Usage:
    python compute_skeleton_stats.py \
        --ann /path/to/person_keypoints_train.json \
        --out stats/skeleton_stats.json \
        [--keypoint-type coco17]
"""

import argparse
import json
import math

import numpy as np


# COCO 17-keypoint skeleton definition
COCO17_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # head
    (5, 6),                                 # shoulders
    (5, 7), (7, 9),                         # left arm
    (6, 8), (8, 10),                        # right arm
    (5, 11), (6, 12),                       # torso
    (11, 12),                               # hips
    (11, 13), (13, 15),                     # left leg
    (12, 14), (14, 16),                     # right leg
]

COCO17_ANGLE_TRIPLETS = [
    (5, 7, 9),    # left shoulder-elbow-wrist
    (6, 8, 10),   # right shoulder-elbow-wrist
    (11, 13, 15), # left hip-knee-ankle
    (12, 14, 16), # right hip-knee-ankle
    (7, 5, 11),   # left elbow-shoulder-hip
    (8, 6, 12),   # right elbow-shoulder-hip
    (5, 11, 13),  # left shoulder-hip-knee
    (6, 12, 14),  # right shoulder-hip-knee
]

COCO17_SYMMETRIC_PAIRS = [
    ((5, 7), (6, 8)),     # upper arms
    ((7, 9), (8, 10)),    # forearms
    ((11, 13), (12, 14)), # upper legs
    ((13, 15), (14, 16)), # lower legs
    ((5, 11), (6, 12)),   # torso sides
]

COCO17_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]


def compute_stats(ann_path: str, keypoint_type: str = 'coco17') -> dict:
    assert keypoint_type == 'coco17', f'Only coco17 supported, got {keypoint_type}'

    edges = COCO17_EDGES
    triplets = COCO17_ANGLE_TRIPLETS
    sym_pairs = COCO17_SYMMETRIC_PAIRS

    with open(ann_path) as f:
        data = json.load(f)

    # Collect all valid keypoint sets
    all_bone_lengths = {i: [] for i in range(len(edges))}
    all_angles = {i: [] for i in range(len(triplets))}

    n_valid = 0
    for ann in data['annotations']:
        kps = ann.get('keypoints', [])
        if len(kps) < 17 * 3:
            continue

        # Parse keypoints: [x1, y1, v1, x2, y2, v2, ...]
        coords = np.array(kps).reshape(-1, 3)
        xy = coords[:, :2]
        vis = coords[:, 2]

        # Compute bone lengths (only if both endpoints are visible)
        for idx, (u, v) in enumerate(edges):
            if vis[u] > 0 and vis[v] > 0:
                length = np.linalg.norm(xy[u] - xy[v])
                if length > 1.0:  # filter degenerate
                    all_bone_lengths[idx].append(length)

        # Compute joint angles (only if all three points visible)
        for idx, (i, j, k) in enumerate(triplets):
            if vis[i] > 0 and vis[j] > 0 and vis[k] > 0:
                v1 = xy[i] - xy[j]
                v2 = xy[k] - xy[j]
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 > 1e-6 and n2 > 1e-6:
                    cos_val = np.dot(v1, v2) / (n1 * n2)
                    cos_val = np.clip(cos_val, -1.0, 1.0)
                    angle = math.acos(cos_val)
                    all_angles[idx].append(angle)

        n_valid += 1

    print(f'Processed {n_valid} valid annotations from {len(data["annotations"])} total')

    # Compute statistics
    bone_means = []
    bone_stds = []
    for idx in range(len(edges)):
        vals = all_bone_lengths[idx]
        if len(vals) < 10:
            print(f'  WARNING: edge {edges[idx]} has only {len(vals)} samples')
            bone_means.append(50.0)  # fallback
            bone_stds.append(20.0)
        else:
            arr = np.array(vals)
            bone_means.append(float(np.mean(arr)))
            bone_stds.append(float(np.std(arr)))

    angle_mins = []
    angle_maxs = []
    for idx in range(len(triplets)):
        vals = all_angles[idx]
        if len(vals) < 10:
            print(f'  WARNING: triplet {triplets[idx]} has only {len(vals)} samples')
            angle_mins.append(0.3)   # ~17 degrees
            angle_maxs.append(2.8)   # ~160 degrees
        else:
            arr = np.array(vals)
            angle_mins.append(float(np.percentile(arr, 5)))
            angle_maxs.append(float(np.percentile(arr, 95)))

    stats = {
        'keypoint_type': keypoint_type,
        'keypoint_names': COCO17_KEYPOINT_NAMES,
        'num_keypoints': 17,
        'edges': [list(e) for e in edges],
        'bone_length_means': bone_means,
        'bone_length_stds': bone_stds,
        'angle_triplets': [list(t) for t in triplets],
        'angle_min': angle_mins,
        'angle_max': angle_maxs,
        'symmetric_pairs': [[[u1, v1], [u2, v2]] for ((u1, v1), (u2, v2)) in sym_pairs],
        'num_annotations_used': n_valid,
    }

    # Print summary
    print(f'\nBone lengths (mean ± std):')
    for idx, (u, v) in enumerate(edges):
        name = f'{COCO17_KEYPOINT_NAMES[u]}-{COCO17_KEYPOINT_NAMES[v]}'
        print(f'  {name:40s} {bone_means[idx]:7.1f} ± {bone_stds[idx]:5.1f}  '
              f'(n={len(all_bone_lengths[idx])})')

    print(f'\nJoint angle ROM (5th-95th percentile, degrees):')
    for idx, (i, j, k) in enumerate(triplets):
        name = f'{COCO17_KEYPOINT_NAMES[i]}-{COCO17_KEYPOINT_NAMES[j]}-{COCO17_KEYPOINT_NAMES[k]}'
        deg_min = math.degrees(angle_mins[idx])
        deg_max = math.degrees(angle_maxs[idx])
        print(f'  {name:50s} [{deg_min:5.1f}°, {deg_max:5.1f}°]  '
              f'(n={len(all_angles[idx])})')

    return stats


def main():
    parser = argparse.ArgumentParser(description='Compute skeleton statistics')
    parser.add_argument('--ann', required=True, help='COCO keypoints JSON path')
    parser.add_argument('--out', required=True, help='Output skeleton_stats.json path')
    parser.add_argument('--keypoint-type', default='coco17')
    args = parser.parse_args()

    stats = compute_stats(args.ann, args.keypoint_type)

    import os
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'\nSaved to {args.out}')


if __name__ == '__main__':
    main()
