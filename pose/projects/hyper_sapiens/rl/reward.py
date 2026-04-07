"""Anatomical reward functions for GRPO-style pose optimization.

Each reward component evaluates a predicted pose's anatomical plausibility
and returns a scalar reward in [0, 1]. The composite reward is a weighted
sum of individual components.

This module is analogous to a Reward Model in RLHF — it scores the
"quality" of a model output (predicted pose) without requiring ground-truth
labels at inference time.
"""

import json
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class BoneLengthReward(nn.Module):
    """Reward based on how well predicted bone lengths match population statistics.

    Computes z-scores of predicted bone lengths against reference mean/std,
    then maps to [0, 1] via sigmoid. Bones close to the population mean
    receive high reward; abnormal lengths are penalized.
    """

    def __init__(self, edges: List[Tuple[int, int]],
                 ref_means: List[float], ref_stds: List[float],
                 temperature: float = 1.0):
        super().__init__()
        self.edges = edges
        self.register_buffer('means', torch.tensor(ref_means, dtype=torch.float32))
        self.register_buffer('stds', torch.tensor(ref_stds, dtype=torch.float32))
        self.temperature = temperature

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, K, 2) or (B, G, K, 2) predicted keypoint coordinates.
        Returns:
            reward: (B,) or (B, G) scalar reward per sample/candidate.
        """
        squeeze = False
        if coords.dim() == 3:
            coords = coords.unsqueeze(1)  # (B, 1, K, 2)
            squeeze = True

        bone_lens = []
        for (u, v) in self.edges:
            diff = coords[..., u, :] - coords[..., v, :]  # (..., 2)
            length = diff.pow(2).sum(dim=-1).sqrt()  # (...)
            bone_lens.append(length)
        L = torch.stack(bone_lens, dim=-1)  # (B, G, E)

        # Scale-invariant: normalize by mean bone length per candidate
        L_mean = L.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        L_normed = L / L_mean

        ref_means = self.means.to(L.device, L.dtype)
        ref_stds = self.stds.to(L.device, L.dtype).clamp_min(1e-6)
        # Normalize reference too
        ref_mean_avg = ref_means.mean().clamp_min(1e-6)
        ref_normed = ref_means / ref_mean_avg

        z = (L_normed - ref_normed.unsqueeze(0).unsqueeze(0)) / \
            (ref_stds / ref_mean_avg).unsqueeze(0).unsqueeze(0)

        # Reward: sigmoid of negative |z|, so z=0 → reward≈0.5, larger |z| → lower
        reward_per_bone = torch.sigmoid(-z.abs() / self.temperature)  # (B, G, E)
        reward = reward_per_bone.mean(dim=-1)  # (B, G)

        return reward.squeeze(1) if squeeze else reward


class JointAngleReward(nn.Module):
    """Reward based on joint angles being within anatomically valid ranges.

    For each joint triplet (parent, joint, child), computes the angle and
    checks whether it falls within the valid ROM (range of motion) interval.
    """

    def __init__(self, triplets: List[Tuple[int, int, int]],
                 angle_min: List[float], angle_max: List[float]):
        """
        Args:
            triplets: List of (i, j, k) index tuples. Angle is at vertex j.
            angle_min: Minimum valid angle in radians for each triplet.
            angle_max: Maximum valid angle in radians for each triplet.
        """
        super().__init__()
        self.triplets = triplets
        self.register_buffer('a_min', torch.tensor(angle_min, dtype=torch.float32))
        self.register_buffer('a_max', torch.tensor(angle_max, dtype=torch.float32))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if coords.dim() == 3:
            coords = coords.unsqueeze(1)
            squeeze = True

        eps = 1e-6
        angles = []
        for (i, j, k) in self.triplets:
            v1 = coords[..., i, :] - coords[..., j, :]
            v2 = coords[..., k, :] - coords[..., j, :]
            cos = (v1 * v2).sum(dim=-1) / \
                  (v1.norm(dim=-1) * v2.norm(dim=-1) + eps)
            angle = torch.acos(cos.clamp(-1 + eps, 1 - eps))  # (B, G)
            angles.append(angle)
        A = torch.stack(angles, dim=-1)  # (B, G, T)

        a_min = self.a_min.to(A.device, A.dtype)
        a_max = self.a_max.to(A.device, A.dtype)

        # Reward = 1.0 if within [min, max], decays outside
        below = torch.relu(a_min - A)
        above = torch.relu(A - a_max)
        deviation = below + above  # (B, G, T)
        reward_per_joint = torch.exp(-deviation * 3.0)  # exponential decay
        reward = reward_per_joint.mean(dim=-1)  # (B, G)

        return reward.squeeze(1) if squeeze else reward


class SymmetryReward(nn.Module):
    """Reward for bilateral symmetry of corresponding limb pairs.

    Penalizes large differences in length between left-right limb pairs
    (e.g., left upper arm vs right upper arm).
    """

    def __init__(self, symmetric_pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]]):
        """
        Args:
            symmetric_pairs: List of ((u1,v1), (u2,v2)) pairs where
                (u1,v1) is the left limb and (u2,v2) is the right limb.
        """
        super().__init__()
        self.pairs = symmetric_pairs

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if coords.dim() == 3:
            coords = coords.unsqueeze(1)
            squeeze = True

        ratios = []
        for ((u1, v1), (u2, v2)) in self.pairs:
            l1 = (coords[..., u1, :] - coords[..., v1, :]).pow(2).sum(-1).sqrt()
            l2 = (coords[..., u2, :] - coords[..., v2, :]).pow(2).sum(-1).sqrt()
            ratio = (l1 / l2.clamp_min(1e-6)).clamp(0.5, 2.0)
            # Perfect symmetry → ratio=1 → reward=1
            ratios.append(1.0 - (ratio - 1.0).abs())
        R = torch.stack(ratios, dim=-1)  # (B, G, P)
        reward = R.mean(dim=-1)

        return reward.squeeze(1) if squeeze else reward


class PoseRewardFunction(nn.Module):
    """Composite reward function combining multiple anatomical criteria.

    This is the equivalent of a Reward Model in RLHF: given a predicted
    pose, it produces a scalar quality score without requiring ground truth.

    Args:
        stats_path: Path to skeleton_stats.json containing bone length
            statistics, ROM ranges, and symmetry pair definitions.
        weights: Dict of reward component weights. Default balances
            bone_length, joint_angle, and symmetry equally.
    """

    def __init__(self, stats_path: str,
                 weights: Optional[Dict[str, float]] = None):
        super().__init__()
        with open(stats_path) as f:
            stats = json.load(f)

        w = weights or {'bone': 0.4, 'angle': 0.3, 'symmetry': 0.3}
        self.w_bone = w.get('bone', 0.4)
        self.w_angle = w.get('angle', 0.3)
        self.w_sym = w.get('symmetry', 0.3)

        self.bone_reward = BoneLengthReward(
            edges=stats['edges'],
            ref_means=stats['bone_length_means'],
            ref_stds=stats['bone_length_stds'],
        )
        self.angle_reward = JointAngleReward(
            triplets=stats['angle_triplets'],
            angle_min=stats['angle_min'],
            angle_max=stats['angle_max'],
        )
        self.sym_reward = SymmetryReward(
            symmetric_pairs=stats['symmetric_pairs'],
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, K, 2) or (B, G, K, 2) keypoint coordinates.
        Returns:
            reward: (B,) or (B, G) composite reward in [0, 1].
        """
        r_bone = self.bone_reward(coords)
        r_angle = self.angle_reward(coords)
        r_sym = self.sym_reward(coords)

        return self.w_bone * r_bone + self.w_angle * r_angle + self.w_sym * r_sym
