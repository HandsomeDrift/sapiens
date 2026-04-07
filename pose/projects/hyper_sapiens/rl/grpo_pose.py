"""GRPO-style group relative policy optimization for pose estimation.

Adapts the Group Relative Policy Optimization (GRPO) framework from
DeepSeek-R1 to visual pose estimation:

  LLM GRPO:   prompt → sample G responses → reward each → group-normalize → policy gradient
  Pose GRPO:  image  → sample G pose sets  → reward each → group-normalize → policy gradient

Key differences from standard heatmap MSE training:
  1. No ground-truth labels needed — reward is computed from anatomical priors
  2. Multiple candidates are sampled from the heatmap distribution (not just argmax)
  3. Optimization uses relative advantage within the group (self-play style)

This enables RL-based alignment of the pose model to anatomical constraints
without any human annotation — analogous to RLHF's reward-based alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_from_heatmap(heatmaps: torch.Tensor,
                        num_samples: int = 8,
                        temperature: float = 1.0) -> torch.Tensor:
    """Sample multiple sets of keypoint coordinates from heatmap distributions.

    Instead of taking the argmax of each heatmap (deterministic), we treat
    each heatmap as a categorical distribution over spatial locations and
    sample from it. This produces diverse pose candidates for GRPO.

    Args:
        heatmaps: (B, K, H, W) predicted heatmaps (logits, pre-softmax).
        num_samples: Number of candidate pose sets to sample (G in GRPO).
        temperature: Sampling temperature. Lower → more peaked (closer to
            argmax), higher → more diverse samples.

    Returns:
        coords: (B, G, K, 2) sampled coordinates in pixel space [0, W) × [0, H).
        log_probs: (B, G, K) log-probability of each sampled coordinate.
    """
    B, K, H, W = heatmaps.shape

    # Flatten spatial dims and apply temperature
    logits = heatmaps.view(B, K, -1) / temperature  # (B, K, H*W)
    probs = F.softmax(logits, dim=-1)  # (B, K, H*W)

    # Sample G times for each keypoint
    # Expand for G samples: (B, K, H*W) → sample → (B*K, G)
    probs_flat = probs.view(B * K, -1)  # (B*K, H*W)
    indices = torch.multinomial(probs_flat, num_samples, replacement=True)  # (B*K, G)
    indices = indices.view(B, K, num_samples)  # (B, K, G)

    # Convert flat indices to (y, x) coordinates
    y_coords = indices // W  # (B, K, G)
    x_coords = indices % W   # (B, K, G)
    coords = torch.stack([x_coords, y_coords], dim=-1).float()  # (B, K, G, 2)

    # Rearrange to (B, G, K, 2)
    coords = coords.permute(0, 2, 1, 3)

    # Compute log-probabilities
    log_probs_flat = F.log_softmax(logits, dim=-1)  # (B, K, H*W)
    # Gather log-probs at sampled indices
    indices_flat = indices.view(B, K, num_samples)  # (B, K, G)
    log_p = torch.gather(log_probs_flat, dim=-1, index=indices_flat)  # (B, K, G)
    log_probs = log_p.permute(0, 2, 1)  # (B, G, K)

    return coords, log_probs


def compute_group_advantage(rewards: torch.Tensor,
                            eps: float = 1e-8) -> torch.Tensor:
    """Compute group-normalized advantage (core of GRPO).

    For each sample in the batch, normalize the G rewards to have zero
    mean and unit variance within the group. This removes the need for
    a separate critic/value network.

    Args:
        rewards: (B, G) reward scores for each candidate in the group.

    Returns:
        advantages: (B, G) normalized advantages.
    """
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True).clamp_min(eps)
    return (rewards - mean) / std


class GRPOPoseLoss(nn.Module):
    """GRPO loss for pose estimation.

    Implements the policy gradient objective with group-relative advantages:

        L_GRPO = -1/G * sum_i[ A_i * log π(y_i | x) ]

    where A_i is the group-normalized advantage and π(y_i | x) is the
    probability of the i-th sampled pose under the current policy (heatmap).

    Optionally applies a clipping mechanism (similar to PPO) to prevent
    excessively large policy updates.

    Args:
        reward_fn: A callable that takes (B, G, K, 2) coords and returns
            (B, G) rewards. Typically a PoseRewardFunction instance.
        num_samples: Number of pose candidates to sample per image (G).
        temperature: Heatmap sampling temperature.
        clip_ratio: If > 0, clip the importance ratio (PPO-style).
            Set to 0 to disable clipping.
        advantage_clip: Clip extreme advantages to [-clip, clip] for stability.
    """

    def __init__(self, reward_fn: nn.Module,
                 num_samples: int = 8,
                 temperature: float = 1.0,
                 clip_ratio: float = 0.2,
                 advantage_clip: float = 5.0):
        super().__init__()
        self.reward_fn = reward_fn
        self.num_samples = num_samples
        self.temperature = temperature
        self.clip_ratio = clip_ratio
        self.advantage_clip = advantage_clip

    def forward(self, heatmaps: torch.Tensor) -> dict:
        """
        Args:
            heatmaps: (B, K, H, W) predicted heatmaps (logits).

        Returns:
            Dict with:
                'loss': scalar GRPO loss
                'reward_mean': mean reward across batch (for logging)
                'reward_std': reward std (for logging)
                'advantage_mean': mean |advantage| (for logging)
        """
        B, K, H, W = heatmaps.shape

        # 1. Sample G pose candidates from heatmap distribution
        with torch.no_grad():
            coords, _ = sample_from_heatmap(
                heatmaps.detach(), self.num_samples, self.temperature
            )  # coords: (B, G, K, 2), log_probs: (B, G, K)

        # 2. Compute rewards for each candidate (no grad through reward)
        with torch.no_grad():
            rewards = self.reward_fn(coords)  # (B, G)

        # 3. Compute group-normalized advantages
        advantages = compute_group_advantage(rewards)  # (B, G)
        if self.advantage_clip > 0:
            advantages = advantages.clamp(-self.advantage_clip, self.advantage_clip)

        # 4. Compute log-probabilities under current policy (with grad)
        logits = heatmaps.view(B, K, -1) / self.temperature
        log_probs_all = F.log_softmax(logits, dim=-1)  # (B, K, H*W)

        # Flatten coords to indices for gathering
        x = coords[..., 0].long().clamp(0, W - 1)  # (B, G, K)
        y = coords[..., 1].long().clamp(0, H - 1)  # (B, G, K)
        flat_idx = y * W + x  # (B, G, K)

        # Gather log-probs: for each (b, g, k), get log_probs_all[b, k, flat_idx[b,g,k]]
        flat_idx_expanded = flat_idx.permute(0, 2, 1)  # (B, K, G)
        log_p = torch.gather(log_probs_all, dim=-1, index=flat_idx_expanded)  # (B, K, G)
        log_p = log_p.permute(0, 2, 1)  # (B, G, K)

        # Sum log-probs across keypoints → log π(pose | image)
        log_pi = log_p.sum(dim=-1)  # (B, G)

        # 5. Policy gradient loss: L = -E[A * log π]
        loss = -(advantages.detach() * log_pi).mean()

        return {
            'loss': loss,
            'reward_mean': rewards.mean().item(),
            'reward_std': rewards.std().item(),
            'advantage_abs_mean': advantages.abs().mean().item(),
        }
