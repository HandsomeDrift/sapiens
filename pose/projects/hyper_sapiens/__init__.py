# Hyper-Sapiens: Parameter-efficient domain adaptation of Sapiens
# vision foundation model for infant pose estimation.
#
# Key modules:
#   - LoRA fine-tuning for Sapiens ViT backbone
#   - Multi-head geometric decoder (Pose + Depth + Normal)
#   - Diffusion-driven prior (DDP) with SDS loss
#   - VLM-based pose quality assessment

from .engine import *  # noqa: F401,F403
