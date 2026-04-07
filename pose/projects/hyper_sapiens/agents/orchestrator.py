"""Multi-agent orchestrator for pose quality control.

Coordinates three specialized agents (Pose Detection, VLM Review,
Structural Validation) to evaluate and filter pose predictions.
Implements a voting/aggregation mechanism for final quality decisions.

Architecture follows the Agent + Tool-Use pattern:
  - Each agent has a defined role and tool set
  - Orchestrator dispatches tasks and aggregates results
  - Final decision uses configurable thresholds

This is analogous to a multi-agent LLM system where different agents
specialize in different aspects of quality assessment.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .tools import (PoseDetectionTool, StructuralValidationTool,
                    ToolResult, VLMAssessmentTool)

logger = logging.getLogger(__name__)


@dataclass
class QualityVerdict:
    """Final quality assessment for a single image."""
    image_path: str
    decision: str  # 'accept', 'reject', 'review'
    confidence: float  # [0, 1]
    pose_result: Optional[ToolResult] = None
    struct_result: Optional[ToolResult] = None
    vlm_result: Optional[ToolResult] = None
    reason: str = ''

    def to_dict(self) -> dict:
        d = {
            'image_path': self.image_path,
            'decision': self.decision,
            'confidence': self.confidence,
            'reason': self.reason,
        }
        if self.pose_result and self.pose_result.success:
            d['keypoints'] = self.pose_result.data.get('keypoints', [])
            d['keypoint_scores'] = self.pose_result.data.get('scores', [])
        if self.struct_result and self.struct_result.success:
            d['plausibility_score'] = self.struct_result.data.get(
                'plausibility_score', 0)
            d['structural_issues'] = self.struct_result.data.get('issues', [])
        if self.vlm_result and self.vlm_result.success:
            d['vlm_quality_score'] = self.vlm_result.data.get(
                'quality_score', 0)
            d['vlm_issues'] = self.vlm_result.data.get('issues', '')
        return d


class QualityOrchestrator:
    """Orchestrates multi-agent quality assessment pipeline.

    Execution flow:
    1. Pose Agent: detect keypoints (if not already provided)
    2. Structural Agent: validate bone lengths, angles, symmetry
    3. VLM Agent (optional): multimodal quality assessment
    4. Aggregation: combine scores → accept / reject / review

    Args:
        stats_path: Path to skeleton_stats.json for structural validation.
        pose_config: Config path for Sapiens model (optional).
        pose_checkpoint: Checkpoint path for Sapiens model (optional).
        vlm_model: VLM model name (e.g., 'Qwen/Qwen2-VL-2B-Instruct').
        use_vlm: Whether to enable VLM assessment.
        accept_threshold: Minimum score to auto-accept.
        reject_threshold: Maximum score to auto-reject.
        device: CUDA device string.
    """

    def __init__(self,
                 stats_path: str,
                 pose_config: Optional[str] = None,
                 pose_checkpoint: Optional[str] = None,
                 vlm_model: str = 'Qwen/Qwen2-VL-2B-Instruct',
                 use_vlm: bool = True,
                 accept_threshold: float = 0.75,
                 reject_threshold: float = 0.4,
                 device: str = 'cuda:0'):
        self.accept_thr = accept_threshold
        self.reject_thr = reject_threshold
        self.use_vlm = use_vlm

        # Initialize tools
        self.pose_tool = PoseDetectionTool(
            config_path=pose_config,
            checkpoint_path=pose_checkpoint,
            device=device,
        )
        self.struct_tool = StructuralValidationTool(stats_path)
        self.vlm_tool = VLMAssessmentTool(
            model_name=vlm_model, device=device,
        ) if use_vlm else None

    def assess_single(self, image_path: str,
                      keypoints: Optional[List[List[float]]] = None,
                      scores: Optional[List[float]] = None,
                      bbox: Optional[List[float]] = None) -> QualityVerdict:
        """Run full quality assessment on a single image.

        Args:
            image_path: Path to the image.
            keypoints: Pre-computed keypoints (skip pose detection if provided).
            scores: Pre-computed keypoint confidence scores.
            bbox: Bounding box for pose detection.
        """
        # Step 1: Pose Detection (if not provided)
        pose_result = None
        if keypoints is None:
            pose_result = self.pose_tool(image_path, bbox)
            if not pose_result.success:
                return QualityVerdict(
                    image_path=image_path, decision='reject',
                    confidence=1.0, pose_result=pose_result,
                    reason=f'Pose detection failed: {pose_result.error}',
                )
            keypoints = pose_result.data['keypoints']
            scores = pose_result.data.get('scores')

        # Step 2: Structural Validation
        struct_result = self.struct_tool(keypoints, scores)
        struct_score = 0.0
        if struct_result.success:
            struct_score = struct_result.data.get('plausibility_score', 0)

        # Step 3: VLM Assessment (optional)
        vlm_result = None
        vlm_score = None
        if self.use_vlm and self.vlm_tool is not None:
            vlm_result = self.vlm_tool(image_path, keypoints, scores)
            if vlm_result.success:
                vlm_score = vlm_result.data.get('quality_score', 0.5)

        # Step 4: Aggregation
        if vlm_score is not None:
            # Weighted combination: struct 0.5, vlm 0.5
            combined_score = 0.5 * struct_score + 0.5 * vlm_score
        else:
            combined_score = struct_score

        # Decision
        reasons = []
        if struct_result.success:
            n_issues = struct_result.data.get('num_issues', 0)
            if n_issues > 0:
                reasons.append(f'{n_issues} structural issues')
        if vlm_result and vlm_result.success:
            vlm_issues = vlm_result.data.get('issues', '')
            if vlm_issues:
                reasons.append(f'VLM: {vlm_issues}')

        if combined_score >= self.accept_thr:
            decision = 'accept'
        elif combined_score < self.reject_thr:
            decision = 'reject'
        else:
            decision = 'review'

        return QualityVerdict(
            image_path=image_path,
            decision=decision,
            confidence=combined_score,
            pose_result=pose_result,
            struct_result=struct_result,
            vlm_result=vlm_result,
            reason='; '.join(reasons) if reasons else 'OK',
        )

    def assess_batch(self, image_paths: List[str],
                     annotations: Optional[List[dict]] = None,
                     progress_callback=None) -> List[QualityVerdict]:
        """Run quality assessment on a batch of images.

        Args:
            image_paths: List of image file paths.
            annotations: Optional list of dicts with 'keypoints' and 'scores'
                keys (pre-computed predictions).
            progress_callback: Optional callable(current, total) for progress.

        Returns:
            List of QualityVerdict for each image.
        """
        results = []
        total = len(image_paths)

        for idx, img_path in enumerate(image_paths):
            kps = None
            scr = None
            if annotations and idx < len(annotations):
                ann = annotations[idx]
                kps = ann.get('keypoints')
                scr = ann.get('scores')

            verdict = self.assess_single(img_path, keypoints=kps, scores=scr)
            results.append(verdict)

            if progress_callback:
                progress_callback(idx + 1, total)

            if (idx + 1) % 50 == 0:
                n_accept = sum(1 for r in results if r.decision == 'accept')
                n_reject = sum(1 for r in results if r.decision == 'reject')
                n_review = sum(1 for r in results if r.decision == 'review')
                logger.info(
                    f'[{idx+1}/{total}] accept={n_accept} reject={n_reject} '
                    f'review={n_review}'
                )

        return results

    @staticmethod
    def save_results(results: List[QualityVerdict], output_path: str):
        """Save assessment results to JSON."""
        data = {
            'num_total': len(results),
            'num_accept': sum(1 for r in results if r.decision == 'accept'),
            'num_reject': sum(1 for r in results if r.decision == 'reject'),
            'num_review': sum(1 for r in results if r.decision == 'review'),
            'results': [r.to_dict() for r in results],
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f'Saved {len(results)} results to {output_path}')

    @staticmethod
    def export_filtered_coco(results: List[QualityVerdict],
                             original_ann_path: str,
                             output_path: str,
                             decisions: tuple = ('accept',)):
        """Export filtered annotations as a new COCO JSON.

        Only keeps annotations for images that received one of the
        specified decisions.
        """
        accepted_images = {
            r.image_path for r in results if r.decision in decisions
        }

        with open(original_ann_path) as f:
            coco = json.load(f)

        # Build image filename → id mapping
        img_id_by_file = {}
        for img in coco.get('images', []):
            img_id_by_file[img['file_name']] = img['id']

        accepted_ids = set()
        for path in accepted_images:
            fname = Path(path).name
            if fname in img_id_by_file:
                accepted_ids.add(img_id_by_file[fname])

        filtered_images = [
            img for img in coco['images'] if img['id'] in accepted_ids
        ]
        filtered_anns = [
            ann for ann in coco.get('annotations', [])
            if ann['image_id'] in accepted_ids
        ]

        filtered_coco = {**coco, 'images': filtered_images, 'annotations': filtered_anns}

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(filtered_coco, f, ensure_ascii=False)
        logger.info(
            f'Exported {len(filtered_images)} images, '
            f'{len(filtered_anns)} annotations to {output_path}'
        )
