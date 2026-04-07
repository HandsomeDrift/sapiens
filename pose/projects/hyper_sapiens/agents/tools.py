"""Tool definitions for the multi-agent pose quality control pipeline.

Each tool is a self-contained callable with a standardized interface:
  - Defined as a class with __call__ method
  - Has a `schema` property returning OpenAI function-calling compatible JSON
  - Returns structured dict results

This follows the function-calling / tool-use pattern used in LLM Agent
systems (e.g., OpenAI function calling, Anthropic tool use), but applied
to a vision task pipeline.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Tool result container
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Standardized tool execution result."""
    tool_name: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Tool 1: Pose Detection
# ---------------------------------------------------------------------------

class PoseDetectionTool:
    """Runs Sapiens pose estimation on an image.

    Wraps the Sapiens model inference into a tool that can be called
    by an Agent. Handles model loading, preprocessing, and postprocessing.
    """

    schema = {
        'name': 'pose_detection',
        'description': 'Detect human pose keypoints in an image using Sapiens model.',
        'parameters': {
            'type': 'object',
            'properties': {
                'image_path': {
                    'type': 'string',
                    'description': 'Path to the input image file.',
                },
                'bbox': {
                    'type': 'array',
                    'items': {'type': 'number'},
                    'description': 'Bounding box [x, y, w, h]. If omitted, uses full image.',
                },
            },
            'required': ['image_path'],
        },
    }

    def __init__(self, config_path: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 device: str = 'cuda:0'):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._model = None

    def _lazy_load(self):
        """Lazy-load model on first call to avoid GPU memory waste."""
        if self._model is not None:
            return
        if self.config_path is None:
            # Lightweight mode: no model, return mock results
            self._model = 'mock'
            return

        from mmpose.apis import init_model
        self._model = init_model(
            self.config_path, self.checkpoint_path, device=self.device
        )

    def __call__(self, image_path: str,
                 bbox: Optional[List[float]] = None) -> ToolResult:
        try:
            self._lazy_load()

            if self._model == 'mock':
                return ToolResult(
                    tool_name='pose_detection',
                    success=False,
                    error='Model not configured. Provide config_path and checkpoint_path.',
                )

            import cv2
            from mmpose.apis import inference_topdown

            img = cv2.imread(image_path)
            if img is None:
                return ToolResult(
                    tool_name='pose_detection', success=False,
                    error=f'Cannot read image: {image_path}',
                )

            if bbox is None:
                h, w = img.shape[:2]
                bbox = [0, 0, w, h]

            results = inference_topdown(self._model, image_path, [bbox])

            if not results:
                return ToolResult(
                    tool_name='pose_detection', success=True,
                    data={'keypoints': [], 'scores': [], 'num_persons': 0},
                )

            pred = results[0].pred_instances
            keypoints = pred.keypoints[0].tolist()
            scores = pred.keypoint_scores[0].tolist()

            return ToolResult(
                tool_name='pose_detection', success=True,
                data={
                    'keypoints': keypoints,
                    'scores': scores,
                    'num_persons': len(results),
                    'mean_confidence': float(np.mean(scores)),
                },
            )
        except Exception as e:
            return ToolResult(
                tool_name='pose_detection', success=False, error=str(e),
            )


# ---------------------------------------------------------------------------
# Tool 2: Structural Validation
# ---------------------------------------------------------------------------

class StructuralValidationTool:
    """Validates anatomical plausibility of predicted keypoints.

    Checks bone length ratios, joint angles, and bilateral symmetry
    against population statistics. Does not require GPU.
    """

    schema = {
        'name': 'structural_validation',
        'description': 'Validate anatomical plausibility of predicted pose keypoints.',
        'parameters': {
            'type': 'object',
            'properties': {
                'keypoints': {
                    'type': 'array',
                    'description': 'List of [x, y] coordinates for each keypoint.',
                },
                'scores': {
                    'type': 'array',
                    'description': 'Confidence score for each keypoint.',
                },
            },
            'required': ['keypoints'],
        },
    }

    def __init__(self, stats_path: str):
        with open(stats_path) as f:
            self.stats = json.load(f)
        self.edges = [tuple(e) for e in self.stats['edges']]
        self.triplets = [tuple(t) for t in self.stats['angle_triplets']]
        self.sym_pairs = [
            (tuple(p[0]), tuple(p[1])) for p in self.stats['symmetric_pairs']
        ]
        self.bone_means = np.array(self.stats['bone_length_means'])
        self.bone_stds = np.array(self.stats['bone_length_stds'])
        self.angle_min = np.array(self.stats['angle_min'])
        self.angle_max = np.array(self.stats['angle_max'])

    def __call__(self, keypoints: List[List[float]],
                 scores: Optional[List[float]] = None) -> ToolResult:
        try:
            kps = np.array(keypoints)
            if kps.shape[0] < 17 or kps.shape[1] < 2:
                return ToolResult(
                    tool_name='structural_validation', success=False,
                    error=f'Expected 17×2 keypoints, got {kps.shape}',
                )

            issues = []
            bone_details = []
            angle_details = []

            # --- Bone length check ---
            bone_lengths = []
            for idx, (u, v) in enumerate(self.edges):
                length = float(np.linalg.norm(kps[u] - kps[v]))
                bone_lengths.append(length)

                if self.bone_stds[idx] > 1e-6:
                    z = abs(length - self.bone_means[idx]) / self.bone_stds[idx]
                else:
                    z = 0.0

                status = 'ok' if z < 3.0 else ('warning' if z < 5.0 else 'abnormal')
                bone_details.append({
                    'edge': [u, v],
                    'length': round(length, 1),
                    'z_score': round(z, 2),
                    'status': status,
                })
                if status == 'abnormal':
                    names = self.stats['keypoint_names']
                    issues.append(
                        f'Bone {names[u]}-{names[v]}: length={length:.0f}, '
                        f'z={z:.1f} (expected {self.bone_means[idx]:.0f}±{self.bone_stds[idx]:.0f})'
                    )

            # --- Joint angle check ---
            for idx, (i, j, k) in enumerate(self.triplets):
                v1 = kps[i] - kps[j]
                v2 = kps[k] - kps[j]
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 < 1e-6 or n2 < 1e-6:
                    angle_details.append({
                        'triplet': [i, j, k], 'angle_deg': 0,
                        'status': 'degenerate',
                    })
                    issues.append(f'Degenerate angle at joint {j}')
                    continue

                cos_val = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                angle = math.acos(cos_val)
                angle_deg = math.degrees(angle)

                in_range = self.angle_min[idx] <= angle <= self.angle_max[idx]
                status = 'ok' if in_range else 'out_of_range'
                angle_details.append({
                    'triplet': [i, j, k],
                    'angle_deg': round(angle_deg, 1),
                    'range_deg': [
                        round(math.degrees(self.angle_min[idx]), 1),
                        round(math.degrees(self.angle_max[idx]), 1),
                    ],
                    'status': status,
                })
                if not in_range:
                    names = self.stats['keypoint_names']
                    issues.append(
                        f'Joint angle {names[i]}-{names[j]}-{names[k]}: '
                        f'{angle_deg:.0f}° (valid: '
                        f'{math.degrees(self.angle_min[idx]):.0f}°-'
                        f'{math.degrees(self.angle_max[idx]):.0f}°)'
                    )

            # --- Symmetry check ---
            sym_details = []
            for (l_edge, r_edge) in self.sym_pairs:
                l_len = float(np.linalg.norm(kps[l_edge[0]] - kps[l_edge[1]]))
                r_len = float(np.linalg.norm(kps[r_edge[0]] - kps[r_edge[1]]))
                if r_len > 1e-6:
                    ratio = l_len / r_len
                else:
                    ratio = float('inf')
                status = 'ok' if 0.6 < ratio < 1.67 else 'asymmetric'
                sym_details.append({
                    'left': list(l_edge), 'right': list(r_edge),
                    'ratio': round(ratio, 3), 'status': status,
                })
                if status == 'asymmetric':
                    issues.append(
                        f'Asymmetry: limb {l_edge} vs {r_edge}, ratio={ratio:.2f}'
                    )

            # --- Overall score ---
            n_bones = len(bone_details)
            n_angles = len(angle_details)
            ok_bones = sum(1 for b in bone_details if b['status'] == 'ok')
            ok_angles = sum(1 for a in angle_details if a['status'] == 'ok')
            ok_sym = sum(1 for s in sym_details if s['status'] == 'ok')
            total_checks = n_bones + n_angles + len(sym_details)
            ok_checks = ok_bones + ok_angles + ok_sym
            plausibility_score = ok_checks / max(total_checks, 1)

            return ToolResult(
                tool_name='structural_validation', success=True,
                data={
                    'plausibility_score': round(plausibility_score, 3),
                    'num_issues': len(issues),
                    'issues': issues,
                    'bone_check': bone_details,
                    'angle_check': angle_details,
                    'symmetry_check': sym_details,
                },
            )
        except Exception as e:
            return ToolResult(
                tool_name='structural_validation', success=False, error=str(e),
            )


# ---------------------------------------------------------------------------
# Tool 3: VLM Assessment
# ---------------------------------------------------------------------------

class VLMAssessmentTool:
    """Uses a Vision-Language Model to assess pose quality.

    Renders predicted keypoints onto the image and asks a VLM to evaluate
    anatomical plausibility, joint visibility, and overall quality.
    Acts as a multimodal reward model.
    """

    schema = {
        'name': 'vlm_assessment',
        'description': 'Use a Vision-Language Model to assess predicted pose quality.',
        'parameters': {
            'type': 'object',
            'properties': {
                'image_path': {
                    'type': 'string',
                    'description': 'Path to the original image.',
                },
                'keypoints': {
                    'type': 'array',
                    'description': 'Predicted [x, y] coordinates for each keypoint.',
                },
                'scores': {
                    'type': 'array',
                    'description': 'Confidence score per keypoint.',
                },
            },
            'required': ['image_path', 'keypoints'],
        },
    }

    PROMPT_TEMPLATE = """Analyze this image with a skeleton overlay showing predicted joint positions for a person (possibly an infant). Evaluate:

1. Anatomical plausibility (0-10): Are the joint positions anatomically reasonable? Are limbs properly connected? Any impossible poses?
2. Joint visibility (0-10): How many joints appear to be correctly visible vs occluded or off-body?
3. Overall quality (0-10): Overall accuracy of the pose estimation.

Respond ONLY with a JSON object:
{{"anatomical": <score>, "visibility": <score>, "overall": <score>, "issues": "<brief description of problems if any>"}}"""

    SKELETON_EDGES = [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6),
        (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ]

    def __init__(self, model_name: str = 'Qwen/Qwen2-VL-2B-Instruct',
                 device: str = 'cuda:0'):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None

    def _lazy_load(self):
        if self._model is not None:
            return
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype='auto', device_map=self.device,
            )
        except Exception as e:
            self._model = 'unavailable'
            self._load_error = str(e)

    def _render_skeleton(self, image_path: str,
                         keypoints: List[List[float]],
                         scores: Optional[List[float]] = None) -> np.ndarray:
        """Draw skeleton overlay on image."""
        import cv2

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f'Cannot read image: {image_path}')

        kps = np.array(keypoints, dtype=np.int32)
        conf = scores if scores else [1.0] * len(keypoints)

        # Draw edges
        for (u, v) in self.SKELETON_EDGES:
            if u < len(kps) and v < len(kps) and conf[u] > 0.3 and conf[v] > 0.3:
                cv2.line(img, tuple(kps[u]), tuple(kps[v]), (0, 255, 0), 2)

        # Draw keypoints
        for i, (x, y) in enumerate(kps):
            color = (0, 0, 255) if conf[i] > 0.5 else (128, 128, 128)
            cv2.circle(img, (x, y), 4, color, -1)

        return img

    def __call__(self, image_path: str,
                 keypoints: List[List[float]],
                 scores: Optional[List[float]] = None) -> ToolResult:
        try:
            self._lazy_load()

            if self._model == 'unavailable':
                return ToolResult(
                    tool_name='vlm_assessment', success=False,
                    error=f'VLM model unavailable: {self._load_error}',
                )

            import cv2
            import tempfile
            import torch

            overlay = self._render_skeleton(image_path, keypoints, scores)
            tmp_path = tempfile.mktemp(suffix='.jpg')
            cv2.imwrite(tmp_path, overlay)

            messages = [
                {'role': 'user', 'content': [
                    {'type': 'image', 'image': f'file://{tmp_path}'},
                    {'type': 'text', 'text': self.PROMPT_TEMPLATE},
                ]},
            ]

            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors='pt',
            ).to(self.device)

            with torch.no_grad():
                output_ids = self._model.generate(**inputs, max_new_tokens=256)

            output_text = self._processor.batch_decode(
                output_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )[0].strip()

            # Parse JSON response
            try:
                # Find JSON in response
                start = output_text.find('{')
                end = output_text.rfind('}') + 1
                if start >= 0 and end > start:
                    result = json.loads(output_text[start:end])
                else:
                    result = {'anatomical': 5, 'visibility': 5, 'overall': 5,
                              'issues': 'Failed to parse VLM output'}
            except json.JSONDecodeError:
                result = {'anatomical': 5, 'visibility': 5, 'overall': 5,
                          'issues': f'JSON parse error: {output_text[:100]}'}

            # Normalize to [0, 1]
            quality_score = (
                result.get('anatomical', 5) +
                result.get('visibility', 5) +
                result.get('overall', 5)
            ) / 30.0

            Path(tmp_path).unlink(missing_ok=True)

            return ToolResult(
                tool_name='vlm_assessment', success=True,
                data={
                    'quality_score': round(quality_score, 3),
                    'anatomical': result.get('anatomical', 5),
                    'visibility': result.get('visibility', 5),
                    'overall': result.get('overall', 5),
                    'issues': result.get('issues', ''),
                    'raw_response': output_text[:500],
                },
            )
        except Exception as e:
            return ToolResult(
                tool_name='vlm_assessment', success=False, error=str(e),
            )
