"""
MODE=torchscript \
    python lite/demo/gradio_pose17.py \
      --pose-checkpoint /data/${USER}/sapiens_lite_host/torchscript/pose/checkpoints/sapiens_2b/sapiens_2b_coco_best_coco_AP_822_torchscript.pt2 \
      --det-config pose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person_no_nms.py \
      --det-checkpoint /data/${USER}/sapiens_lite_host/torchscript/detector/checkpoints/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
      --device cuda:0 \
      --num-keypoints 17
"""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import cv2
import gradio as gr
import numpy as np
import torch

from classes_and_palettes import (
    COCO_KPTS_COLORS,
    COCO_SKELETON_INFO,
    COCO_WHOLEBODY_KPTS_COLORS,
    COCO_WHOLEBODY_SKELETON_INFO,
    GOLIATH_KPTS_COLORS,
    GOLIATH_SKELETON_INFO,
)
from detector_utils import adapt_mmdet_pipeline, init_detector, process_images_detector
from pose_utils import udp_decode
from vis_pose import load_model, preprocess_pose


def default_paths(mode: str) -> Tuple[Path, Path, Path]:
    base_root = Path(
        os.environ.get(
            "SAPIENS_CHECKPOINT_ROOT",
            f"/data/{os.environ.get('USER', 'user')}/sapiens_lite_host",
        )
    )
    root = base_root / mode
    repo_root = Path(__file__).resolve().parents[2]
    pose_ckpt = (
        root
        / "pose"
        / "checkpoints"
        / "sapiens_2b"
        / f"sapiens_2b_coco_best_coco_AP_822_{mode}.pt2"
    )
    det_cfg = (
        repo_root
        / "pose"
        / "demo"
        / "mmdetection_cfg"
        / "rtmdet_m_640-8xb32_coco-person_no_nms.py"
    )
    det_ckpt = (
        root
        / "detector"
        / "checkpoints"
        / "rtmpose"
        / "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
    )
    return pose_ckpt, det_cfg, det_ckpt


def pick_palette(num_keypoints: int):
    if num_keypoints == 17:
        return COCO_KPTS_COLORS, COCO_SKELETON_INFO
    if num_keypoints == 308:
        return GOLIATH_KPTS_COLORS, GOLIATH_SKELETON_INFO
    return COCO_WHOLEBODY_KPTS_COLORS, COCO_WHOLEBODY_SKELETON_INFO


class PoseDemo:
    def __init__(
        self,
        pose_checkpoint: Path,
        det_config: Path,
        det_checkpoint: Path,
        device: str = "cuda:0",
        num_keypoints: int = 17,
        heatmap_scale: int = 4,
        kpt_thr: float = 0.3,
        radius: int = 6,
        thickness: int = -1,
        image_shape: Tuple[int, int] = (1024, 768),
        use_fp16: bool = False,
    ):
        self.pose_checkpoint = Path(pose_checkpoint)
        self.det_config = Path(det_config)
        self.det_checkpoint = Path(det_checkpoint)
        self.device = device
        self.num_keypoints = num_keypoints
        self.heatmap_scale = heatmap_scale
        self.kpt_thr = kpt_thr
        self.radius = radius
        self.thickness = thickness if thickness != -1 else radius
        self.image_shape = image_shape
        self.use_fp16 = use_fp16

        self.pose_estimator: torch.nn.Module = None
        self.detector = None
        self.dtype = torch.float32
        self.kpt_colors, self.skeleton_info = pick_palette(num_keypoints)

    def load_models(self):
        if self.pose_estimator is None:
            use_torchscript = "_torchscript" in str(self.pose_checkpoint)
            pose_estimator = load_model(self.pose_checkpoint, use_torchscript)
            pose_estimator = pose_estimator.to(self.device)
            device_type = "cuda" if "cuda" in self.device else "cpu"
            if use_torchscript:
                self.dtype = torch.float32
            else:
                if device_type == "cuda":
                    self.dtype = torch.float16 if self.use_fp16 else torch.bfloat16
                    pose_estimator = pose_estimator.to(self.dtype)
                    pose_estimator = torch.compile(
                        pose_estimator, mode="max-autotune", fullgraph=True
                    )
                else:
                    self.dtype = torch.float32
            self.pose_estimator = pose_estimator

        if self.detector is None:
            detector = init_detector(
                str(self.det_config), str(self.det_checkpoint), device=self.device
            )
            detector.cfg = adapt_mmdet_pipeline(detector.cfg)
            self.detector = detector

    def run(self, image: np.ndarray):
        if image is None:
            return None, {"error": "No image provided"}

        self.load_models()
        input_shape = (3, self.image_shape[0], self.image_shape[1])
        input_wh = (self.image_shape[1], self.image_shape[0])

        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 255)).astype(np.uint8)
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        # gradio image comes in RGB; convert to BGR to align with cv2 usage
        bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        det_args = SimpleNamespace(
            det_cat_id=0, bbox_thr=0.3, nms_thr=0.3, det_checkpoint=str(self.det_checkpoint)
        )
        det_input = bgr_img[..., ::-1][None, ...]  # to RGB batch for detector
        bboxes_batch = process_images_detector(det_args, det_input, self.detector)
        bboxes = bboxes_batch[0] if bboxes_batch else []
        if len(bboxes) == 0:
            h, w = bgr_img.shape[:2]
            bboxes = np.array([[0, 0, w, h]])

        pose_imgs, pose_img_centers, pose_img_scales = preprocess_pose(
            bgr_img,
            bboxes,
            (input_shape[1], input_shape[2]),
            [123.5, 116.5, 103.5],
            [58.5, 57.0, 57.5],
        )

        if len(pose_imgs) == 0:
            return image, {"error": "Failed to build pose inputs"}

        imgs = torch.stack(pose_imgs, dim=0).to(self.device)

        device_type = "cuda" if "cuda" in self.device else "cpu"
        autocast_dtype = self.dtype if device_type == "cuda" else torch.float32
        with torch.no_grad(), torch.autocast(
            device_type=device_type, dtype=autocast_dtype
        ):
            heatmaps = self.pose_estimator(imgs)

        annotated, keypoints_payload = self._decode_and_draw(
            bgr_img.copy(),
            heatmaps,
            pose_img_centers,
            pose_img_scales,
            input_wh,
        )

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        return rgb, keypoints_payload

    def _decode_and_draw(
        self,
        img: np.ndarray,
        heatmaps: torch.Tensor,
        centres: List,
        scales: List,
        input_wh: Tuple[int, int],
    ) -> Tuple[np.ndarray, Dict]:
        instance_keypoints = []
        instance_scores = []
        input_w, input_h = input_wh
        for idx, heatmap in enumerate(heatmaps):
            result = udp_decode(
                heatmap.cpu().unsqueeze(0).float().data[0].numpy(),
                (
                    input_w,
                    input_h,
                ),
                (
                    int(input_w / self.heatmap_scale),
                    int(input_h / self.heatmap_scale),
                ),
            )
            keypoints, keypoint_scores = result
            keypoints = (
                (keypoints / (input_w, input_h))
                * scales[idx]
                + centres[idx]
                - 0.5 * scales[idx]
            )
            instance_keypoints.append(keypoints[0])
            instance_scores.append(keypoint_scores[0])

        keypoints_visible = np.ones((len(instance_keypoints), self.num_keypoints))
        for kpts, score, visible in zip(
            instance_keypoints, instance_scores, keypoints_visible
        ):
            for kid, kpt in enumerate(kpts):
                if score[kid] < self.kpt_thr or not visible[kid]:
                    continue
                color = self.kpt_colors[kid]
                if not isinstance(color, str):
                    color = tuple(int(c) for c in color[::-1])
                img = cv2.circle(
                    img, (int(kpt[0]), int(kpt[1])), int(self.radius), color, -1
                )
            for _, link_info in self.skeleton_info.items():
                pt1_idx, pt2_idx = link_info["link"]
                color = link_info["color"][::-1]
                pt1, pt2 = kpts[pt1_idx], kpts[pt2_idx]
                pt1_score, pt2_score = score[pt1_idx], score[pt2_idx]
                if pt1_score > self.kpt_thr and pt2_score > self.kpt_thr:
                    cv2.line(
                        img,
                        (int(pt1[0]), int(pt1[1])),
                        (int(pt2[0]), int(pt2[1])),
                        color,
                        self.thickness,
                    )

        payload = {
            "instances": [
                {
                    "keypoints": kp.tolist(),
                    "keypoint_scores": sc.tolist(),
                }
                for kp, sc in zip(instance_keypoints, instance_scores)
            ]
        }
        return img, payload


def build_demo(args):
    pose_ckpt = Path(args.pose_checkpoint)
    det_cfg = Path(args.det_config)
    det_ckpt = Path(args.det_checkpoint)
    demo_runner = PoseDemo(
        pose_ckpt,
        det_cfg,
        det_ckpt,
        device=args.device,
        num_keypoints=args.num_keypoints,
        heatmap_scale=args.heatmap_scale,
        kpt_thr=args.kpt_thr,
        radius=args.radius,
        thickness=args.thickness,
        image_shape=tuple(args.shape),
        use_fp16=args.fp16,
    )

    custom_css = """
    <style>
    :root {
        --bg: #f5f7fb;
        --panel: #ffffff;
        --glow: rgba(37, 99, 235, 0.18);
        --text: #1f2937;
        --muted: #6b7280;
        --accent: #2563eb;
        --accent-2: #0ea5e9;
        --stroke: #e5e7eb;
    }
    html, body {
        background: var(--bg) !important;
        color: var(--text);
        font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
        min-height: 100vh;
    }
    .gradio-container, .app {
        width: 96vw; max-width: 1360px; margin: 0 auto; color: var(--text);
        background: transparent !important;
    }
    .block, .panel, .gr-box, .gr-panel, .tabs, .tabitem {
        background: var(--panel) !important;
        color: var(--text) !important;
        border-color: var(--stroke) !important;
    }
    label, .label-wrap, .wrap.svelte-18f9e1u, .wrap.svelte-10irmof, .wrap {
        color: var(--text) !important;
    }
    .container.svelte-1ipelgc, .container {
        background: transparent !important;
    }
    .panel {
        border: 1px solid var(--stroke);
        background: var(--panel);
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08), 0 0 0 1px rgba(255,255,255,0.6);
        border-radius: 18px; padding: 18px;
    }
    .hero {
        border-radius: 20px;
        padding: 18px 20px;
        background: linear-gradient(120deg, rgba(37,99,235,0.12), rgba(14,165,233,0.1));
        border: 1px solid var(--stroke);
        box-shadow: 0 16px 32px rgba(15, 23, 42, 0.12);
    }
    .primary-btn button {
        background: linear-gradient(120deg, var(--accent), var(--accent-2));
        border: none; color: #ffffff; font-weight: 700;
        box-shadow: 0 10px 22px rgba(37,99,235,0.25);
        width: 240px; height: 50px; font-size: 16px;
    }
    .primary-btn button:hover {filter: brightness(1.05);}
    .stat {
        padding: 8px 12px; border-radius: 12px;
        background: rgba(37,99,235,0.08);
        display: inline-block; margin-right: 10px; font-size: 13px; color: var(--text);
        border: 1px solid rgba(37,99,235,0.18);
    }
    .gradio-image, .image-frame, .preview {
        border: 1px dashed var(--stroke) !important;
        border-radius: 14px !important;
        background: #f8fafc !important;
    }
    input, textarea, select {
        background: #f8fafc !important;
        color: var(--text) !important;
        border: 1px solid var(--stroke) !important;
        border-radius: 12px !important;
    }
    .json-wrap pre {
        background: #f8fafc !important;
        border-radius: 14px !important;
        color: var(--text) !important;
        border: 1px solid var(--stroke) !important;
    }
    footer {color: var(--muted) !important;}
    </style>
    """

    with gr.Blocks() as demo:
        # 注入样式
        gr.HTML(custom_css, elem_id="custom-css")
        gr.HTML(
            f"""
            <div class="hero">
                <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap;">
                    <div>
                        <div style="font-size:18px; letter-spacing:0.5px;">✨ Sapiens Pose Demo · {args.num_keypoints} Keypoints</div>
                        <div style="color: var(--muted); font-size:14px;">上传图片 → 检测 → 姿态估计 → 可视化与 JSON</div>
                    </div>
                    <div>
                        <span class="stat">模型: {pose_ckpt.name}</span>
                        <span class="stat">设备: {args.device}</span>
                        <span class="stat">输入尺寸: {args.shape[0]}x{args.shape[1]}</span>
                    </div>
                </div>
            </div>
            """
        )
        gr.Markdown(
            "默认启用 RTMDet 行人检测，未检测到人物时自动回退为整图；输出包含渲染图和关键点 JSON。",
            elem_classes=["panel"],
        )
        with gr.Row():
            with gr.Column(scale=3, elem_classes=["panel"]):
                image_in = gr.Image(type="numpy", label="上传图片", height=340)
                run_btn = gr.Button("🚀 运行推理", variant="primary", elem_classes=["primary-btn"])
            with gr.Column(scale=3, elem_classes=["panel"]):
                image_out = gr.Image(type="numpy", label="可视化输出", height=340)
                json_out = gr.JSON(label="关键点 JSON")

        run_btn.click(
            fn=demo_runner.run,
            inputs=image_in,
            outputs=[image_out, json_out],
        )
    return demo


def parse_args():
    mode = os.environ.get("MODE", "torchscript")
    pose_ckpt, det_cfg, det_ckpt = default_paths(mode)
    parser = argparse.ArgumentParser(description="Gradio demo for pose estimation")
    parser.add_argument(
        "--pose-checkpoint",
        type=str,
        default=str(pose_ckpt),
        help="Path to pose checkpoint (.pt2).",
    )
    parser.add_argument(
        "--det-config",
        type=str,
        default=str(det_cfg),
        help="Path to detector config file.",
    )
    parser.add_argument(
        "--det-checkpoint",
        type=str,
        default=str(det_ckpt),
        help="Path to detector checkpoint.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device for inference."
    )
    parser.add_argument(
        "--num-keypoints",
        type=int,
        default=17,
        choices=[17, 133, 308],
        help="Number of keypoints in pose model.",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        default=[1024, 768],
        help="Input size (height width).",
    )
    parser.add_argument(
        "--heatmap-scale", type=int, default=4, help="Heatmap downsample ratio."
    )
    parser.add_argument(
        "--kpt-thr", type=float, default=0.3, help="Keypoint confidence threshold."
    )
    parser.add_argument(
        "--radius", type=int, default=6, help="Keypoint radius for visualization."
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=-1,
        help="Skeleton thickness. Defaults to radius when -1.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 for exported models (ignored for torchscript).",
    )
    parser.add_argument(
        "--share", action="store_true", help="Enable gradio share tunnel."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    demo = build_demo(args)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
