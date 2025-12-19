import argparse
import os
from pathlib import Path
import json

import gradio as gr

from gradio_pose17 import PoseDemo, default_paths


def load_cached_image(path_str):
    import cv2
    import numpy as np

    if path_str is None:
        return None
    img = cv2.imread(path_str, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_cached_json(path_str):
    if path_str is None:
        return None
    try:
        with open(path_str, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def build_demo(args):
    pose_ckpt = Path(args.pose_checkpoint)
    det_cfg = Path(args.det_config)
    det_ckpt = Path(args.det_checkpoint)
    cached_image_np = load_cached_image(args.cached_output)
    cached_json_obj = load_cached_json(args.cached_json)
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
    label, .label-wrap, .wrap {
        color: var(--text) !important;
    }
    .container {
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
        gr.HTML(custom_css, elem_id="custom-css-dual")

        gr.HTML(
            f"""
            <div class="hero">
                <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap;">
                    <div>
                        <div style="font-size:18px; letter-spacing:0.5px;">✨ Sapiens Pose Demo · 双侧可上传</div>
                        <div style="color: var(--muted); font-size:14px;">左侧上传 → 推理；右侧既可显示结果也可自行上传图片（供比对/再推理）。</div>
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
            "默认启用 RTMDet 行人检测，未检测到人物时自动回退为整图。"
            " 本演示可指定一张预先推理好的可视化图片（启动参数 `--cached-output`），点击运行时将优先展示该图片以节省推理时间；若未提供，则按输入推理。",
            elem_classes=["panel"],
        )
        with gr.Row():
            with gr.Column(scale=3, elem_classes=["panel"]):
                image_left = gr.Image(type="numpy", label="上传图片（左）", height=340)
                run_btn = gr.Button("🚀 运行推理", variant="primary", elem_classes=["primary-btn"])
            with gr.Column(scale=3, elem_classes=["panel"]):
                image_right = gr.Image(
                    type="numpy",
                    label="可视化输出",
                    height=340,
                    interactive=False,
                )
                json_out = gr.JSON(label="关键点 JSON", value=None)

        def run_inference(left_img, right_img):
            progress = gr.Progress(track_tqdm=False)

            # 若指定了预先结果，模拟推理进度后直接返回
            if cached_image_np is not None:
                progress(0.1, desc="排队中…")
                import time

                time.sleep(0.5)
                progress(0.5, desc="加载预先结果…")
                time.sleep(0.5)
                progress(0.9, desc="完成")
                return cached_image_np, cached_json_obj

            # 优先使用左侧输入；若左侧为空则用右侧输入
            chosen = left_img if left_img is not None else right_img
            if chosen is None:
                return None, None

            progress(0.15, desc="加载模型…")
            # 进入真实推理
            img, payload = demo_runner.run(chosen)
            return img, payload

        run_btn.click(
            fn=run_inference,
            inputs=[image_left, image_right],
            outputs=[image_right, json_out],
        )
    return demo


def parse_args():
    mode = os.environ.get("MODE", "torchscript")
    pose_ckpt, det_cfg, det_ckpt = default_paths(mode)
    parser = argparse.ArgumentParser(description="Gradio dual-upload demo for pose estimation")
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
    parser.add_argument(
        "--cached-output",
        type=str,
        default=None,
        help="Path to a pre-computed visualization image. If set, the app will simulate progress and directly show this image instead of running the model.",
    )
    parser.add_argument(
        "--cached-json",
        type=str,
        default=None,
        help="Path to a pre-computed keypoints JSON to display with the cached output image.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    demo = build_demo(args)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
