#!/usr/bin/env python
"""CLI entry point for the multi-agent pose quality control pipeline.

Examples:
    # Structural validation only (no GPU needed for VLM):
    python run_quality_pipeline.py \
        --images-dir /data/xxt/sapiens_data/xt_val \
        --stats ../stats/skeleton_stats_coco17.json \
        --ann /data/xxt/sapiens_data/annotations/person_keypoints_xt_val.json \
        --output quality_results.json \
        --no-vlm

    # Full pipeline with VLM:
    python run_quality_pipeline.py \
        --images-dir /data/xxt/sapiens_data/xt_val \
        --stats ../stats/skeleton_stats_coco17.json \
        --output quality_results.json \
        --vlm-model Qwen/Qwen2-VL-2B-Instruct

    # With pre-computed predictions (skip pose detection):
    python run_quality_pipeline.py \
        --predictions predictions.json \
        --stats ../stats/skeleton_stats_coco17.json \
        --output quality_results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Multi-agent pose quality control pipeline')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--images-dir', type=str,
                             help='Directory containing images to assess')
    input_group.add_argument('--predictions', type=str,
                             help='JSON file with pre-computed predictions')

    parser.add_argument('--stats', required=True,
                        help='Path to skeleton_stats.json')
    parser.add_argument('--ann', type=str, default=None,
                        help='COCO annotation JSON (for pre-computed keypoints)')
    parser.add_argument('--output', required=True,
                        help='Output JSON path for quality results')

    parser.add_argument('--no-vlm', action='store_true',
                        help='Disable VLM assessment (structural only)')
    parser.add_argument('--vlm-model', default='Qwen/Qwen2-VL-2B-Instruct',
                        help='VLM model name')
    parser.add_argument('--device', default='cuda:0')

    parser.add_argument('--accept-threshold', type=float, default=0.75)
    parser.add_argument('--reject-threshold', type=float, default=0.4)
    parser.add_argument('--max-images', type=int, default=None,
                        help='Limit number of images to process')

    # Export filtered COCO
    parser.add_argument('--export-coco', type=str, default=None,
                        help='Export accepted annotations as filtered COCO JSON')

    args = parser.parse_args()

    from orchestrator import QualityOrchestrator

    # Initialize orchestrator
    orch = QualityOrchestrator(
        stats_path=args.stats,
        use_vlm=not args.no_vlm,
        vlm_model=args.vlm_model,
        accept_threshold=args.accept_threshold,
        reject_threshold=args.reject_threshold,
        device=args.device,
    )

    # Prepare inputs
    image_paths = []
    annotations = None

    if args.images_dir:
        img_dir = Path(args.images_dir)
        exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_paths = sorted(
            str(p) for p in img_dir.iterdir() if p.suffix.lower() in exts
        )
        logger.info(f'Found {len(image_paths)} images in {args.images_dir}')

        # Load pre-computed keypoints from COCO annotation if available
        if args.ann:
            with open(args.ann) as f:
                coco = json.load(f)
            # Build image_id → annotation mapping
            ann_by_imgid = {}
            for ann in coco.get('annotations', []):
                imgid = ann['image_id']
                if imgid not in ann_by_imgid:
                    ann_by_imgid[imgid] = ann

            fname_to_id = {
                img['file_name']: img['id'] for img in coco['images']
            }

            annotations = []
            for p in image_paths:
                fname = Path(p).name
                imgid = fname_to_id.get(fname)
                if imgid and imgid in ann_by_imgid:
                    ann = ann_by_imgid[imgid]
                    kps_raw = ann.get('keypoints', [])
                    if len(kps_raw) >= 17 * 3:
                        kps = [[kps_raw[i*3], kps_raw[i*3+1]]
                               for i in range(17)]
                        scr = [kps_raw[i*3+2] for i in range(17)]
                        annotations.append({'keypoints': kps, 'scores': scr})
                    else:
                        annotations.append({})
                else:
                    annotations.append({})

    elif args.predictions:
        with open(args.predictions) as f:
            preds = json.load(f)
        for p in preds:
            image_paths.append(p['image_path'])
        annotations = [
            {'keypoints': p.get('keypoints'), 'scores': p.get('scores')}
            for p in preds
        ]

    if args.max_images:
        image_paths = image_paths[:args.max_images]
        if annotations:
            annotations = annotations[:args.max_images]

    logger.info(f'Processing {len(image_paths)} images...')

    # Run pipeline
    def progress(cur, total):
        if cur % 10 == 0 or cur == total:
            print(f'\r  Progress: {cur}/{total}', end='', flush=True)

    results = orch.assess_batch(image_paths, annotations, progress)
    print()  # newline after progress

    # Summary
    n_accept = sum(1 for r in results if r.decision == 'accept')
    n_reject = sum(1 for r in results if r.decision == 'reject')
    n_review = sum(1 for r in results if r.decision == 'review')
    logger.info(
        f'Results: {n_accept} accept, {n_reject} reject, {n_review} review '
        f'(total {len(results)})'
    )

    # Save
    orch.save_results(results, args.output)

    # Export filtered COCO if requested
    if args.export_coco and args.ann:
        orch.export_filtered_coco(results, args.ann, args.export_coco)


if __name__ == '__main__':
    main()
