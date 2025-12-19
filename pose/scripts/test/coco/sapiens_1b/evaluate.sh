# CONFIG=configs/sapiens_pose/coco/sapiens_1b-210e_coco-1024x768.py
# CKPT="/data/xiangxiantong/sapiens_lite_host/torchscript/pose/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2"  # 改成你的训练产物
# GPUS=8  # 按你的机器改

# bash tools/dist_test.sh $CONFIG $CKPT $GPUS \
#   --cfg-options \
#   val_dataloader.dataset.data_root=/data-nxs/xiangxiantong/fine-tuning_data \
#   val_dataloader.dataset.ann_file=annotations/person_keypoints_val2017.json \
#   val_dataloader.dataset.bbox_file=/data-nxs/xiangxiantong/fine-tuning_data/person_detection_results/COCO_val2017_detections_AP_H_70_person.json \
#   default_hooks.visualization.enable=False


# 设定仓库根目录 & 提前把 repo 内的 mmpretrain 放在最前
REPO_ROOT=/home/xiangxiantong/sapiens
export PYTHONPATH="$REPO_ROOT/pretrain:$REPO_ROOT/pose:$PYTHONPATH"

python - <<'PY'
import sys
import mmpretrain.models.backbones.vision_transformer as vt

# 1) 路径检查：必须来自你的仓库，而不是 site-packages
print("Using vision_transformer.py:", vt.__file__)
assert vt.__file__.startswith("/home/xiangxiantong/sapiens/pretrain/"), \
    "❌ 当前导入的不是仓库内的 mmpretrain，请检查 PYTHONPATH 优先级。"

# 2) 架构别名检查：sapiens_1b 在 arch_zoo（如果该版本有 arch_zoo）
if hasattr(vt, "arch_zoo"):
    assert "sapiens_1b" in vt.arch_zoo, \
        "❌ arch_zoo 里没有 'sapiens_1b'，请确认你仓库版本是否包含该条目。"

# 3) 实例化检查：能否直接用 arch='sapiens_1b' 构造模型
#    注意参数需与项目一致；这里给出常见 1B 设置与关键信号位
model = vt.VisionTransformer(
    arch='sapiens_1b',
    with_cls_token=False,
    out_type='featmap',
    img_size=(1024, 768),   # 与你的 config 对齐
    patch_size=16
)
n_params = sum(p.numel() for p in model.parameters())
print(f"✅ VisionTransformer('sapiens_1b') 构建成功，参数量：{n_params/1e6:.2f}M")
PY
