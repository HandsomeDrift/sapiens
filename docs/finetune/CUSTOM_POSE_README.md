# Finetuning Sapiens: Custom 2D Pose (COCO17)
本指南沿用 `docs/finetune/POSE_README.md` 的格式，面向你们的自定义数据集（COCO17 关键点）。

## 📂 1. 数据准备
你们当前标注是 LabelMe JSON（最终标注文件不含 `_labelme_init_coco17` 后缀）。需要先转成 COCO Keypoints 规范，并整理成官方微调格式的目录结构：

```
$DATA_ROOT/
  xt_train/
  xt_val/
  annotations/
    person_keypoints_xt_train.json
    person_keypoints_xt_val.json
  person_detection_results/
    COCO_xt_val_detections_AP_H_70_person.json
```

### A. LabelMe → COCO17 + 目录重排
使用转换脚本自动递归读取三级目录的 LabelMe JSON，并按比例拆分 train/val。  
默认使用 **符号链接** 将图片放到 `train2017/` 和 `val2017/`，也支持 `--copy` 或 `--move`。  
如果希望 `train2017/` 和 `val2017/` 下不保留原始子目录层级，添加 `--flat-images`。

```bash
python pose/scripts/convert/labelme_to_coco_pose17.py \
  --input-root /path/to/raw_labelme_root \
  --output-root /path/to/output_dataset_root \
  --train-ratio 0.9 --seed 42 \
  --copy-mode symlink \
  --train-name xt_train \
  --val-name xt_val \
  --flat-images
```

### B. 生成检测框结果（bbox_file）
参考官方推理脚本思路，使用 MMDetection 检测器生成 `bbox_file`。

```bash
export PYTHONPATH="/home/drift/sapiens:/home/drift/sapiens/pretrain:$PYTHONPATH"
python pose/scripts/convert/generate_person_dets.py \
  --ann-file /path/to/output_dataset_root/annotations/person_keypoints_xt_val.json \
  --data-root /path/to/output_dataset_root \
  --image-prefix xt_val \
  --det-config pose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person_no_nms.py \
  --det-checkpoint /path/to/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
  --out /path/to/output_dataset_root/person_detection_results/COCO_xt_val_detections_AP_H_70_person.json \
  --score-thr 0.3 \
  --device cuda:0
```

## ⚙️ 2. 配置更新
拷贝并修改配置：
```
pose/configs/sapiens_pose/custom/sapiens_1b-quickstart_custom_coco17.py
```

需要更新的关键字段：
1. `pretrained_checkpoint`：Sapiens 预训练权重路径  
2. `data_root`：指向上面生成的 `$DATA_ROOT`  
3. `bbox_file`：指向 `COCO_val2017_detections_AP_H_70_person.json`  

## 🏋️ 3. 微调
单机示例（仅供参考，具体参数按服务器调整）：
```bash
export PYTHONPATH="/home/drift/sapiens:/home/drift/sapiens/pretrain:$PYTHONPATH"
python pose/tools/train.py pose/configs/sapiens_pose/custom/sapiens_1b-quickstart_custom_coco17.py \
  --work-dir work_dirs/custom_coco17_sup \
  --cfg-options load_from=/path/to/sapiens_1b_checkpoint.pth train_dataloader.batch_size=4
```

如需脚本化多卡，可参考 `pose/scripts/finetune/coco/sapiens_1b/node.sh` 的结构，将配置路径替换为自定义配置。
