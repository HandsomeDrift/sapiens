# SAGE-Pose（基于 Sapiens 的半监督微调）一页说明

> 这份文档总结了我们目前已经完成并验证可跑的**半监督/无监督**微调方案（在 Sapiens Pose 上）。你可以把它发到新的聊天窗口，让对方“秒懂”工程结构、关键改动、如何运行与常见坑位。

------

## 0. 项目位置与版本约定

- 仓库根：`/home/xiangxiantong/sapiens`

- Pose 子项目：`/home/xiangxiantong/sapiens/pose`

- 重要：使用**仓库自带**的 mmpretrain（包含定制 `sapiens_*.py/ViT`），必须让下面两条路径**优先**于 site-packages：

  ```bash
  export PYTHONPATH=/home/xiangxiantong/sapiens:/home/xiangxiantong/sapiens/pretrain:$PYTHONPATH
  ```

- 训练/推理均基于 **MMEngine + MMPose**（Top-Down pipeline）。

------

## 1. 总体思路（Teacher-Student + 多视图一致性 + 结构先验）

- **学生（Student）**：`TopdownPoseEstimator(ViT backbone + HeatmapHead)`，直接复用官方 Sapiens 配置（1B/0.6B 可切换），仅对 head 通道数等做尺寸对齐。
- **教师（Teacher）**：采用**单教师**（由学生 EMA 得到）。我们不再保留两套独立教师权重，而是用**同一套 teacher**分别喂“几何弱增强”和“外观弱增强”两路输入，融合后作为教师目标。
- **一致性（U 路）**：为每张无标注裁出的人框构造 1~M 组**强增强视图**，将教师的弱增强热图**仿射对齐**到强视图，再与学生输出做**一致性损失**（我们已支持“概率 + KL/JS”方案，较原始 MSE(logits) 更稳）。
- **结构先验**：在学生强视图（最后一组）上对 `soft-argmax` 出的坐标施加三类约束：**拓扑（Laplacian）、骨长、关节角**。
- **AMP + 显存优化**：权重保持 **FP32**，前向用 **bfloat16 autocast**；几何运算（矩阵逆/网格采样）强制 **FP32**；打开 ViT 激活检查点 `with_cp=True`；降分辨率 + 渐进式策略；必要时梯度累积。

------

## 2. 代码改动概览（核心文件与功能）

> 所有路径均相对 `sapiens/pose/`。

### 2.1 半监督模型包装器

**`semisup/models/dual_teacher_wrapper.py`**

- **单教师**：`self.teacher = build(student_cfg)`；初始化时**完整拷贝**学生权重；训练中**每步 EMA 更新**。
- **bf16 前向（teacher）**：教师弱增强在 `autocast(dtype=torch.bfloat16)` 下运行，节省显存；参数仍为 FP32。
- **一致性构建**：
  1. `U = _build_unsup_views(inputs)` 生成：`weak_geom/weak_app/strong_views/affine_mats`。
  2. `fuse_teachers(...)` 融合两路教师（见 §2.3）。
  3. `warp_heatmaps_affine(...)` 将教师热图对齐到学生强视图（见 §2.2）。
  4. 一致性损失：两种可选实现
     - **概率 + KL/JS（推荐）**：对学生/教师在空间维 softmax，再做 KL 或 JS，并按掩码归一（避免被保留比例稀释）。
     - MSE(logits)：保留兼容，量级较小。
- **动态掩码**：`dynamic_kpt_mask(conf, stats, beta, ...)`，支持**热身**（前 N iter 全 1）、**最小保留比例**与**百分位兜底**（避免 mask 全 0 / 全 1）。
- **结构先验**：`soft_argmax_2d(stu_out_last.detach()) → coords_px（热图像素坐标） → topo/bone/angle`。确保**坐标单位**与参考骨长一致。
- **调试日志**：`debug_log_interval` 周期打印 `mask_mean / tea_conf(mean/max) / loss_cons`。

### 2.2 仿射对齐 & 精度修复

**`semisup/utils/geometry.py` → `warp_heatmaps_affine`**

- **FP32 求逆**：`with autocast(False)`，在 FP32 下拼 3×3、`torch.linalg.inv`，避免 bf16 不支持 `inverse` 报错。
- **批维对齐**：展平通道到 `(B*K, 1, H, W)` 采样时，`theta.repeat_interleave(K, dim=0)` 对齐到 `(B*K, 2, 3)`，修复 `affine_grid` 的 batch mismatch。
- 返回前**cast 回原 dtype**（与输入教师热图一致）。

### 2.3 教师融合与置信统计

**`semisup/utils/pseudo_label.py`**

- **`fuse_teachers`**：按关键点置信融合两路教师，并可选在**空间维**做温度化 softmax（`spatial_tau<1` 锐化）。
  - 返回**概率图**（若 `sharpen=True`）与**置信**（默认用概率图的空间最大值 `prob_max`）。
- **`dynamic_kpt_mask`**：指数滑动更新（动量可配），阈值 `thr = μ − βσ`；支持 `min_keep_ratio`（样本维保底）与 `percentile`（关键点维保底）。

### 2.4 无标注数据集（U 路）

**`semisup/data/unlabeled_coco_topdown.py`**

- 输入：`det_json`（与 COCO 测试 `bbox_file` 同 schema），形如 `[{"image_id":int,"category_id":1,"bbox":[x,y,w,h],...}, ...]`。
- 默认将 `image_id` 映射到 `"{:012d}.jpg"`；已提供检测**一致性自检**工具（见 §5.3）。
- Pipeline（U 路最小化）：`LoadImage → GetBBoxCenterScale → EnsureBatchedCenterScale → TopdownAffine → PackPoseInputs`。
  - **注意**：Top-Down 单实例约束，`bbox_center/scale` 形状必须 `(1,2)`。

------

## 3. 配置与路径（两种使用模式）

### 3.1 配置文件

- 纯无监督 / 半监督阶段：
   `configs/sapiens_pose/semisup/stage1_sagepose_semi.py`
- （可选）有监督热身阶段：
   `configs/sapiens_pose/semisup/stage0_sup_warmup.py`

### 3.2 关键配置字段（以 stage1 为例）

```python
# 数据
DATA_ROOT = '/data-nxs/xiangxiantong/stand_data'
train_dataloader = dict(
  batch_size=4, num_workers=4, persistent_workers=False,
  dataset=dict(
    type='UnlabeledCocoTopDownDataset',
    data_root='${DATA_ROOT}',
    img_folder='stand_6fps_together',  # 图像目录
    det_json='${DATA_ROOT}/person_detection_results/stand_detections_person.json',
    pipeline=[ ... 如上所述 ... ],
  )
)

# 模型骨架
model = dict(
  type='DualTeacherWrapper',
  num_keypoints=17,
  student=dict(                 # 与官方 Sapiens 评测一致
    type='TopdownPoseEstimator',
    data_preprocessor=dict(type='PoseDataPreprocessor', mean=[...], std=[...], bgr_to_rgb=True),
    backbone=dict(
      type='mmpretrain.VisionTransformer',
      arch='sapiens_0.6b',     # 或 'sapiens_1b'
      img_size=(H, W), patch_size=16, with_cls_token=False, out_type='featmap',
      with_cp=True,            # 打开激活检查点省显存
      patch_cfg=dict(padding=2),
      init_cfg=dict(type='Pretrained', checkpoint='/ABS/PATH/TO/SAPIENS_0.6B.pth'),
    ),
    head=dict(
      type='HeatmapHead',
      in_channels=<embed_dim_for_model>, out_channels=17,
      deconv_out_channels=(512,512), conv_out_channels=(512,512),
      loss=dict(type='KeypointMSELoss', use_target_weight=True),
      decoder=dict(type='MSRAHeatmap', input_size=(W,H), heatmap_size=(W//4,H//4)),
    ),
    test_cfg=dict(flip_test=True, flip_mode='heatmap', shift_heatmap=False),
  ),

  # 半监督控制
  semi_cfg=dict(
    unsup_only=True,           # 纯无监督；若半监督，把它设 False，且给 L 路 dataloader/evaluator
    lambda_u=5.0,              # 一致性权重
    lambda_topo=0.02, lambda_bone=0.02, lambda_angle=0.01,
    temperature=1.0,           # 融合 w 的温度
    teacher_tau=0.5,           # 教师空间 softmax 温度（<=1 锐化）
    prob_temperature=0.5,      # 概率一致性的温度（<=1 锐化）
    beta=2.0,                  # 掩码阈值强度
    consistency_warmup_iters=1000,  # 掩码热身
    min_keep_ratio=0.3,        # 掩码样本维保底
    num_strong_views=1,        # M，建议 1 以控显存
    edges=[...], angle_triplets=[...], ref_bone_lengths=[...],  # 结构先验
  ),
)

# 训练策略
train_cfg = dict(by_epoch=True, max_epochs=40, val_interval=1)
optim_wrapper = dict(
  type='AmpOptimWrapper', dtype='bfloat16',
  optimizer=dict(type='AdamW', lr=5e-5, weight_decay=0.05),
  accumulative_counts=8,   # 有效 batch = batch_size * accumulative_counts
)
param_scheduler = [ ... 与 epoch 数同步 ... ]
```

> `max_epochs` 在 `train_cfg.max_epochs`；如需命令行覆盖：
>  `--cfg-options train_cfg.max_epochs=60`

------

## 4. 运行方式

### 4.1 单卡（推荐先做冒烟）

```bash
# 纯无监督（U 路）
python pose/tools/train.py \
  pose/configs/sapiens_pose/semisup/stage1_sagepose_semi.py \
  --work-dir /your/work_dir/semi_u_only
```

### 4.2 多卡（注意 NCCL 环境）

```bash
# 例如 4 卡
torchrun --nproc_per_node=4 pose/tools/train.py \
  pose/configs/sapiens_pose/semisup/stage1_sagepose_semi.py \
  --launcher pytorch --work-dir /your/work_dir/semi_u_only

# 建议的 NCCL 环境（按机器网络/驱动情况调整）
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1            # 若无 IB/不稳定
export NCCL_P2P_DISABLE=0           # 同机直连开/关按实际试
export NCCL_NSOCKS_PERTHREAD=2
export NCCL_SOCKET_NTHREADS=2
```

> 若多卡超时（ALLGATHER/ALLREDUCE 超时），优先**确认数据/迭代是否卡死**、`num_workers` 是否过大、磁盘/网络 I/O 是否瓶颈；单卡可跑说明模型 OK，通常是通信或 Dataloader 抖动问题。

### 4.3 半监督两阶段（可选）

1. **有监督热身**（小标注集）：`stage0_sup_warmup.py`
2. **切换无监督/半监督**：`unsup_only=False`（若仍要跑监督分支），并配置 `val/test`。

------

## 5. 数据准备与自检

### 5.1 人体检测 JSON（U 路必需）

- 与 COCO 测试 `bbox_file` 相同 schema：
   `image_id / category_id(=1 for person) / bbox=[x,y,w,h] / score`。
- Sapiens 推理通常**先人检**再关键点（Top-Down），此处我们使用**现成 person 检测器**在新数据集上生成该 JSON。

### 5.2 `image_id` 与文件名

- 我们默认：`img_path = f"{image_id:012d}.jpg"`。
- 你已经做了 **basename→id** 的映射并用自检脚本验证**完全一致**；因此不需要修改微调程序。若有自定义命名规则，可在数据集类中改造为读取 `file_name` 字段。

### 5.3 自检脚本（你已跑过）

- **检测映射一致性**：`check_image_id_mapping.py --img-dir ... --map-json ... --start-from 1`
   输出 `[OK] 完全一致` 即与数据构建规则一致。
- **取样打印**：`ds[0]` 应包含：
   `inputs(3×H×W)`, `data_samples.metainfo` 中的 `bbox_center/scale/img_path/img_id/det_score` 等；
   `center/scale` 形状应为 `(1,2)`。

------

## 6. 日志约定与常见解读

- `Exp name: <cfg_basename>_<time>`：工作目录名。
- `Epoch(train)  [e][i/total]`：第 `e` 个 epoch，当前迭代 `i`，本 epoch 总迭代 `total`。
- 半监督监控：我们在 wrapper 里按 `debug_log_interval` 打印
  - `mask_mean`（掩码保留率）
  - `tea_conf(mean/max)`（教师置信空间最大值的均值/最大）
  - `cons`（一致性标量；如果使用 KL/JS，数学上应 ≥0，数量级与温度相关）
- `The model and loaded state dict do not match exactly`：常见于把 Sapiens 预训练 ViT 迁移到下游（例如无 cls_token、位置编码重采样），**属正常**；只要缺失/意外键可解释即可。

------

## 7. 常见问题与修复（我们已踩过的坑）

1. **用了 site-packages 里的官方 mmpretrain** → 认不出 `arch='sapiens_1b'`

   - 现象：`AssertionError: Arch sapiens_1b is not in default archs`

   - 解决：确保 `PYTHONPATH` 指向仓库版 `sapiens/pretrain`，并在启动前显式导入：

     ```python
     import mmpretrain.models.backbones.vision_transformer
     ```

2. **`PackPoseInputs.__init__() got unexpected keyword 'input_key'`**

   - 原因：版本不一致；移除该参数，按当前 MMPose 版本使用默认键。

3. **Top-Down 单实例断言**：`bbox_center` 形状错误

   - 修：`EnsureBatchedCenterScale`，确保 `(1,2)`。

4. **`warp_heatmaps_affine` 报 bf16 逆阵/批维不匹配**

   - 修：在函数内 `autocast(False)` 用 FP32 逆阵；`theta.repeat_interleave(K)` 与 `(B*K,...)` 对齐。

5. **一致性为 0 或非常小**

   - 先确认 teacher **从 student 拷权初始化**且**每步 EMA 更新**。
   - `fuse_teachers` 开启**空间 softmax（spatial_tau≤1）**，`conf_from='prob_max'`。
   - 一致性改为**概率 + KL/JS**，并对 `mask` 做归一。
   - 设 `beta`/`min_keep_ratio`/`warmup_iters`，避免 mask 全 0 / 全 1。
   - 在强增强中加入**几何扰动**（`RandomAffine/Rotate/Scale/Translate`），并确保 `affine_mats` 正确。

6. **`loss_cons` 出现负数**

   - 使用 `F.kl_div(input=log_softmax(student), target=softmax(teacher), reduction='none')`，**target 必须是概率分布**，warp 后再**严格归一化**；或改用 JS。
   - 若仍负，打印 `p_t.sum()≈1? p_t.min()≥0?`，修正即可。

7. **OOM**

   - 单教师（我们已改）> 降分辨率（如 `768×576` 或 `640×480`）> `with_cp=True` > 降 head 通道 > 梯度累积。
   - AMP 用 bf16；但几何与三角函数务必 FP32。
   - 必要时 `num_strong_views=1`，`batch_size` ↓，`accumulative_counts` ↑。

8. **分布式 NCCL 超时**

   - 先单卡验证稳定；多卡时降低 `num_workers`，检查数据/磁盘吞吐；必要时设置 `NCCL_*` 环境（见 §4.2）。

------

## 8. 运行小抄（最小可用）

**纯无监督（U 路）**：

```bash
export PYTHONPATH=/home/xiangxiantong/sapiens:/home/xiangxiantong/sapiens/pretrain:$PYTHONPATH
python pose/tools/train.py pose/configs/sapiens_pose/semisup/stage1_sagepose_semi.py \
  --work-dir /your/work_dir/semi_u_only \
  --cfg-options train_cfg.max_epochs=40 \
                model.semi_cfg.unsup_only=True \
                model.semi_cfg.num_strong_views=1
```

**两阶段（先有监督热身，再半监督）**：

```bash
# (1) 监督热身
python pose/tools/train.py pose/configs/sapiens_pose/semisup/stage0_sup_warmup.py \
  --work-dir /your/work_dir/stage0_warmup

# (2) 半/无监督
python pose/tools/train.py pose/configs/sapiens_pose/semisup/stage1_sagepose_semi.py \
  --work-dir /your/work_dir/stage1_semi \
  --cfg-options model.semi_cfg.unsup_only=False
```

------

## 9. 你可以告诉下一个助理的“关键状态”

- 目前代码已经**合并为单教师**；教师初始化与 EMA 更新需要在 `DualTeacherWrapper` 中**确保执行**。
- `warp_heatmaps_affine` 已修复**精度**和**批维**问题。
- 我们已接入 `fuse_teachers` 的**空间 softmax + 温度**，并可选用“概率 + KL/JS”一致性；掩码支持**热身/百分位/保底**。
- 数据集（U 路）使用 `UnlabeledCocoTopDownDataset`，路径：
  - 图像：`/data-nxs/xiangxiantong/stand_data/stand_6fps_together/`
  - 人检 JSON：`/data-nxs/xiangxiantong/stand_data/person_detection_results/stand_detections_person.json`（映射已校验一致）。
- 建议的显存方案：`sapiens_0.6b` + `image_size≈768×576` + `with_cp=True` + `num_strong_views=1` + `accumulative_counts`。
- 常看日志项：`mask_mean / tea_conf(mean/max) / loss_cons / loss_bone`；`loss_cons` 若极小或 0，优先检查教师初始化与仿射对齐。

------

下面给出**我们在 Sapiens 项目上新增/自建的代码**的文件树（含作用说明）。根目录以你本地为准：`/home/xiangxiantong/sapiens/pose/`。

```text
pose/
└─ semisup/
   ├─ __init__.py
   ├─ data/
   │  ├─ unlabeled_coco_topdown.py
   │  └─ pipelines/
   │     └─ fix_center_scale.py
   ├─ models/
   │  ├─ dual_teacher_wrapper.py
   │  └─ losses/
   │     └─ structural_priors.py
   └─ utils/
      ├─ geometry.py
      └─ pseudo_label.py

configs/sapiens_pose/semisup/
├─ stage0_sup_warmup.py               # 有监督热身（可选）
└─ stage1_sagepose_semi.py            # 无监督半监督阶段（核心）
scripts/train/semisup/
├─ node_sup.sh                        # 跑 Stage 0
└─ node_semi.sh                       # 跑 Stage 1（可单独用）

```

> 