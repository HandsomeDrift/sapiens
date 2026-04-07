# sapiens 环境配置记录

本记录总结了当前可用的 Conda 环境版本组合与安装步骤，便于后续快速重建。

## 目标环境

- Python 3.10（环境名示例：`sapiens-clean`）
- PyTorch 2.2.2 + CUDA 12.1（官方 cu121 轮子）
- numpy 1.26.4（避免与 torch/mmcv 预编译扩展冲突）
- mmengine 0.10.7
- mmcv 2.1.0（cu121 + torch2.2 预编译轮子）
- mmdet 3.2.0
- OpenCV 4.9.0.80
- 其他辅助包：future、tensorboard、chumpy、scipy、munkres、cython、tqdm、fsspec、yapf==0.40.1、matplotlib、packaging、omegaconf、ipdb、ftfy、regex、json_tricks、terminaltables、modelindex、prettytable、albumentations

## 安装步骤

在远程服务器执行，路径以 `/data/xxt/sapiens_code/sapiens` 为例。

1) 创建环境
```bash
conda create -n sapiens-clean python=3.10 -y
conda activate sapiens-clean
pip install "pip<24" "setuptools<70" wheel
```

2) 安装核心依赖
```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy==1.26.4"
pip install "opencv-python==4.9.0.80"
pip install "mmengine==0.10.7"
pip install --no-cache-dir openmim
mim install --no-cache-dir "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.2/index.html
mim install --no-cache-dir "mmdet==3.2.0"
pip install future tensorboard
# 其余依赖
pip install chumpy scipy munkres tqdm cython fsspec yapf==0.40.1 matplotlib packaging omegaconf ipdb ftfy regex json_tricks terminaltables modelindex prettytable albumentations
```

3) 安装仓库子模块（可编辑模式，关闭 PEP517/隔离）
```bash
cd /data/xxt/sapiens_code/sapiens
pip install -e engine
pip install -e cv
pip install -e pretrain
pip install -e pose
pip install -e det
pip install -e seg   # 不跑分割可略
```

4) 设置环境变量（使用仓库自带 mmpretrain/mmpose）
```bash
export PYTHONPATH="/data/xxt/sapiens_code/sapiens:/data/xxt/sapiens_code/sapiens/pretrain:/data/xxt/sapiens_code/sapiens/pose:$PYTHONPATH"
```

5) 运行前快速自检
```bash
python - <<'PY'
import torch, numpy as np, mmcv, mmdet
from mmdet.apis import inference_detector, init_detector
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("numpy", np.__version__)
print("mmcv", mmcv.__version__)
print("mmdet", mmdet.__version__)
PY
```

## 常见问题与处理

- `torch._inductor.config.force_fuse_int_mm_with_mul` 缺失：需使用 torch ≥ 2.2。
- `ModuleNotFoundError: mmcv._ext`：未装到预编译轮子，按上面的 mmcv 链接重装，确保 torch 版本与 URL 匹配。
- numpy 2.x 相关报错：将 numpy 固定为 1.26.4，并使用与之兼容的 opencv 版本。
- 输出目录权限错误：修改脚本中的 OUTPUT 路径到有写权限的位置（例如 `/data/xxt/...`）。
