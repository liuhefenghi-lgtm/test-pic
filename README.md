# PCM 彩板表面缺陷检测系统

基于 **YOLO11**（Ultralytics 8.4.x）的 PCM（Pre-Coated Metal / 彩涂板）表面质量自动检测系统，支持 **划伤、凸包、脏污、压痕、色差、起泡** 等 6 类缺陷的实时识别。

## 缺陷类别

| ID | 英文名       | 中文名 |
|----|-------------|--------|
| 0  | scratch     | 划伤   |
| 1  | bump        | 凸包   |
| 2  | stain       | 脏污   |
| 3  | indentation | 压痕   |
| 4  | color_diff  | 色差   |
| 5  | blister     | 起泡   |

## 项目结构

```
test-pic/
├── configs/
│   └── defect_config.yaml   # 数据集、模型、训练、推理配置
├── data/
│   ├── augmentation.py      # 金属表面专用数据增强管线
│   └── dataset.py           # 数据集验证、YAML生成、类别统计
├── models/
│   └── detector.py          # PCMDefectDetector（YOLO11封装）
├── utils/
│   ├── visualization.py     # 检测结果可视化（带颜色编码边框）
│   └── metrics.py           # mAP/Precision/Recall/F1/混淆矩阵
├── train.py                 # 训练入口
├── predict.py               # 推理入口
├── evaluate.py              # 评估入口
├── demo.py                  # 快速演示（无需真实数据）
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 快速演示（无需真实数据）

```bash
python demo.py
```

脚本将自动合成模拟 PCM 板缺陷图像，快速训练 5 个 epoch，并生成可视化结果图 `runs/demo/demo_result.jpg`。

---

## 数据集准备

### 目录结构（YOLO 格式）

```
dataset/
├── dataset.yaml       # 数据集描述（自动生成）
├── images/
│   ├── train/         # 训练图像 *.jpg
│   ├── val/           # 验证图像
│   └── test/          # 测试图像
└── labels/
    ├── train/         # YOLO标注 *.txt
    ├── val/
    └── test/
```

### 标注格式（YOLO txt）

每个 `.txt` 文件对应一张图像，每行一个缺陷：

```
<class_id> <cx> <cy> <width> <height>
```

坐标均为归一化值（0~1），例：
```
0 0.512 0.341 0.124 0.058
2 0.748 0.612 0.089 0.091
```

### 推荐标注工具

- [LabelImg](https://github.com/HumanSignal/labelImg)（本地，免费）
- [Roboflow](https://roboflow.com)（在线，支持团队协作）
- [CVAT](https://cvat.ai)（在线，支持视频标注）

### 数据集划分

如果已有原始标注，可使用内置工具自动划分：

```python
from data.dataset import split_dataset

split_dataset(
    source_dir="raw_data/",   # 含 images/ 和 labels/ 的原始目录
    output_dir="dataset/",
    train_ratio=0.7,
    val_ratio=0.15,           # test = 1 - 0.7 - 0.15 = 0.15
)
```

---

## 训练

### 基本训练

```bash
python train.py --data dataset/dataset.yaml
```

### 自定义参数

```bash
python train.py \
    --data dataset/dataset.yaml \
    --config configs/defect_config.yaml \
    --epochs 200 \
    --batch 16 \
    --device 0          # GPU 0
```

### 断点续训

```bash
python train.py --data dataset/dataset.yaml --resume
```

### 训练前验证数据集

```bash
python train.py --data dataset/dataset.yaml --verify-data
```

训练结果保存于 `runs/pcm_defect_train/`：
- `weights/best.pt` — 验证集最优权重
- `weights/last.pt` — 最后 epoch 权重
- `results.png` — 训练曲线

---

## 推理

### 单张图片

```bash
python predict.py \
    --model runs/pcm_defect_train/weights/best.pt \
    --source /path/to/pcm_image.jpg
```

### 批量目录

```bash
python predict.py \
    --model best.pt \
    --source /path/to/images/ \
    --output ./results \
    --report              # 同时生成 JSON 统计报告
```

### 摄像头实时检测

```bash
python predict.py --model best.pt --source 0 --show
```

---

## 评估

```bash
python evaluate.py \
    --model runs/pcm_defect_train/weights/best.pt \
    --data dataset/dataset.yaml \
    --split test \
    --save-report         # 生成混淆矩阵、PR曲线、JSON报告
```

输出指标示例：
```
  mAP@0.50:      0.8923
  mAP@0.50:0.95: 0.6741
  Precision:     0.8812
  Recall:        0.9034
  F1-Score:      0.8922
```

---

## 配置说明

编辑 `configs/defect_config.yaml` 调整：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `model.architecture` | 模型大小（n/s/m/l/x） | `yolo11s.pt` |
| `model.imgsz` | 训练图像尺寸 | `640` |
| `train.epochs` | 训练轮数 | `200` |
| `train.batch` | 批次大小（-1=自动） | `16` |
| `train.lr0` | 初始学习率 | `0.01` |
| `train.mosaic` | Mosaic 增强 | `1.0` |
| `inference.conf` | 检测置信度阈值 | `0.25` |
| `inference.iou` | NMS IoU 阈值 | `0.45` |

---

## 模型选型

| 模型 | 参数量 | 速度（T4 GPU） | 推荐场景 |
|------|--------|----------------|----------|
| yolo11n | 2.6M  | ~200 FPS | 边缘设备/实时性优先 |
| **yolo11s** | **9.4M**  | **~120 FPS** | **推荐：平衡精度与速度** |
| yolo11m | 20.1M | ~80 FPS  | 精度优先 |
| yolo11l | 25.3M | ~60 FPS  | 高精度离线检测 |
| yolo11x | 56.9M | ~40 FPS  | 最高精度 |

## 数据建议

| 项目 | 建议 |
|------|------|
| 最少数据量 | 每类 ≥ 200 张标注图像 |
| 推荐数据量 | 每类 ≥ 500 张（含合成增强）|
| 图像分辨率 | ≥ 640×640（与 imgsz 一致）|
| 类别平衡 | 各类别数量差异 < 5:1 |
| 数据划分 | train:val:test = 7:1.5:1.5 |

---

## 技术栈

- **模型**：YOLO11（Ultralytics 8.4.17）
- **框架**：PyTorch 2.x
- **增强**：Albumentations
- **可视化**：OpenCV + Matplotlib + Seaborn
