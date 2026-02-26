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
│   └── defect_config.yaml      # 数据集、模型、训练、推理配置
├── data/
│   ├── augmentation.py         # 金属表面专用数据增强管线
│   └── dataset.py              # 数据集验证、YAML生成、类别统计
├── models/
│   └── detector.py             # PCMDefectDetector（YOLO11封装）
├── utils/
│   ├── visualization.py        # 检测结果可视化（带颜色编码边框）
│   └── metrics.py              # mAP/Precision/Recall/F1/混淆矩阵
├── raw_data/                   # ← 将自己的图像放这里打标记
│   ├── classes.txt             # LabelImg 类别文件（已配置好，勿改顺序）
│   ├── images/                 # 放原始 PCM 图像
│   └── labels/                 # LabelImg 自动保存标注到此
├── prepare_dataset.py          # 打完标记后一键准备训练数据集
├── visualize_labels.py         # 标注质量检查（QA）工具
├── train.py                    # 训练入口
├── predict.py                  # 推理入口
├── evaluate.py                 # 评估入口
├── demo.py                     # 快速演示（无需真实数据）
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

## 自行打标记 → 训练（完整流程）

### 总体步骤

```
Step 1: 采集 PCM 图像   →   放入 raw_data/images/
Step 2: LabelImg 打标记  →   标注保存到 raw_data/labels/
Step 3: 一键准备数据集   →   python prepare_dataset.py ...
Step 4: 检查标注质量     →   python visualize_labels.py ...
Step 5: 开始训练         →   python train.py ...
Step 6: 评估/推理        →   python evaluate.py / predict.py
```

---

### Step 1: 采集图像

将 PCM 彩板图像（`.jpg` / `.png`）放入：
```
raw_data/images/
```

> **建议**：每类缺陷至少 200 张，推荐 500 张以上。图像分辨率 ≥ 640×640。

---

### Step 2: 用 LabelImg 打标记

**安装 LabelImg**：
```bash
pip install labelImg
labelImg
```

**配置步骤**（只需配置一次）：

| 步骤 | 操作 |
|------|------|
| 1 | 菜单 → `Change Save Dir` → 选择 `raw_data/labels/` |
| 2 | 菜单 → `Open Dir` → 选择 `raw_data/images/` |
| 3 | 菜单 → `View` → 勾选 **YOLO** 格式 |
| 4 | 勾选左侧 **Auto Save** |

**打标记操作**：

| 按键 | 功能 |
|------|------|
| `W`  | 画矩形框（拖动） |
| `D`  | 下一张图 |
| `A`  | 上一张图 |
| `Ctrl+S` | 保存标注 |
| `Del` | 删除选中的框 |

**`raw_data/classes.txt` 已配置好**（与类别 ID 严格对应，**勿修改顺序**）：
```
scratch       ← ID 0：划伤
bump          ← ID 1：凸包
stain         ← ID 2：脏污
indentation   ← ID 3：压痕
color_diff    ← ID 4：色差
blister       ← ID 5：起泡
```

**各类缺陷打框技巧**：

- **划伤**：框住整条划痕含两端，细划伤可加 5~10px 余量
- **凸包**：框住亮斑及周围轻微形变区（通常椭圆形）
- **脏污**：框住整块污迹含浅色边缘；多块分散则各自画框
- **压痕**：框住内凹区域及周边阴影
- **色差**：框住颜色明显异常的整块区域
- **起泡**：框住气泡隆起部分及周边变形区

---

### Step 3: 一键准备数据集

```bash
# 基本用法（7:1.5:1.5 划分）
python prepare_dataset.py --source raw_data/ --output dataset/

# 自定义划分比例（更多验证集）
python prepare_dataset.py --source raw_data/ --output dataset/ --train 0.7 --val 0.2

# 仅验证标注格式，不复制文件
python prepare_dataset.py --source raw_data/ --validate-only

# 覆盖已有数据集目录
python prepare_dataset.py --source raw_data/ --output dataset/ --overwrite
```

脚本会自动：
1. 检查标注完整性（哪些图缺标注、类别 ID 是否正确）
2. 随机划分 train / val / test
3. 生成 `dataset/dataset.yaml`
4. 打印类别分布统计

---

### Step 4: 检查标注质量（可选但推荐）

在训练前目视抽检标注是否正确：

```bash
# 查看 train 集 16 张（保存为图片）
python visualize_labels.py --data dataset/ --split train --n 16 --save

# 只看划伤类（class-id=0）
python visualize_labels.py --data dataset/ --split train --class-id 0 --n 20 --save

# 直接弹窗显示（需图形界面）
python visualize_labels.py --data dataset/ --split train --n 8 --show
```

结果保存于 `runs/labels_vis/labels_train.jpg`，打开确认：
- 每个缺陷都有框
- 框的颜色与类别匹配（红=划伤，橙=凸包，黄=脏污...）
- 无明显漏标或错标

---

## 数据集准备

### 目录结构（YOLO 格式，由 prepare_dataset.py 自动生成）

```
dataset/
├── dataset.yaml       # 数据集描述（自动生成）
├── class_distribution.png  # 类别分布图
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

### 其他标注工具

- [LabelImg](https://github.com/HumanSignal/labelImg)（本地，免费，**推荐**）
- [Roboflow](https://roboflow.com)（在线，支持团队协作）
- [CVAT](https://cvat.ai)（在线，支持视频标注）

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
