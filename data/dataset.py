"""
PCM 彩板缺陷检测 - 数据集工具

功能：
- 数据集结构验证
- YOLO 格式 dataset.yaml 生成
- 类别分布统计与可视化
- 训练/验证/测试集划分
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# 类别定义
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES_EN = {
    0: "scratch",
    1: "bump",
    2: "stain",
    3: "indentation",
    4: "color_diff",
    5: "blister",
}

CLASS_NAMES_CN = {
    0: "划伤",
    1: "凸包",
    2: "脏污",
    3: "压痕",
    4: "色差",
    5: "起泡",
}


# ─────────────────────────────────────────────────────────────────────────────
# 数据集验证
# ─────────────────────────────────────────────────────────────────────────────

def verify_dataset(data_dir: str, splits: Tuple[str, ...] = ("train", "val", "test")) -> Dict:
    """
    验证数据集结构完整性。

    数据集预期结构:
        data_dir/
        ├── images/
        │   ├── train/  *.jpg/*.png
        │   ├── val/
        │   └── test/
        └── labels/
            ├── train/  *.txt (YOLO格式)
            ├── val/
            └── test/

    Args:
        data_dir: 数据集根目录
        splits: 要验证的子集列表

    Returns:
        包含统计信息的字典
    """
    data_dir = Path(data_dir)
    stats = {}

    print(f"\n{'='*60}")
    print(f"  数据集验证: {data_dir}")
    print(f"{'='*60}")

    total_images = 0
    total_labels = 0
    class_counts = defaultdict(int)

    for split in splits:
        img_dir = data_dir / "images" / split
        lbl_dir = data_dir / "labels" / split

        if not img_dir.exists():
            print(f"  [WARN] {split} 图像目录不存在: {img_dir}")
            continue

        # 收集图像文件
        img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        images = [p for p in img_dir.iterdir() if p.suffix.lower() in img_extensions]

        missing_labels = []
        invalid_labels = []
        split_class_counts = defaultdict(int)

        for img_path in images:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                missing_labels.append(img_path.name)
                continue

            # 解析 YOLO 格式标注
            try:
                with open(lbl_path) as f:
                    lines = f.read().strip().splitlines()
                for line in lines:
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        invalid_labels.append(str(lbl_path))
                        break
                    cls_id = int(parts[0])
                    split_class_counts[cls_id] += 1
                    class_counts[cls_id] += 1
            except Exception as e:
                invalid_labels.append(f"{lbl_path}: {e}")

        split_stats = {
            "images": len(images),
            "labels": len(images) - len(missing_labels),
            "missing_labels": len(missing_labels),
            "invalid_labels": len(invalid_labels),
            "class_distribution": dict(split_class_counts),
        }
        stats[split] = split_stats
        total_images += len(images)
        total_labels += split_stats["labels"]

        # 打印分割统计
        print(f"\n  [{split.upper()}]")
        print(f"    图像数量: {len(images)}")
        print(f"    有标注:   {split_stats['labels']}")
        if missing_labels:
            print(f"    缺失标注: {len(missing_labels)}")
        if invalid_labels:
            print(f"    无效标注: {len(invalid_labels)}")
        print(f"    类别分布:")
        for cls_id, count in sorted(split_class_counts.items()):
            name_cn = CLASS_NAMES_CN.get(cls_id, f"class_{cls_id}")
            name_en = CLASS_NAMES_EN.get(cls_id, f"class_{cls_id}")
            print(f"      {cls_id}: {name_cn}({name_en}) = {count} 个缺陷")

    print(f"\n  [总计] 图像: {total_images}, 有标注: {total_labels}")
    print(f"\n  [全局类别分布]")
    for cls_id, count in sorted(class_counts.items()):
        name_cn = CLASS_NAMES_CN.get(cls_id, f"class_{cls_id}")
        pct = 100 * count / max(1, sum(class_counts.values()))
        bar = "█" * int(pct / 2)
        print(f"    {cls_id} {name_cn}: {count:>6} ({pct:5.1f}%) {bar}")

    stats["total"] = {
        "images": total_images,
        "labels": total_labels,
        "class_distribution": dict(class_counts),
    }
    print(f"{'='*60}\n")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# dataset.yaml 生成
# ─────────────────────────────────────────────────────────────────────────────

def create_dataset_yaml(
    data_dir: str,
    output_path: Optional[str] = None,
    class_names: Optional[Dict[int, str]] = None,
) -> str:
    """
    生成 YOLO11 所需的 dataset.yaml 文件。

    Args:
        data_dir: 数据集根目录（绝对路径）
        output_path: yaml 输出路径（默认放在 data_dir/dataset.yaml）
        class_names: 类别名称字典 {id: name}，默认使用英文名

    Returns:
        生成的 yaml 文件路径
    """
    data_dir = Path(data_dir).resolve()
    if output_path is None:
        output_path = data_dir / "dataset.yaml"

    if class_names is None:
        class_names = CLASS_NAMES_EN

    yaml_content = {
        "path": str(data_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": {int(k): str(v) for k, v in class_names.items()},
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"[INFO] dataset.yaml 已生成: {output_path}")
    return str(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 数据集划分工具
# ─────────────────────────────────────────────────────────────────────────────

def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, int]:
    """
    将原始标注图像按比例划分为 train/val/test。

    source_dir 预期结构:
        source_dir/
        ├── images/  *.jpg
        └── labels/  *.txt

    Args:
        source_dir: 原始数据目录
        output_dir: 输出目录（将创建 YOLO 格式子目录）
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子

    Returns:
        各分割的图像数量
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio > 0, "train_ratio + val_ratio 必须小于 1.0"

    # 收集所有有标注的图像
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    img_dir = source_dir / "images"
    lbl_dir = source_dir / "labels"

    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in img_extensions:
            continue
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))

    random.seed(seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }

    counts = {}
    for split, split_pairs in splits.items():
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img_src, lbl_src in split_pairs:
            shutil.copy2(img_src, output_dir / "images" / split / img_src.name)
            shutil.copy2(lbl_src, output_dir / "labels" / split / lbl_src.name)
        counts[split] = len(split_pairs)
        print(f"[INFO] {split}: {len(split_pairs)} 张图像")

    print(f"[INFO] 数据集划分完成，输出目录: {output_dir}")
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# YOLO 格式标注工具
# ─────────────────────────────────────────────────────────────────────────────

def bbox_xyxy_to_yolo(
    x1: float, y1: float, x2: float, y2: float,
    img_w: int, img_h: int,
) -> Tuple[float, float, float, float]:
    """将绝对坐标 (x1,y1,x2,y2) 转换为 YOLO 归一化格式 (cx,cy,w,h)。"""
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def bbox_yolo_to_xyxy(
    cx: float, cy: float, w: float, h: float,
    img_w: int, img_h: int,
) -> Tuple[int, int, int, int]:
    """将 YOLO 归一化格式 (cx,cy,w,h) 转换为绝对坐标 (x1,y1,x2,y2)。"""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2


def save_yolo_label(
    label_path: str,
    annotations: List[Tuple[int, float, float, float, float]],
) -> None:
    """
    保存 YOLO 格式标注文件。

    Args:
        label_path: 输出 .txt 文件路径
        annotations: [(class_id, cx, cy, w, h), ...]（均为归一化值）
    """
    with open(label_path, "w") as f:
        for cls_id, cx, cy, w, h in annotations:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def load_yolo_label(
    label_path: str,
) -> List[Tuple[int, float, float, float, float]]:
    """
    加载 YOLO 格式标注文件。

    Returns:
        [(class_id, cx, cy, w, h), ...]
    """
    annotations = []
    path = Path(label_path)
    if not path.exists():
        return annotations
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                annotations.append((cls_id, cx, cy, w, h))
    return annotations


# ─────────────────────────────────────────────────────────────────────────────
# 数据集统计可视化
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(
    data_dir: str,
    splits: Tuple[str, ...] = ("train", "val", "test"),
    save_path: Optional[str] = None,
) -> None:
    """
    绘制数据集类别分布柱状图。

    Args:
        data_dir: 数据集根目录
        splits: 要统计的子集
        save_path: 图像保存路径，None 则显示
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams["font.family"] = ["DejaVu Sans", "SimHei", "sans-serif"]
    except ImportError:
        print("[WARN] matplotlib 未安装，跳过可视化")
        return

    data_dir = Path(data_dir)
    split_counts: Dict[str, Dict[int, int]] = {}

    for split in splits:
        lbl_dir = data_dir / "labels" / split
        if not lbl_dir.exists():
            continue
        counts = defaultdict(int)
        for lbl_file in lbl_dir.glob("*.txt"):
            for cls_id, *_ in load_yolo_label(str(lbl_file)):
                counts[cls_id] += 1
        split_counts[split] = dict(counts)

    if not split_counts:
        print("[WARN] 未找到任何标注数据")
        return

    class_ids = sorted(set(k for d in split_counts.values() for k in d.keys()))
    x = np.arange(len(class_ids))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#4C8BF5", "#F4B400", "#0F9D58"]

    for i, (split, counts) in enumerate(split_counts.items()):
        values = [counts.get(cls_id, 0) for cls_id in class_ids]
        bars = ax.bar(x + i * width, values, width, label=split, color=colors[i % len(colors)], alpha=0.85)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        str(val), ha="center", va="bottom", fontsize=8)

    labels = [f"{CLASS_NAMES_CN.get(cls_id, cls_id)}\n({CLASS_NAMES_EN.get(cls_id, cls_id)})"
              for cls_id in class_ids]
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("缺陷数量")
    ax.set_title("PCM 彩板缺陷类别分布")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] 类别分布图已保存: {save_path}")
    else:
        plt.show()
    plt.close()
