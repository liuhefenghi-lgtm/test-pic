#!/usr/bin/env python3
"""
PCM 彩板缺陷检测 - 数据集准备脚本

将 LabelImg 打好标记的原始图像一键整理为 YOLO 训练格式：
  1. 扫描并报告标注完整性（有多少图有标注）
  2. 自动划分 train / val / test
  3. 生成 dataset.yaml
  4. 验证最终数据集结构
  5. 输出类别分布图

【使用前准备】
  raw_data/
  ├── classes.txt        ← LabelImg 类别文件（已提供，见 raw_data/classes.txt）
  ├── images/
  │   ├── pcm_001.jpg
  │   └── ...
  └── labels/
      ├── pcm_001.txt    ← LabelImg 输出的 YOLO 格式标注
      └── ...

【典型用法】
  # 基本（7:1.5:1.5 划分）
  python prepare_dataset.py --source raw_data/ --output dataset/

  # 自定义比例（更多验证集）
  python prepare_dataset.py --source raw_data/ --output dataset/ --train 0.7 --val 0.2

  # 仅验证标注格式，不复制文件
  python prepare_dataset.py --source raw_data/ --validate-only

  # 覆盖已有 dataset/ 目录
  python prepare_dataset.py --source raw_data/ --output dataset/ --overwrite
"""

import argparse
import shutil
import sys
from pathlib import Path

# ── 项目路径 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.dataset import (
    CLASS_NAMES_EN,
    CLASS_NAMES_CN,
    split_dataset,
    create_dataset_yaml,
    verify_dataset,
    plot_class_distribution,
    load_yolo_label,
)


# ─────────────────────────────────────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="PCM 缺陷数据集准备工具（LabelImg → YOLO 训练格式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="原始数据目录（含 images/ 和 labels/ 子目录）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset",
        help="输出数据集目录（默认: dataset/）",
    )
    parser.add_argument(
        "--train",
        type=float,
        default=0.7,
        help="训练集比例（默认: 0.7）",
    )
    parser.add_argument(
        "--val",
        type=float,
        default=0.15,
        help="验证集比例（默认: 0.15，剩余为测试集）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="划分随机种子（保证可复现，默认: 42）",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="仅验证原始标注格式，不执行划分和复制",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有输出目录（默认: 不覆盖，报错退出）",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="不生成类别分布图",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 原始数据扫描与验证
# ─────────────────────────────────────────────────────────────────────────────

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VALID_CLASS_IDS = set(CLASS_NAMES_EN.keys())


def scan_raw_data(source_dir: Path) -> dict:
    """
    扫描原始数据目录，统计图像/标注情况。

    Args:
        source_dir: 原始数据目录

    Returns:
        统计信息字典
    """
    img_dir = source_dir / "images"
    lbl_dir = source_dir / "labels"

    if not img_dir.exists():
        print(f"[ERROR] 图像目录不存在: {img_dir}")
        print(f"  请将图像放入 {img_dir}/ 目录")
        sys.exit(1)

    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS])
    n_images = len(images)

    if n_images == 0:
        print(f"[ERROR] 未在 {img_dir} 中找到任何图像文件")
        sys.exit(1)

    has_label = 0
    missing_label = []
    invalid_annotations = []
    class_counts = {cls_id: 0 for cls_id in VALID_CLASS_IDS}
    unknown_classes = set()

    for img_path in images:
        lbl_path = lbl_dir / (img_path.stem + ".txt") if lbl_dir.exists() else None
        if lbl_path is None or not lbl_path.exists():
            missing_label.append(img_path.name)
            continue

        has_label += 1
        annots = load_yolo_label(str(lbl_path))

        if not annots:
            # 允许空标注（该图无缺陷），但提示
            pass

        for cls_id, cx, cy, w, h in annots:
            if cls_id not in VALID_CLASS_IDS:
                unknown_classes.add(cls_id)
                invalid_annotations.append(f"{lbl_path.name}: 未知类别 ID={cls_id}")
            else:
                class_counts[cls_id] += 1
            # 检查坐标范围
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                invalid_annotations.append(
                    f"{lbl_path.name}: 坐标超出范围 ({cls_id} {cx:.3f} {cy:.3f} {w:.3f} {h:.3f})"
                )

    return {
        "images": images,
        "n_images": n_images,
        "has_label": has_label,
        "missing_label": missing_label,
        "invalid_annotations": invalid_annotations,
        "class_counts": class_counts,
        "unknown_classes": unknown_classes,
        "lbl_dir_exists": lbl_dir.exists(),
    }


def print_scan_report(stats: dict, source_dir: Path) -> None:
    """打印原始数据扫描报告。"""
    print("\n" + "=" * 62)
    print("  原始数据扫描报告")
    print("=" * 62)
    print(f"  来源目录:      {source_dir}")
    print(f"  图像总数:      {stats['n_images']}")
    print(f"  有标注图像:    {stats['has_label']}")
    print(f"  缺失标注:      {len(stats['missing_label'])}")

    if stats["missing_label"]:
        pct = 100 * len(stats["missing_label"]) / stats["n_images"]
        print(f"  [WARN] {len(stats['missing_label'])} 张图像缺少标注文件（占 {pct:.1f}%）")
        if len(stats["missing_label"]) <= 10:
            for name in stats["missing_label"]:
                print(f"    - {name}")
        else:
            for name in stats["missing_label"][:5]:
                print(f"    - {name}")
            print(f"    ... 等 {len(stats['missing_label'])} 张")

    if stats["invalid_annotations"]:
        print(f"\n  [ERROR] 发现 {len(stats['invalid_annotations'])} 条异常标注：")
        for msg in stats["invalid_annotations"][:10]:
            print(f"    - {msg}")
        if len(stats["invalid_annotations"]) > 10:
            print(f"    ... 等 {len(stats['invalid_annotations'])} 条")

    if stats["unknown_classes"]:
        print(f"\n  [ERROR] 未知类别 ID: {sorted(stats['unknown_classes'])}")
        print(f"  有效类别 ID: 0-5（共 6 类，见 raw_data/classes.txt）")

    total_defects = sum(stats["class_counts"].values())
    print(f"\n  缺陷标注总数:  {total_defects}")
    print(f"\n  类别分布:")
    for cls_id in sorted(CLASS_NAMES_EN.keys()):
        count = stats["class_counts"].get(cls_id, 0)
        cn = CLASS_NAMES_CN[cls_id]
        en = CLASS_NAMES_EN[cls_id]
        pct = 100 * count / max(1, total_defects)
        bar = "█" * int(pct / 3)
        print(f"    {cls_id}: {cn}({en}): {count:>6} ({pct:5.1f}%) {bar}")

    print("=" * 62)


def check_data_quality(stats: dict) -> bool:
    """
    检查数据质量，返回是否可以继续。

    Returns:
        True = 可以继续；False = 有严重问题需要修复
    """
    ok = True

    if not stats["lbl_dir_exists"]:
        print("\n[ERROR] labels/ 目录不存在。请先用 LabelImg 进行标注。")
        ok = False

    if stats["has_label"] == 0:
        print("\n[ERROR] 没有任何带标注的图像，无法继续。")
        ok = False

    if stats["unknown_classes"]:
        print(f"\n[ERROR] 存在未知类别 ID {sorted(stats['unknown_classes'])}，请检查 LabelImg 的 classes.txt。")
        ok = False

    if stats["has_label"] < 30:
        print(f"\n[WARN] 有效标注图像仅 {stats['has_label']} 张，建议至少 30 张才能训练。")
        print("  （每类缺陷建议 ≥ 200 张，推荐 ≥ 500 张）")
        # 不阻止，只警告

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    source_dir = Path(args.source).resolve()
    output_dir = Path(args.output).resolve()

    if not source_dir.exists():
        print(f"[ERROR] 原始数据目录不存在: {source_dir}")
        sys.exit(1)

    print(f"\n{'='*62}")
    print("  PCM 彩板缺陷检测 - 数据集准备工具")
    print(f"{'='*62}")

    # ── Step 1: 扫描原始数据 ─────────────────────────────────────────────────
    print("\n[Step 1/4] 扫描原始数据...")
    stats = scan_raw_data(source_dir)
    print_scan_report(stats, source_dir)

    if not check_data_quality(stats):
        print("\n[ERROR] 数据质量检查未通过，请修复上述问题后重试。")
        sys.exit(1)

    if args.validate_only:
        print("\n[INFO] --validate-only 模式，仅验证不复制，流程结束。")
        return

    # ── Step 2: 检查输出目录 ─────────────────────────────────────────────────
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            print(f"\n[ERROR] 输出目录已存在且非空: {output_dir}")
            print("  使用 --overwrite 参数覆盖，或指定不同的 --output 路径。")
            sys.exit(1)
        print(f"\n[WARN] 覆盖已有目录: {output_dir}")
        shutil.rmtree(output_dir)

    # ── Step 3: 划分数据集 ───────────────────────────────────────────────────
    val_ratio = args.val
    test_ratio = 1.0 - args.train - val_ratio
    if test_ratio <= 0:
        print(f"\n[ERROR] train({args.train}) + val({val_ratio}) >= 1.0，请调整比例。")
        sys.exit(1)

    print(f"\n[Step 2/4] 划分数据集 (train={args.train:.0%} / val={val_ratio:.0%} / test={test_ratio:.0%})...")
    counts = split_dataset(
        source_dir=str(source_dir),
        output_dir=str(output_dir),
        train_ratio=args.train,
        val_ratio=val_ratio,
        seed=args.seed,
    )
    print(f"  train: {counts.get('train', 0)} 张")
    print(f"  val:   {counts.get('val', 0)} 张")
    print(f"  test:  {counts.get('test', 0)} 张")

    # ── Step 4: 生成 dataset.yaml ────────────────────────────────────────────
    print("\n[Step 3/4] 生成 dataset.yaml...")
    yaml_path = create_dataset_yaml(
        data_dir=str(output_dir),
        output_path=str(output_dir / "dataset.yaml"),
        class_names=CLASS_NAMES_EN,
    )

    # ── Step 5: 验证最终数据集 ───────────────────────────────────────────────
    print("\n[Step 4/4] 验证最终数据集结构...")
    verify_dataset(str(output_dir), splits=("train", "val", "test"))

    # ── 可选：类别分布图 ─────────────────────────────────────────────────────
    if not args.no_plot:
        try:
            dist_path = str(output_dir / "class_distribution.png")
            plot_class_distribution(
                data_dir=str(output_dir),
                splits=("train", "val", "test"),
                save_path=dist_path,
            )
        except Exception as e:
            print(f"[WARN] 类别分布图生成失败（可能缺少 matplotlib）: {e}")

    # ── 完成提示 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  数据集准备完成！")
    print("=" * 62)
    print(f"  数据集目录: {output_dir}")
    print(f"  YAML 文件:  {yaml_path}")
    print(f"\n  下一步 - 可选：标注质量检查")
    print(f"    python visualize_labels.py --data {output_dir}/ --split train --n 16 --save")
    print(f"\n  下一步 - 开始训练")
    print(f"    python train.py --data {yaml_path}")
    print(f"    python train.py --data {yaml_path} --epochs 200 --batch 16 --device 0")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
