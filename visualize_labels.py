#!/usr/bin/env python3
"""
PCM 彩板缺陷检测 - 标注质量检查（QA）可视化工具

在训练前检查 LabelImg 打出的标注是否正确：
- 框的位置是否准确覆盖缺陷区域
- 类别是否选对（如凸包不要标成脏污）
- 是否有漏标

【用法】
  # 随机查看 train 集 16 张（保存为图片）
  python visualize_labels.py --data dataset/ --split train --n 16 --save

  # 只查看某类缺陷（0=划伤, 1=凸包, 2=脏污）
  python visualize_labels.py --data dataset/ --split train --class-id 0

  # 直接显示（需要图形界面）
  python visualize_labels.py --data dataset/ --split train --n 8 --show

  # 查看 val 集全部图像（保存为 grid）
  python visualize_labels.py --data dataset/ --split val --all --save
"""

import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.dataset import (
    CLASS_NAMES_CN,
    CLASS_NAMES_EN,
    ID_TO_NAME,
    load_yolo_label,
    bbox_yolo_to_xyxy,
)
from utils.visualization import CLASS_COLORS_BGR, CLASS_NAMES_CN as VIS_CN


# ─────────────────────────────────────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="PCM 缺陷标注 QA 可视化工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="数据集目录（含 images/ 和 labels/ 子目录）",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="要查看的数据子集",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=16,
        help="随机查看图像数量",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="查看该子集全部图像（覆盖 --n）",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        dest="class_id",
        help="只显示包含指定类别的图像（0=划伤 1=凸包 2=脏污 3=压痕 4=色差 5=起泡）",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="保存为网格图（runs/labels_vis/）",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="直接显示（需要图形界面）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/labels_vis",
        help="保存目录",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=400,
        help="网格中每张图的尺寸（像素）",
    )
    parser.add_argument(
        "--n-cols",
        type=int,
        default=4,
        help="网格列数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（None=不固定）",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 标注绘制
# ─────────────────────────────────────────────────────────────────────────────

def draw_ground_truth(
    image: np.ndarray,
    annotations: List[Tuple[int, float, float, float, float]],
    show_label: bool = True,
) -> np.ndarray:
    """
    在图像上绘制 YOLO 格式的真实标注框。

    Args:
        image:       原始图像 (H, W, 3) BGR
        annotations: [(class_id, cx, cy, w, h), ...] YOLO 格式
        show_label:  是否显示类别文字

    Returns:
        绘制了标注框的图像副本
    """
    h, w = image.shape[:2]
    vis = image.copy()

    if not annotations:
        # 无缺陷图像：右下角标记"无缺陷"
        cv2.putText(vis, "OK (no defect)", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 100), 1, cv2.LINE_AA)
        return vis

    for cls_id, cx, cy, bw, bh in annotations:
        x1, y1, x2, y2 = bbox_yolo_to_xyxy(cx, cy, bw, bh, w, h)

        name_en = ID_TO_NAME.get(cls_id, f"class_{cls_id}")
        name_cn = VIS_CN.get(name_en, name_en)
        color = CLASS_COLORS_BGR.get(name_en, (255, 255, 255))

        # 边框
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        if show_label:
            label = f"{cls_id}:{name_cn}"
            (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(y1, lh + bl + 2)
            # 标签背景
            overlay = vis.copy()
            cv2.rectangle(overlay, (x1, ty - lh - bl - 2), (x1 + lw + 4, ty), color, -1)
            cv2.addWeighted(overlay, 0.65, vis, 0.35, 0, vis)
            cv2.putText(vis, label, (x1 + 2, ty - bl),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # 左上角：缺陷数量汇总
    summary = f"Defects: {len(annotations)}"
    cv2.putText(vis, summary, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(vis, summary, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return vis


# ─────────────────────────────────────────────────────────────────────────────
# 网格图生成
# ─────────────────────────────────────────────────────────────────────────────

def make_grid(
    images: List[np.ndarray],
    annotations_list: List[List],
    image_names: List[str],
    n_cols: int = 4,
    cell_size: int = 400,
) -> np.ndarray:
    """
    将多张标注图拼成网格。

    Args:
        images:           原始图像列表
        annotations_list: 对应的标注列表
        image_names:      图像文件名（用于标题）
        n_cols:           列数
        cell_size:        每格像素尺寸（正方形）

    Returns:
        网格图像 (H, W, 3)
    """
    n = len(images)
    n_rows = (n + n_cols - 1) // n_cols

    canvas = np.ones((n_rows * cell_size, n_cols * cell_size, 3), dtype=np.uint8) * 40

    for idx, (img, annots, name) in enumerate(zip(images, annotations_list, image_names)):
        row, col = divmod(idx, n_cols)
        y0, x0 = row * cell_size, col * cell_size

        # 绘制标注
        vis = draw_ground_truth(img, annots)

        # 等比例缩放到 cell_size
        h, w = vis.shape[:2]
        scale = cell_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(vis, (new_w, new_h))

        # 居中放置
        pad_y = (cell_size - new_h) // 2
        pad_x = (cell_size - new_w) // 2
        canvas[y0 + pad_y: y0 + pad_y + new_h, x0 + pad_x: x0 + pad_x + new_w] = resized

        # 图像名称（右下角）
        short_name = Path(name).name[:25]
        cv2.putText(canvas, short_name,
                    (x0 + 4, y0 + cell_size - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # 顶部标题栏
    header = np.zeros((38, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, "PCM Defect Label QA - Ground Truth Visualization",
                (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)
    return np.vstack([header, canvas])


# ─────────────────────────────────────────────────────────────────────────────
# 图例
# ─────────────────────────────────────────────────────────────────────────────

def make_legend() -> np.ndarray:
    """生成类别图例（附加在网格右侧或单独保存）。"""
    w, h = 240, 220
    legend = np.ones((h, w, 3), dtype=np.uint8) * 35
    cv2.putText(legend, "Defect Classes", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1)
    for i, cls_id in enumerate(sorted(ID_TO_NAME.keys())):
        name_en = ID_TO_NAME[cls_id]
        name_cn = VIS_CN.get(name_en, name_en)
        color = CLASS_COLORS_BGR.get(name_en, (200, 200, 200))
        y = 50 + i * 26
        cv2.rectangle(legend, (10, y - 13), (28, y + 2), color, -1)
        cv2.putText(legend, f"{cls_id}: {name_cn}({name_en})",
                    (36, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    return legend


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def main():
    args = parse_args()

    data_dir = Path(args.data)
    img_dir = data_dir / "images" / args.split
    lbl_dir = data_dir / "labels" / args.split

    if not img_dir.exists():
        print(f"[ERROR] 图像目录不存在: {img_dir}")
        sys.exit(1)

    # 收集图像
    all_images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS])
    if not all_images:
        print(f"[ERROR] {img_dir} 中没有图像文件")
        sys.exit(1)

    # 过滤指定类别
    if args.class_id is not None:
        cls_id_filter = args.class_id
        name_cn = CLASS_NAMES_CN.get(cls_id_filter, str(cls_id_filter))
        filtered = []
        for img_path in all_images:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            annots = load_yolo_label(str(lbl_path))
            if any(a[0] == cls_id_filter for a in annots):
                filtered.append(img_path)
        print(f"[INFO] 过滤类别 {cls_id_filter}({name_cn})：找到 {len(filtered)}/{len(all_images)} 张")
        all_images = filtered

    if not all_images:
        print("[WARN] 没有符合条件的图像。")
        return

    # 采样
    if args.all:
        selected = all_images
    else:
        if args.seed is not None:
            random.seed(args.seed)
        n = min(args.n, len(all_images))
        selected = random.sample(all_images, n)
        selected.sort()

    print(f"[INFO] 显示 {len(selected)} 张图像（{args.split} 集，共 {len(all_images)} 张）")

    # 加载图像和标注
    images, annotations_list, names = [], [], []
    for img_path in selected:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] 无法读取图像: {img_path}")
            continue
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        annots = load_yolo_label(str(lbl_path))
        images.append(img)
        annotations_list.append(annots)
        names.append(img_path.name)

    if not images:
        print("[ERROR] 没有可用图像")
        return

    # 打印统计
    total_defects = sum(len(a) for a in annotations_list)
    from collections import Counter
    cls_counter: Counter = Counter()
    for annots in annotations_list:
        for cls_id, *_ in annots:
            cls_counter[cls_id] += 1

    print(f"\n  样本统计（选取的 {len(images)} 张）:")
    print(f"    缺陷总数: {total_defects}")
    for cls_id, count in sorted(cls_counter.items()):
        cn = CLASS_NAMES_CN.get(cls_id, str(cls_id))
        en = CLASS_NAMES_EN.get(cls_id, str(cls_id))
        print(f"    {cls_id}: {cn}({en}): {count}")

    # 生成网格图
    grid = make_grid(images, annotations_list, names, n_cols=args.n_cols, cell_size=args.cell_size)

    # 保存
    if args.save:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        grid_path = output_dir / f"labels_{args.split}.jpg"
        cv2.imwrite(str(grid_path), grid)
        print(f"\n[INFO] 网格图已保存: {grid_path}")

        legend = make_legend()
        legend_path = output_dir / "legend.png"
        cv2.imwrite(str(legend_path), legend)
        print(f"[INFO] 图例已保存: {legend_path}")

    # 显示
    if args.show:
        max_w = 1400
        if grid.shape[1] > max_w:
            scale = max_w / grid.shape[1]
            grid_show = cv2.resize(grid, (max_w, int(grid.shape[0] * scale)))
        else:
            grid_show = grid
        cv2.imshow("PCM Label QA - Press any key to close", grid_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if not args.save and not args.show:
        print("\n[HINT] 请添加 --save 保存图片，或 --show 直接显示（需图形界面）")


if __name__ == "__main__":
    main()
