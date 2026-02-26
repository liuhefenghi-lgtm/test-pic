"""
PCM 彩板缺陷检测 - 可视化工具

功能：
- 在图像上绘制检测框、类别标签、置信度
- 保存单张/批量检测结果
- 生成检测结果汇总图（多图网格）
- 每类缺陷使用固定颜色编码
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 类别颜色定义（BGR 格式）
# ─────────────────────────────────────────────────────────────────────────────

# 每类缺陷固定颜色（直觉映射：红=危险/划伤，黄=警示/脏污，蓝=注意/起泡）
CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "scratch":     (0, 0, 255),      # 红色 - 划伤（最严重）
    "bump":        (0, 128, 255),    # 橙色 - 凸包
    "stain":       (0, 255, 255),    # 黄色 - 脏污
    "indentation": (255, 0, 255),    # 紫色 - 压痕
    "color_diff":  (0, 255, 0),      # 绿色 - 色差
    "blister":     (255, 0, 0),      # 蓝色 - 起泡
}

# 中文类别名
CLASS_NAMES_CN: Dict[str, str] = {
    "scratch":     "划伤",
    "bump":        "凸包",
    "stain":       "脏污",
    "indentation": "压痕",
    "color_diff":  "色差",
    "blister":     "起泡",
}

# 类别ID → 名称（0-5）
ID_TO_NAME: Dict[int, str] = {
    0: "scratch",
    1: "bump",
    2: "stain",
    3: "indentation",
    4: "color_diff",
    5: "blister",
}

# 默认回退颜色（白色）
DEFAULT_COLOR = (255, 255, 255)


def _get_color(class_name: str) -> Tuple[int, int, int]:
    """获取类别对应颜色。"""
    return CLASS_COLORS_BGR.get(class_name, DEFAULT_COLOR)


def _get_label_text(class_name: str, conf: float, show_cn: bool = True) -> str:
    """构造标注文字。"""
    cn = CLASS_NAMES_CN.get(class_name, class_name)
    if show_cn:
        return f"{cn}({class_name}) {conf:.2f}"
    return f"{class_name} {conf:.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# 核心绘制函数
# ─────────────────────────────────────────────────────────────────────────────

def draw_detections(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    class_ids: List[int],
    confidences: List[float],
    class_names: Optional[Dict[int, str]] = None,
    conf_threshold: float = 0.0,
    show_cn: bool = True,
    box_thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """
    在图像上绘制检测框和标签。

    Args:
        image:         输入图像 (H, W, 3) BGR
        boxes:         边界框列表 [(x1, y1, x2, y2), ...]（绝对像素坐标）
        class_ids:     类别 ID 列表
        confidences:   置信度列表
        class_names:   类别 ID → 名称映射（默认使用 ID_TO_NAME）
        conf_threshold: 低于此阈值的检测跳过（已过滤则设0）
        show_cn:       是否显示中文名称
        box_thickness: 边框线宽
        font_scale:    字体大小系数

    Returns:
        绘制了检测结果的图像副本
    """
    if class_names is None:
        class_names = ID_TO_NAME

    vis = image.copy()
    h, w = vis.shape[:2]

    for (x1, y1, x2, y2), cls_id, conf in zip(boxes, class_ids, confidences):
        if conf < conf_threshold:
            continue

        # 坐标裁剪
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))

        name = class_names.get(cls_id, f"class_{cls_id}")
        color = _get_color(name)
        label = _get_label_text(name, conf, show_cn=show_cn)

        # 绘制边框
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, box_thickness)

        # 计算标签背景区域
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        label_y = max(y1, lh + baseline + 4)

        # 半透明标签背景
        overlay = vis.copy()
        cv2.rectangle(
            overlay,
            (x1, label_y - lh - baseline - 4),
            (x1 + lw + 4, label_y),
            color,
            -1,
        )
        cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)

        # 文字（白色，确保可读性）
        cv2.putText(
            vis,
            label,
            (x1 + 2, label_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # 左上角汇总信息
    total = len(boxes)
    if total > 0:
        summary = f"Total defects: {total}"
        cv2.putText(vis, summary, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, summary, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    return vis


def draw_detections_from_ultralytics(
    image: np.ndarray,
    result,
    conf_threshold: float = 0.25,
    show_cn: bool = True,
) -> np.ndarray:
    """
    从 Ultralytics Result 对象直接绘制检测结果。

    Args:
        image:         原始图像 (H, W, 3) BGR
        result:        ultralytics.engine.results.Results 对象
        conf_threshold: 置信度阈值
        show_cn:       是否显示中文标签

    Returns:
        带标注的图像
    """
    boxes_data = result.boxes
    if boxes_data is None or len(boxes_data) == 0:
        return image.copy()

    boxes = boxes_data.xyxy.cpu().numpy().astype(int).tolist()
    class_ids = boxes_data.cls.cpu().numpy().astype(int).tolist()
    confidences = boxes_data.conf.cpu().numpy().tolist()
    class_names = result.names  # {id: name}

    return draw_detections(
        image,
        boxes,
        class_ids,
        confidences,
        class_names=class_names,
        conf_threshold=conf_threshold,
        show_cn=show_cn,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 保存工具
# ─────────────────────────────────────────────────────────────────────────────

def save_detection_result(
    image: np.ndarray,
    result,
    output_path: str,
    conf_threshold: float = 0.25,
    show_cn: bool = True,
) -> str:
    """
    将单张检测结果保存为图像文件。

    Args:
        image:         原始图像
        result:        ultralytics Results 对象
        output_path:   输出图像路径
        conf_threshold: 置信度阈值
        show_cn:       是否显示中文标签

    Returns:
        实际保存路径
    """
    vis = draw_detections_from_ultralytics(image, result, conf_threshold, show_cn)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, vis)
    return output_path


def save_batch_results(
    image_paths: List[str],
    results,
    output_dir: str,
    conf_threshold: float = 0.25,
    show_cn: bool = True,
) -> List[str]:
    """
    批量保存检测结果。

    Args:
        image_paths:   原始图像路径列表
        results:       ultralytics Results 列表（与 image_paths 对应）
        output_dir:    输出目录
        conf_threshold: 置信度阈值
        show_cn:       是否显示中文标签

    Returns:
        已保存文件路径列表
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for img_path, result in zip(image_paths, results):
        image = cv2.imread(img_path)
        if image is None:
            continue
        out_path = output_dir / Path(img_path).name
        save_detection_result(image, result, str(out_path), conf_threshold, show_cn)
        saved_paths.append(str(out_path))

    print(f"[INFO] {len(saved_paths)} 张结果已保存至 {output_dir}")
    return saved_paths


# ─────────────────────────────────────────────────────────────────────────────
# 汇总可视化
# ─────────────────────────────────────────────────────────────────────────────

def visualize_grid(
    image_paths: List[str],
    results=None,
    n_cols: int = 4,
    cell_size: Tuple[int, int] = (320, 320),
    save_path: Optional[str] = None,
    conf_threshold: float = 0.25,
    title: str = "PCM 缺陷检测结果",
) -> Optional[np.ndarray]:
    """
    生成多张检测结果的网格汇总图。

    Args:
        image_paths:   图像路径列表（最多显示 n_cols*4 张）
        results:       对应的 ultralytics Results 列表（None 则只显示原图）
        n_cols:        每行列数
        cell_size:     每格尺寸 (宽, 高)
        save_path:     保存路径（None 则返回图像 array）
        conf_threshold: 置信度阈值
        title:         图表标题

    Returns:
        网格图像 ndarray（若 save_path 为 None）
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("[WARN] matplotlib 未安装，跳过网格可视化")
        return None

    max_images = n_cols * 4  # 最多显示 4 行
    image_paths = image_paths[:max_images]
    n = len(image_paths)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    fig.suptitle(title, fontsize=14, y=1.01)

    if n_rows == 1:
        axes = [axes] if n_cols == 1 else [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, "__iter__") else [row])]

    for i, ax in enumerate(axes_flat):
        if i >= n:
            ax.axis("off")
            continue

        img = cv2.imread(image_paths[i])
        if img is None:
            ax.axis("off")
            continue

        if results is not None and i < len(results):
            vis = draw_detections_from_ultralytics(img, results[i], conf_threshold, show_cn=True)
        else:
            vis = img

        # BGR → RGB
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        ax.imshow(vis_rgb)
        ax.set_title(Path(image_paths[i]).name, fontsize=8)
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] 网格图已保存: {save_path}")
        plt.close()
        return None
    else:
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)


def draw_legend(
    img_size: Tuple[int, int] = (300, 220),
    class_names: Optional[Dict[int, str]] = None,
) -> np.ndarray:
    """
    生成缺陷类别图例图像。

    Args:
        img_size:    图例图像尺寸 (宽, 高)
        class_names: 类别 ID → 英文名映射

    Returns:
        图例图像 (H, W, 3) BGR
    """
    if class_names is None:
        class_names = ID_TO_NAME

    w, h = img_size
    legend = np.ones((h, w, 3), dtype=np.uint8) * 30  # 深灰背景

    cv2.putText(legend, "Defect Legend", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 1, cv2.LINE_AA)

    for i, (cls_id, name) in enumerate(sorted(class_names.items())):
        y = 55 + i * 28
        color = _get_color(name)
        cn = CLASS_NAMES_CN.get(name, name)
        # 色块
        cv2.rectangle(legend, (10, y - 14), (30, y + 2), color, -1)
        # 文字
        label = f"{cls_id}: {cn} ({name})"
        cv2.putText(legend, label, (38, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (220, 220, 220), 1, cv2.LINE_AA)

    return legend
