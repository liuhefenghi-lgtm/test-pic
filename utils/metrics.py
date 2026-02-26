"""
PCM 彩板缺陷检测 - 评估指标工具

功能：
- mAP@0.5 / mAP@0.5:0.95 计算
- 每类 Precision / Recall / F1
- 混淆矩阵生成与可视化
- PR 曲线绘制
- JSON 评估报告生成
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# 类别中英文名
CLASS_NAMES_CN = {
    "scratch": "划伤",
    "bump": "凸包",
    "stain": "脏污",
    "indentation": "压痕",
    "color_diff": "色差",
    "blister": "起泡",
}


# ─────────────────────────────────────────────────────────────────────────────
# IoU 计算
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """
    计算两组矩形框之间的 IoU。

    Args:
        box1: (N, 4) [x1, y1, x2, y2]
        box2: (M, 4) [x1, y1, x2, y2]

    Returns:
        (N, M) IoU 矩阵
    """
    b1 = box1[:, None, :]  # (N, 1, 4)
    b2 = box2[None, :, :]  # (1, M, 4)

    inter_x1 = np.maximum(b1[..., 0], b2[..., 0])
    inter_y1 = np.maximum(b1[..., 1], b2[..., 1])
    inter_x2 = np.minimum(b1[..., 2], b2[..., 2])
    inter_y2 = np.minimum(b1[..., 3], b2[..., 3])

    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    union_area = area1[:, None] + area2[None, :] - inter_area
    iou = np.where(union_area > 0, inter_area / union_area, 0.0)
    return iou


# ─────────────────────────────────────────────────────────────────────────────
# AP 计算（VOC 11-point 或 101-point）
# ─────────────────────────────────────────────────────────────────────────────

def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    计算单类 AP（区域积分法，与 COCO 一致）。

    Args:
        recall:    召回率数组（升序）
        precision: 精确率数组

    Returns:
        AP 值
    """
    # 在两端补 0/1
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # 单调递减平滑
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 计算面积
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


# ─────────────────────────────────────────────────────────────────────────────
# 从预测/标注计算指标
# ─────────────────────────────────────────────────────────────────────────────

def compute_detection_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict],
    class_names: Dict[int, str],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.0,
) -> Dict:
    """
    计算检测指标（单 IoU 阈值）。

    Args:
        predictions: 预测列表，每个元素:
            {
                "image_id": str,
                "boxes": [[x1,y1,x2,y2], ...],
                "class_ids": [int, ...],
                "scores": [float, ...]
            }
        ground_truths: 真实标注列表，每个元素:
            {
                "image_id": str,
                "boxes": [[x1,y1,x2,y2], ...],
                "class_ids": [int, ...]
            }
        class_names:  {id: name}
        iou_threshold: IoU 匹配阈值
        conf_threshold: 置信度过滤阈值

    Returns:
        指标字典
    """
    # 按 image_id 索引 GT
    gt_by_image: Dict[str, Dict] = {d["image_id"]: d for d in ground_truths}

    # 按类别分类预测
    per_class_preds: Dict[int, List] = {cls_id: [] for cls_id in class_names}
    per_class_gts: Dict[int, int] = {cls_id: 0 for cls_id in class_names}

    for gt in ground_truths:
        for cls_id in gt["class_ids"]:
            per_class_gts[cls_id] = per_class_gts.get(cls_id, 0) + 1

    for pred in predictions:
        img_id = pred["image_id"]
        gt = gt_by_image.get(img_id, {"boxes": [], "class_ids": []})
        gt_boxes = np.array(gt["boxes"]) if gt["boxes"] else np.zeros((0, 4))
        gt_cls = np.array(gt["class_ids"])
        gt_matched = np.zeros(len(gt_cls), dtype=bool)

        # 按置信度排序
        scores = np.array(pred.get("scores", []))
        pred_boxes = np.array(pred.get("boxes", []))
        pred_cls = np.array(pred.get("class_ids", []))

        if len(scores) == 0:
            continue

        order = np.argsort(-scores)
        pred_boxes, pred_cls, scores = pred_boxes[order], pred_cls[order], scores[order]

        for box, cls_id, score in zip(pred_boxes, pred_cls, scores):
            if score < conf_threshold:
                continue
            if cls_id not in per_class_preds:
                per_class_preds[cls_id] = []

            tp = 0
            if len(gt_boxes) > 0:
                ious = compute_iou(box[None], gt_boxes)[0]
                # 只考虑同类别 GT
                same_class = gt_cls == cls_id
                ious = ious * same_class
                best_idx = int(np.argmax(ious))
                if ious[best_idx] >= iou_threshold and not gt_matched[best_idx]:
                    tp = 1
                    gt_matched[best_idx] = True

            per_class_preds[cls_id].append((score, tp))

    # 计算每类 AP
    class_metrics = {}
    aps = []

    for cls_id, name in class_names.items():
        preds = per_class_preds.get(cls_id, [])
        n_gt = per_class_gts.get(cls_id, 0)

        if n_gt == 0:
            continue

        if not preds:
            class_metrics[name] = {"AP": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
            aps.append(0.0)
            continue

        # 按置信度排序
        preds_sorted = sorted(preds, key=lambda x: -x[0])
        tp_cumsum = np.cumsum([p[1] for p in preds_sorted])
        fp_cumsum = np.cumsum([1 - p[1] for p in preds_sorted])

        recall_arr = tp_cumsum / (n_gt + 1e-8)
        precision_arr = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

        ap = compute_ap(recall_arr, precision_arr)
        aps.append(ap)

        final_p = float(precision_arr[-1]) if len(precision_arr) else 0.0
        final_r = float(recall_arr[-1]) if len(recall_arr) else 0.0
        f1 = 2 * final_p * final_r / (final_p + final_r + 1e-8)

        class_metrics[name] = {
            "AP": round(ap, 4),
            "precision": round(final_p, 4),
            "recall": round(final_r, 4),
            "f1": round(f1, 4),
            "n_gt": n_gt,
            "n_pred": len(preds),
        }

    map50 = float(np.mean(aps)) if aps else 0.0

    return {
        "mAP50": round(map50, 4),
        "iou_threshold": iou_threshold,
        "per_class": class_metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 从 Ultralytics 结果提取并格式化指标
# ─────────────────────────────────────────────────────────────────────────────

def extract_ultralytics_metrics(metrics, class_names: Dict[int, str]) -> Dict:
    """
    从 Ultralytics val() 返回的 metrics 对象提取结构化指标。

    Args:
        metrics:     ultralytics.utils.metrics.DetMetrics 对象
        class_names: {id: name}

    Returns:
        结构化指标字典
    """
    result = {}
    try:
        box = metrics.box
        result["overall"] = {
            "mAP50": round(float(box.map50), 4),
            "mAP50_95": round(float(box.map), 4),
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
        }
        # F1
        p, r = float(box.mp), float(box.mr)
        result["overall"]["f1"] = round(2 * p * r / (p + r + 1e-8), 4)

        result["per_class"] = {}
        for i, (p, r, ap50, ap) in enumerate(zip(box.p, box.r, box.ap50, box.ap)):
            name = class_names.get(i, f"class_{i}")
            name_cn = CLASS_NAMES_CN.get(name, name)
            f1 = 2 * float(p) * float(r) / (float(p) + float(r) + 1e-8)
            result["per_class"][name] = {
                "name_cn": name_cn,
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1": round(f1, 4),
                "AP50": round(float(ap50), 4),
                "AP50_95": round(float(ap), 4),
            }
    except Exception as e:
        result["error"] = str(e)

    return result


def print_metrics_table(metrics_dict: Dict) -> None:
    """打印格式化的指标表格。"""
    overall = metrics_dict.get("overall", {})
    per_class = metrics_dict.get("per_class", {})

    print("\n" + "=" * 72)
    print("  PCM 缺陷检测 - 评估报告")
    print("=" * 72)
    print(f"  {'指标':<25} {'值':>10}")
    print(f"  {'-'*35}")
    print(f"  {'mAP@0.50':<25} {overall.get('mAP50', 0):>10.4f}")
    print(f"  {'mAP@0.50:0.95':<25} {overall.get('mAP50_95', 0):>10.4f}")
    print(f"  {'Precision':<25} {overall.get('precision', 0):>10.4f}")
    print(f"  {'Recall':<25} {overall.get('recall', 0):>10.4f}")
    print(f"  {'F1-Score':<25} {overall.get('f1', 0):>10.4f}")

    if per_class:
        print(f"\n  {'类别':<20} {'P':>7} {'R':>7} {'F1':>7} {'AP50':>8} {'AP':>8}")
        print(f"  {'-'*58}")
        for name, cls_m in per_class.items():
            cn = cls_m.get("name_cn", name)
            label = f"{cn}({name})"
            print(f"  {label:<20} "
                  f"{cls_m.get('precision', 0):>7.4f} "
                  f"{cls_m.get('recall', 0):>7.4f} "
                  f"{cls_m.get('f1', 0):>7.4f} "
                  f"{cls_m.get('AP50', 0):>8.4f} "
                  f"{cls_m.get('AP50_95', 0):>8.4f}")
    print("=" * 72 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 混淆矩阵可视化
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True,
    title: str = "混淆矩阵",
) -> None:
    """
    绘制混淆矩阵热力图。

    Args:
        cm:          混淆矩阵 (N, N) 或 (N+1, N+1)（含 background）
        class_names: 类别名称列表
        save_path:   保存路径（None 则显示）
        normalize:   是否归一化（百分比）
        title:       图标题
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[WARN] matplotlib/seaborn 未安装，跳过混淆矩阵可视化")
        return

    if normalize:
        cm_plot = cm.astype(float)
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm_plot, row_sums, where=row_sums > 0)
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] 混淆矩阵已保存: {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# PR 曲线
# ─────────────────────────────────────────────────────────────────────────────

def plot_pr_curve(
    results: Dict,
    save_path: Optional[str] = None,
    title: str = "Precision-Recall 曲线",
) -> None:
    """
    绘制各类别的 PR 曲线。

    Args:
        results:   metrics_dict（包含 per_class 和 pr_curves 数据）
        save_path: 保存路径
        title:     标题
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib 未安装，跳过 PR 曲线")
        return

    pr_data = results.get("pr_curves", {})
    if not pr_data:
        print("[WARN] 无 PR 曲线数据，请确保 compute_detection_metrics 中包含 recall/precision 序列")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#e41a1c", "#ff7f00", "#ffff33", "#984ea3", "#4daf4a", "#377eb8"]

    for i, (name, data) in enumerate(pr_data.items()):
        recall = data["recall"]
        precision = data["precision"]
        ap = data.get("ap", 0.0)
        cn = CLASS_NAMES_CN.get(name, name)
        ax.plot(recall, precision, color=colors[i % len(colors)],
                label=f"{cn}({name}) AP={ap:.3f}", linewidth=2)

    ax.set_xlabel("Recall（召回率）")
    ax.set_ylabel("Precision（精确率）")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] PR 曲线已保存: {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# JSON 报告生成
# ─────────────────────────────────────────────────────────────────────────────

def generate_evaluation_report(
    metrics_dict: Dict,
    save_dir: str,
    model_name: str = "yolo11s",
    dataset_name: str = "PCM_Defect",
) -> str:
    """
    生成 JSON 格式评估报告。

    Args:
        metrics_dict: 指标字典
        save_dir:     报告保存目录
        model_name:   模型名称
        dataset_name: 数据集名称

    Returns:
        报告文件路径
    """
    import datetime

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "report_time": datetime.datetime.now().isoformat(),
        "model": model_name,
        "dataset": dataset_name,
        "metrics": metrics_dict,
    }

    report_path = save_dir / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 评估报告已保存: {report_path}")
    return str(report_path)
