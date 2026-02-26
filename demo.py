#!/usr/bin/env python3
"""
PCM 彩板缺陷检测 - 快速演示脚本

无需真实数据集，通过合成图像演示完整检测流程：
1. 用 OpenCV 合成模拟 PCM 板表面缺陷图像（划伤/凸包/脏污）
2. 创建临时 YOLO 格式数据集
3. 用 YOLO11n（最轻量）快速 finetune 5 个 epoch
4. 对合成测试图进行推理并可视化结果
5. 清理临时文件（可选）

用法:
    # 完整演示（训练 + 推理）
    python demo.py

    # 跳过训练，直接使用已有权重推理
    python demo.py --skip-train --model runs/pcm_defect_train/weights/best.pt

    # 保留临时数据集
    python demo.py --keep-data

    # 指定设备
    python demo.py --device cpu
"""

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# 合成图像生成
# ─────────────────────────────────────────────────────────────────────────────

# 缺陷类别（演示用 3 类简化版）
DEMO_CLASSES = {
    0: "scratch",    # 划伤
    1: "bump",       # 凸包
    2: "stain",      # 脏污
}
DEMO_CLASSES_CN = {
    0: "划伤",
    1: "凸包",
    2: "脏污",
}


def make_pcm_background(width: int = 640, height: int = 640) -> np.ndarray:
    """
    生成模拟 PCM 彩板底色（蓝灰色金属纹理）。

    模拟：均匀浅蓝灰色 + 轻微金属拉丝纹理
    """
    # 基础蓝灰色
    base_color = np.array([185, 175, 160], dtype=np.float32)  # BGR 浅蓝灰
    img = np.full((height, width, 3), base_color, dtype=np.float32)

    # 轻微随机纹理（模拟金属辊轧纹）
    noise = np.random.normal(0, 6, (height, width)).astype(np.float32)
    # 水平方向拉丝（低频）
    for i in range(0, height, np.random.randint(15, 35)):
        stripe_val = np.random.uniform(-8, 8)
        img[i : i + 2, :] += stripe_val

    for c in range(3):
        img[:, :, c] += noise
    img = np.clip(img, 0, 255).astype(np.uint8)

    # 轻微高斯模糊，使纹理更自然
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    return img


def add_scratch(
    img: np.ndarray,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    添加划伤缺陷（细长暗色线条）。

    Returns:
        (修改后图像, (cx, cy, w, h) YOLO 归一化坐标)
    """
    h, img_w = img.shape[:2]
    # 划伤起止点
    x0 = rng.randint(50, img_w - 50)
    y0 = rng.randint(50, h - 50)
    length = rng.randint(80, 250)
    angle = rng.uniform(-30, 30)  # 近水平划伤（生产方向）
    dx = int(length * np.cos(np.radians(angle)))
    dy = int(length * np.sin(np.radians(angle)))
    x1_s, y1_s = x0, y0
    x2_s, y2_s = x0 + dx, y0 + dy

    thickness = rng.randint(1, 3)
    color = tuple(int(c * rng.uniform(0.4, 0.65)) for c in img[y0, x0].tolist())
    cv2.line(img, (x1_s, y1_s), (x2_s, y2_s), color, thickness, cv2.LINE_AA)

    # 计算 YOLO bbox（含 padding）
    pad = 8
    bx1 = max(0, min(x1_s, x2_s) - pad)
    by1 = max(0, min(y1_s, y2_s) - pad)
    bx2 = min(img_w - 1, max(x1_s, x2_s) + pad)
    by2 = min(h - 1, max(y1_s, y2_s) + pad)
    cx = (bx1 + bx2) / 2 / img_w
    cy = (by1 + by2) / 2 / h
    bw = (bx2 - bx1) / img_w
    bh = (by2 - by1) / h
    return img, (cx, cy, bw, bh)


def add_bump(
    img: np.ndarray,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    添加凸包缺陷（椭圆亮斑，模拟表面凸起反光）。

    Returns:
        (修改后图像, YOLO 归一化坐标)
    """
    h, w = img.shape[:2]
    cx_px = rng.randint(80, w - 80)
    cy_px = rng.randint(80, h - 80)
    rx = rng.randint(15, 45)
    ry = rng.randint(10, 35)

    # 创建亮斑 mask
    y_idx, x_idx = np.ogrid[:h, :w]
    mask = ((x_idx - cx_px) / rx) ** 2 + ((y_idx - cy_px) / ry) ** 2 <= 1.0

    # 中心亮，边缘渐暗（高斯分布）
    dist = ((x_idx - cx_px) / rx) ** 2 + ((y_idx - cy_px) / ry) ** 2
    intensity = np.exp(-dist * 1.5) * rng.uniform(40, 80)
    for c in range(3):
        img[:, :, c] = np.clip(
            img[:, :, c].astype(np.float32) + intensity * mask,
            0, 255,
        ).astype(np.uint8)

    # YOLO bbox
    pad = 5
    bx1 = max(0, cx_px - rx - pad)
    by1 = max(0, cy_px - ry - pad)
    bx2 = min(w - 1, cx_px + rx + pad)
    by2 = min(h - 1, cy_px + ry + pad)
    norm_cx = (bx1 + bx2) / 2 / w
    norm_cy = (by1 + by2) / 2 / h
    norm_w = (bx2 - bx1) / w
    norm_h = (by2 - by1) / h
    return img, (norm_cx, norm_cy, norm_w, norm_h)


def add_stain(
    img: np.ndarray,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    添加脏污缺陷（不规则暗色斑块）。

    Returns:
        (修改后图像, YOLO 归一化坐标)
    """
    h, w = img.shape[:2]
    cx_px = rng.randint(80, w - 80)
    cy_px = rng.randint(80, h - 80)
    r = rng.randint(20, 55)

    # 随机多边形近似不规则污迹
    n_pts = rng.randint(6, 12)
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    radii = r * rng.uniform(0.5, 1.0, size=n_pts)
    pts = np.column_stack([
        cx_px + radii * np.cos(angles),
        cy_px + radii * np.sin(angles),
    ]).astype(np.int32)

    stain_color = tuple(
        int(c * rng.uniform(0.5, 0.75)) for c in img[cy_px, cx_px].tolist()
    )
    cv2.fillPoly(img, [pts], stain_color)
    # 边缘模糊
    img = cv2.GaussianBlur(img, (7, 7), 2.0)

    # YOLO bbox
    pad = 5
    bx1 = max(0, cx_px - r - pad)
    by1 = max(0, cy_px - r - pad)
    bx2 = min(w - 1, cx_px + r + pad)
    by2 = min(h - 1, cy_px + r + pad)
    norm_cx = (bx1 + bx2) / 2 / w
    norm_cy = (by1 + by2) / 2 / h
    norm_w = (bx2 - bx1) / w
    norm_h = (by2 - by1) / h
    return img, (norm_cx, norm_cy, norm_w, norm_h)


def synthesize_defect_image(
    seed: int = 0,
    width: int = 640,
    height: int = 640,
    n_defects: int = None,
) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
    """
    合成带缺陷的 PCM 板图像及 YOLO 标注。

    Args:
        seed:      随机种子（确保可复现）
        width:     图像宽度
        height:    图像高度
        n_defects: 缺陷数量（None=随机 1-3）

    Returns:
        (图像 ndarray, [(class_id, cx, cy, w, h), ...])
    """
    rng = np.random.RandomState(seed)
    img = make_pcm_background(width, height)

    if n_defects is None:
        n_defects = rng.randint(1, 4)

    annotations = []
    defect_funcs = [add_scratch, add_bump, add_stain]
    class_ids = rng.choice(len(defect_funcs), size=n_defects, replace=True)

    for cls_id in class_ids:
        fn = defect_funcs[cls_id]
        img, bbox = fn(img, rng)
        # 过滤极小 bbox
        if bbox[2] > 0.01 and bbox[3] > 0.01:
            annotations.append((int(cls_id), *bbox))

    return img, annotations


# ─────────────────────────────────────────────────────────────────────────────
# 临时数据集创建
# ─────────────────────────────────────────────────────────────────────────────

def create_demo_dataset(
    output_dir: Path,
    n_train: int = 60,
    n_val: int = 15,
    n_test: int = 15,
) -> str:
    """
    创建合成演示数据集（YOLO 格式）。

    Args:
        output_dir: 数据集根目录
        n_train:    训练集图像数
        n_val:      验证集图像数
        n_test:     测试集图像数

    Returns:
        dataset.yaml 路径
    """
    import yaml

    print(f"[INFO] 正在生成合成数据集: {output_dir}")
    splits = {"train": n_train, "val": n_val, "test": n_test}
    base_seed = 0

    for split, n in splits.items():
        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n):
            seed = base_seed + i
            img, annots = synthesize_defect_image(seed=seed)
            img_path = img_dir / f"pcm_{split}_{i:04d}.jpg"
            lbl_path = lbl_dir / f"pcm_{split}_{i:04d}.txt"
            cv2.imwrite(str(img_path), img)
            with open(lbl_path, "w") as f:
                for cls_id, cx, cy, w, h in annots:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        base_seed += n
        print(f"  [{split}] {n} 张图像生成完毕")

    # dataset.yaml
    yaml_content = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(DEMO_CLASSES),
        "names": DEMO_CLASSES,
    }
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        import yaml as _yaml
        _yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)

    print(f"[INFO] dataset.yaml 已生成: {yaml_path}")
    return str(yaml_path)


# ─────────────────────────────────────────────────────────────────────────────
# 推理可视化
# ─────────────────────────────────────────────────────────────────────────────

def visualize_predictions(
    images: List[np.ndarray],
    results,
    class_names: dict,
    save_path: str,
) -> None:
    """将多张推理结果拼成网格图保存。"""
    from utils.visualization import draw_detections_from_ultralytics, CLASS_NAMES_CN

    n_cols = min(4, len(images))
    n_rows = (len(images) + n_cols - 1) // n_cols
    cell_h, cell_w = 320, 320

    canvas = np.zeros((n_rows * cell_h, n_cols * cell_w, 3), dtype=np.uint8)

    for idx, (img, result) in enumerate(zip(images, results)):
        row, col = divmod(idx, n_cols)
        vis = draw_detections_from_ultralytics(img, result, conf_threshold=0.1, show_cn=True)
        cell = cv2.resize(vis, (cell_w, cell_h))
        canvas[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w] = cell

    # 标题
    header = np.zeros((50, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, "PCM Surface Defect Detection - YOLO11 Demo",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    final = np.vstack([header, canvas])

    cv2.imwrite(save_path, final)
    print(f"[INFO] 推理结果网格图已保存: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="PCM 缺陷检测演示（无需真实数据）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="跳过训练，使用 --model 指定的权重直接推理",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="（skip-train 时）已训练模型权重路径",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="演示训练轮数（建议 5-20）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="设备（''=自动, 'cpu', '0'）",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="演示结束后保留合成数据集",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=60,
        help="合成训练图像数量",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./runs/demo",
        help="演示结果输出目录",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 检查依赖 ─────────────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics 未安装！请运行:")
        print("  pip install ultralytics==8.4.17")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  PCM 彩板缺陷检测 - YOLO11 快速演示")
    print("=" * 60)
    print("  缺陷类别: 划伤(scratch) / 凸包(bump) / 脏污(stain)")
    print("  模型:     yolo11n.pt（最轻量，演示用）")
    print("=" * 60 + "\n")

    # ── 1. 生成合成数据集 ────────────────────────────────────────────────────
    tmp_dir = Path(tempfile.mkdtemp(prefix="pcm_demo_"))
    dataset_yaml = create_demo_dataset(
        tmp_dir, n_train=args.n_train, n_val=15, n_test=15
    )

    best_model_path = args.model

    # ── 2. 训练（可选）──────────────────────────────────────────────────────
    if not args.skip_train:
        print(f"\n[INFO] 开始 YOLO11n 快速训练（{args.epochs} epochs）...")
        print("[INFO] 注意：演示训练仅验证流程，生产环境请使用 yolo11s.pt 训练 200+ epochs\n")

        t0 = time.time()
        model = YOLO("yolo11n.pt")
        train_params = {
            "data": dataset_yaml,
            "epochs": args.epochs,
            "imgsz": 320,        # 小尺寸加速演示
            "batch": 8,
            "project": str(output_dir / "train"),
            "name": "demo",
            "exist_ok": True,
            "verbose": False,
            "plots": False,
        }
        if args.device:
            train_params["device"] = args.device

        results = model.train(**train_params)
        elapsed = time.time() - t0

        # 查找最优权重
        best_model_path = str(output_dir / "train" / "demo" / "weights" / "best.pt")
        if not Path(best_model_path).exists():
            best_model_path = str(output_dir / "train" / "demo" / "weights" / "last.pt")

        print(f"\n[INFO] 训练完成，耗时 {elapsed:.1f}s")
        print(f"[INFO] 最优权重: {best_model_path}")

    # ── 3. 推理演示 ──────────────────────────────────────────────────────────
    print(f"\n[INFO] 加载模型进行推理: {best_model_path}")
    if best_model_path is None or not Path(best_model_path).exists():
        print("[WARN] 未找到训练权重，使用 yolo11n.pt 原始权重演示推理流程")
        best_model_path = "yolo11n.pt"

    infer_model = YOLO(best_model_path)

    # 合成测试图像
    print("[INFO] 生成 8 张测试图像...")
    test_images = []
    for i in range(8):
        img, _ = synthesize_defect_image(seed=1000 + i, n_defects=np.random.randint(1, 4))
        test_images.append(img)

    # 保存测试图像
    test_dir = output_dir / "test_images"
    test_dir.mkdir(exist_ok=True)
    test_paths = []
    for i, img in enumerate(test_images):
        p = test_dir / f"test_{i:02d}.jpg"
        cv2.imwrite(str(p), img)
        test_paths.append(str(p))

    # 执行推理
    print("[INFO] 执行推理...")
    t0 = time.time()
    results = infer_model.predict(
        source=str(test_dir),
        conf=0.1,           # 演示用低阈值（模型未充分训练）
        save=False,
        verbose=False,
    )
    elapsed = time.time() - t0
    fps = len(results) / elapsed
    print(f"[INFO] 推理完成: {len(results)} 张, {elapsed:.2f}s, {fps:.1f} FPS")

    # ── 4. 可视化输出 ────────────────────────────────────────────────────────
    grid_path = str(output_dir / "demo_result.jpg")
    visualize_predictions(test_images, results, infer_model.names, grid_path)

    # 打印缺陷统计
    total_defects = sum(len(r.boxes) if r.boxes else 0 for r in results)
    print(f"\n[INFO] 演示结果统计:")
    print(f"  测试图像: {len(results)} 张")
    print(f"  检测缺陷: {total_defects} 处")
    print(f"  结果图:   {grid_path}")

    # ── 5. 清理（可选）──────────────────────────────────────────────────────
    if not args.keep_data:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[INFO] 临时数据集已清理: {tmp_dir}")
    else:
        print(f"[INFO] 演示数据集保留于: {tmp_dir}")

    print("\n" + "=" * 60)
    print("  演示完成！")
    print("=" * 60)
    print("\n  下一步（使用真实数据）:")
    print("  1. 准备标注数据（推荐使用 LabelImg 或 Roboflow）")
    print("  2. 按 YOLO 格式组织数据集（参见 README.md）")
    print("  3. 运行: python train.py --data dataset/dataset.yaml")
    print("  4. 评估: python evaluate.py --model runs/pcm_defect_train/weights/best.pt --data dataset/dataset.yaml")
    print("  5. 推理: python predict.py  --model best.pt --source /path/to/pcm_image.jpg\n")


if __name__ == "__main__":
    main()
