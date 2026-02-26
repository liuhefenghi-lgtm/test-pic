#!/usr/bin/env python3
"""
PCM 彩板缺陷检测 - 推理脚本

支持：单张图片 / 目录批量 / 视频 / 摄像头

用法:
    # 单张图片
    python predict.py --model best.pt --source image.jpg

    # 批量目录
    python predict.py --model best.pt --source /path/to/images/

    # 视频
    python predict.py --model best.pt --source video.mp4

    # 摄像头
    python predict.py --model best.pt --source 0

    # 调整置信度，不保存结果
    python predict.py --model best.pt --source image.jpg --conf 0.4 --no-save

    # 保存带标注的结果 + 统计报告
    python predict.py --model best.pt --source /images/ --output ./results --report
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import List

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="PCM 彩板缺陷检测 - 推理",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型权重路径（.pt 文件）",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="推理来源：图片路径/目录/视频/摄像头索引(0)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/defect_config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="置信度阈值（默认使用配置文件中的值）",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="NMS IoU 阈值",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="推理图像尺寸",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./runs/predict",
        help="结果保存目录",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存标注结果图",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="保存 YOLO 格式预测标注文件",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="实时显示检测结果（需要 GUI）",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="生成 JSON 统计报告",
    )
    parser.add_argument(
        "--no-cn",
        action="store_true",
        help="不显示中文类别名称",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="推理设备（''=自动, 'cpu', '0'）",
    )
    return parser.parse_args()


def collect_image_paths(source: str) -> List[str]:
    """收集图像文件路径列表。"""
    src = Path(source)
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    if src.is_file() and src.suffix.lower() in img_exts:
        return [str(src)]
    elif src.is_dir():
        paths = [str(p) for p in sorted(src.iterdir()) if p.suffix.lower() in img_exts]
        return paths
    return []


def generate_report(results, output_dir: str, class_names: dict) -> str:
    """生成 JSON 统计报告。"""
    from utils.visualization import CLASS_NAMES_CN

    output_dir = Path(output_dir)
    stats = {
        "total_images": len(results),
        "total_defects": 0,
        "defect_free_images": 0,
        "class_counts": defaultdict(int),
        "class_counts_cn": {},
        "per_image": [],
    }

    for result in results:
        img_path = str(result.path)
        boxes = result.boxes

        n_defects = len(boxes) if boxes is not None else 0
        stats["total_defects"] += n_defects
        if n_defects == 0:
            stats["defect_free_images"] += 1

        img_stat = {
            "image": Path(img_path).name,
            "n_defects": n_defects,
            "defects": [],
        }
        if boxes is not None and len(boxes) > 0:
            for cls_id, conf in zip(
                boxes.cls.cpu().numpy().astype(int),
                boxes.conf.cpu().numpy(),
            ):
                name = class_names.get(int(cls_id), f"class_{cls_id}")
                stats["class_counts"][name] += 1
                img_stat["defects"].append({
                    "class": name,
                    "class_cn": CLASS_NAMES_CN.get(name, name),
                    "confidence": round(float(conf), 4),
                })

        stats["per_image"].append(img_stat)

    # 添加中文名
    for name, count in stats["class_counts"].items():
        from utils.visualization import CLASS_NAMES_CN as CN
        stats["class_counts_cn"][CN.get(name, name)] = count
    stats["class_counts"] = dict(stats["class_counts"])

    report_path = output_dir / "predict_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return str(report_path)


def print_summary(results, class_names: dict) -> None:
    """打印推理结果汇总。"""
    from utils.visualization import CLASS_NAMES_CN

    total_imgs = len(results)
    total_defects = 0
    class_counts = defaultdict(int)

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for cls_id in boxes.cls.cpu().numpy().astype(int):
                name = class_names.get(int(cls_id), f"class_{cls_id}")
                class_counts[name] += 1
                total_defects += 1

    defect_free = sum(1 for r in results if r.boxes is None or len(r.boxes) == 0)

    print("\n" + "=" * 55)
    print("  推理结果汇总")
    print("=" * 55)
    print(f"  总图像数:    {total_imgs}")
    print(f"  无缺陷图像:  {defect_free} ({100*defect_free/max(1,total_imgs):.1f}%)")
    print(f"  有缺陷图像:  {total_imgs - defect_free}")
    print(f"  缺陷总数:    {total_defects}")
    if class_counts:
        print(f"\n  缺陷类型分布:")
        for name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            cn = CLASS_NAMES_CN.get(name, name)
            pct = 100 * count / max(1, total_defects)
            bar = "█" * int(pct / 4)
            print(f"    {cn}({name}): {count:>5} ({pct:5.1f}%) {bar}")
    print("=" * 55 + "\n")


def main():
    args = parse_args()

    # 验证模型文件
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] 模型文件不存在: {model_path}")
        sys.exit(1)

    # 初始化检测器
    config_path = ROOT / args.config if Path(args.config).is_relative_to(ROOT) or not Path(args.config).is_absolute() else Path(args.config)
    config_path = ROOT / args.config

    from models.detector import PCMDefectDetector
    detector = PCMDefectDetector.load(
        model_path=str(model_path),
        config_path=str(config_path) if config_path.exists() else None,
    )

    # 推理参数
    predict_kwargs = {
        "imgsz": args.imgsz,
        "save": not args.no_save,
        "save_txt": args.save_txt,
        "show": args.show,
        "output_dir": args.output,
        "verbose": False,
    }
    if args.device:
        predict_kwargs["device"] = args.device

    print(f"\n[INFO] 开始推理: {args.source}")
    t0 = time.time()

    results = detector.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        **predict_kwargs,
    )

    elapsed = time.time() - t0
    n = len(results)
    fps = n / elapsed if elapsed > 0 else 0
    print(f"[INFO] 推理完成: {n} 张图像, 耗时 {elapsed:.2f}s, {fps:.1f} FPS")

    if n == 0:
        print("[WARN] 未检测到任何图像，请检查 --source 路径")
        return

    # 获取类别名
    class_names = results[0].names if results else {}

    # 打印汇总
    print_summary(results, class_names)

    # 保存 JSON 报告
    if args.report:
        report_path = generate_report(results, args.output, class_names)
        print(f"[INFO] JSON 报告已保存: {report_path}")

    if not args.no_save:
        print(f"[INFO] 结果图像已保存至: {args.output}/")


if __name__ == "__main__":
    main()
