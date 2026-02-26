#!/usr/bin/env python3
"""
PCM 彩板缺陷检测 - 评估脚本

在测试集上评估训练好的模型，输出：
- mAP@0.50, mAP@0.50:0.95
- 每类 Precision / Recall / F1
- 混淆矩阵（可视化）
- PR 曲线
- JSON 报告

用法:
    # 在测试集评估
    python evaluate.py --model runs/pcm_defect_train/weights/best.pt --data dataset/dataset.yaml

    # 在验证集评估
    python evaluate.py --model best.pt --data dataset/dataset.yaml --split val

    # 自定义阈值
    python evaluate.py --model best.pt --data dataset/dataset.yaml --conf 0.3 --iou 0.5

    # 保存完整报告（含混淆矩阵/PR曲线）
    python evaluate.py --model best.pt --data dataset/dataset.yaml --save-report
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="PCM 彩板缺陷检测 - 模型评估",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="训练好的模型权重路径（.pt）",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="数据集 YAML 路径",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/defect_config.yaml",
        help="系统配置文件路径",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="评估数据子集",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="置信度阈值",
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
        help="评估图像尺寸",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./runs/evaluate",
        help="评估结果保存目录",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="保存 JSON 报告 + 可视化（混淆矩阵、PR 曲线）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="评估设备（''=自动, 'cpu', '0'）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 验证文件 ─────────────────────────────────────────────────────────────
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] 模型文件不存在: {model_path}")
        sys.exit(1)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] 数据集文件不存在: {data_path}")
        sys.exit(1)

    config_path = ROOT / args.config

    # ── 初始化检测器 ─────────────────────────────────────────────────────────
    from models.detector import PCMDefectDetector
    detector = PCMDefectDetector.load(
        model_path=str(model_path),
        config_path=str(config_path) if config_path.exists() else None,
    )

    # ── 评估参数 ─────────────────────────────────────────────────────────────
    eval_kwargs = {
        "imgsz": args.imgsz,
        "project": args.output,
        "plots": args.save_report,
        "save_json": args.save_report,
    }
    if args.device:
        eval_kwargs["device"] = args.device

    print(f"\n[INFO] 开始评估 ({args.split} 集): {data_path}")
    print(f"[INFO] 模型: {model_path}")

    # ── 执行评估 ─────────────────────────────────────────────────────────────
    metrics = detector.evaluate(
        data_yaml=str(data_path),
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        **eval_kwargs,
    )

    # ── 提取并展示指标 ───────────────────────────────────────────────────────
    class_names = metrics.names if hasattr(metrics, "names") else {}

    from utils.metrics import extract_ultralytics_metrics, print_metrics_table
    metrics_dict = extract_ultralytics_metrics(metrics, class_names)
    print_metrics_table(metrics_dict)

    # ── 生成完整报告 ─────────────────────────────────────────────────────────
    if args.save_report:
        from utils.metrics import generate_evaluation_report
        report_path = generate_evaluation_report(
            metrics_dict=metrics_dict,
            save_dir=args.output,
            model_name=model_path.stem,
            dataset_name=data_path.stem,
        )
        print(f"[INFO] 完整评估报告已保存: {report_path}")
        print(f"[INFO] 可视化图表已保存至: {args.output}/")
        print(f"       - 混淆矩阵: {args.output}/confusion_matrix.png")
        print(f"       - PR 曲线:  {args.output}/PR_curve.png")
        print(f"       - F1 曲线:  {args.output}/F1_curve.png")


if __name__ == "__main__":
    main()
