#!/usr/bin/env python3
"""
PCM 彩板缺陷检测 - 训练脚本

使用 YOLO11（Ultralytics 8.4.x）进行迁移学习训练。

用法:
    # 基本训练（使用默认配置）
    python train.py --data dataset/dataset.yaml

    # 自定义参数
    python train.py \\
        --data dataset/dataset.yaml \\
        --config configs/defect_config.yaml \\
        --epochs 200 \\
        --batch 16 \\
        --device 0

    # 断点续训
    python train.py --data dataset/dataset.yaml --resume

    # 仅 CPU（调试）
    python train.py --data dataset/dataset.yaml --device cpu --epochs 5 --batch 4
"""

import argparse
import sys
from pathlib import Path

# 项目根路径
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="PCM 彩板缺陷检测 - YOLO11 训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="数据集 YAML 路径（dataset/dataset.yaml）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/defect_config.yaml",
        help="系统配置文件路径",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型权重路径（None=使用配置中的预训练权重）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="训练轮数（覆盖配置文件）",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="批次大小（-1=自动）",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="输入图像尺寸",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="训练设备（''=自动, 'cpu', '0', '0,1'）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从上次检查点断点续训",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="输出目录（覆盖配置）",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="实验名称（覆盖配置）",
    )
    parser.add_argument(
        "--verify-data",
        action="store_true",
        help="训练前验证数据集结构",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 1. 验证配置文件 ──────────────────────────────────────────────────────
    config_path = ROOT / args.config
    if not config_path.exists():
        print(f"[ERROR] 配置文件不存在: {config_path}")
        print(f"  请确认路径或使用默认配置: configs/defect_config.yaml")
        sys.exit(1)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] 数据集文件不存在: {data_path}")
        print("  请先创建数据集，参考 README.md 中的数据集准备说明")
        sys.exit(1)

    # ── 2. 可选：验证数据集结构 ──────────────────────────────────────────────
    if args.verify_data:
        from data.dataset import verify_dataset
        import yaml
        with open(data_path) as f:
            data_cfg = yaml.safe_load(f)
        dataset_root = data_cfg.get("path", str(data_path.parent))
        verify_dataset(dataset_root)

    # ── 3. 初始化检测器 ──────────────────────────────────────────────────────
    from models.detector import PCMDefectDetector
    detector = PCMDefectDetector(
        config_path=str(config_path),
        model_path=args.model,
    )

    # ── 4. 构建自定义训练参数 ────────────────────────────────────────────────
    train_kwargs = {}
    if args.imgsz is not None:
        train_kwargs["imgsz"] = args.imgsz
    if args.device is not None:
        train_kwargs["device"] = args.device
    if args.project is not None:
        train_kwargs["project"] = args.project
    if args.name is not None:
        train_kwargs["name"] = args.name

    # ── 5. 开始训练 ──────────────────────────────────────────────────────────
    try:
        results = detector.train(
            data_yaml=str(data_path),
            epochs=args.epochs,
            batch=args.batch,
            resume=args.resume,
            **train_kwargs,
        )

        print("\n[SUCCESS] 训练完成！")
        print("  最优权重保存于: runs/pcm_defect_train/weights/best.pt")
        print("  最后权重保存于: runs/pcm_defect_train/weights/last.pt")
        print("\n  下一步:")
        print("  1. 评估: python evaluate.py --model runs/pcm_defect_train/weights/best.pt --data dataset/dataset.yaml")
        print("  2. 推理: python predict.py  --model runs/pcm_defect_train/weights/best.pt --source /path/to/images")

    except KeyboardInterrupt:
        print("\n[INFO] 训练被用户中断。")
        print("  如需续训，请使用 --resume 参数重新运行。")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] 训练失败: {e}")
        raise


if __name__ == "__main__":
    main()
