"""
PCM 彩板缺陷检测器

封装 YOLO11（Ultralytics 8.4.x）提供：
- 训练（支持断点续训）
- 推理（单图/批量/视频）
- 评估（测试集完整指标）
- 导出（ONNX/TensorRT）
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# 主检测器类
# ─────────────────────────────────────────────────────────────────────────────

class PCMDefectDetector:
    """
    PCM 彩板表面缺陷检测器。

    封装 YOLO11，针对金属缺陷检测场景调优。

    用法:
        # 从配置文件初始化（训练新模型）
        detector = PCMDefectDetector(config_path="configs/defect_config.yaml")
        detector.train("dataset/dataset.yaml")

        # 从已训练权重加载（推理）
        detector = PCMDefectDetector.load("runs/pcm_defect_train/weights/best.pt")
        results = detector.predict("path/to/image.jpg")
    """

    # 缺陷类别（英文 → 中文）
    CLASS_NAMES_CN = {
        "scratch": "划伤",
        "bump": "凸包",
        "stain": "脏污",
        "indentation": "压痕",
        "color_diff": "色差",
        "blister": "起泡",
    }

    def __init__(
        self,
        config_path: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        """
        初始化检测器。

        Args:
            config_path: 配置文件路径（configs/defect_config.yaml）
            model_path:  已训练模型路径（.pt 文件）；若提供则优先加载
        """
        self.config = {}
        if config_path:
            self._load_config(config_path)

        self.model = None
        self._model_path = model_path

        # 延迟导入（避免在没有 GPU 的机器上启动报错）
        self._yolo_class = None

    def _load_config(self, config_path: str) -> None:
        """加载 YAML 配置文件。"""
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def _get_yolo(self):
        """延迟导入 Ultralytics YOLO。"""
        if self._yolo_class is None:
            try:
                from ultralytics import YOLO
                self._yolo_class = YOLO
            except ImportError:
                raise ImportError(
                    "请安装 ultralytics: pip install ultralytics==8.4.17"
                )
        return self._yolo_class

    def _init_model(self, model_path: Optional[str] = None) -> None:
        """初始化或加载模型。"""
        YOLO = self._get_yolo()
        if model_path:
            self.model = YOLO(model_path)
        elif self._model_path:
            self.model = YOLO(self._model_path)
        else:
            # 使用配置文件中指定的架构（COCO 预训练）
            arch = self.config.get("model", {}).get("architecture", "yolo11s.pt")
            self.model = YOLO(arch)

    # ─────────────────────────────────────────────────────────────────────
    # 训练
    # ─────────────────────────────────────────────────────────────────────

    def train(
        self,
        data_yaml: str,
        epochs: Optional[int] = None,
        batch: Optional[int] = None,
        resume: bool = False,
        **kwargs,
    ) -> Dict:
        """
        启动训练。

        Args:
            data_yaml: dataset.yaml 路径
            epochs:    训练轮数（覆盖配置文件）
            batch:     批次大小（覆盖配置文件）
            resume:    是否从最后检查点续训
            **kwargs:  其他 ultralytics train() 参数

        Returns:
            训练结果指标字典
        """
        self._init_model()

        train_cfg = self.config.get("train", {})
        model_cfg = self.config.get("model", {})
        output_cfg = self.config.get("output", {})

        train_params = {
            "data": str(Path(data_yaml).resolve()),
            "epochs": epochs or train_cfg.get("epochs", 200),
            "batch": batch or train_cfg.get("batch", 16),
            "imgsz": model_cfg.get("imgsz", 640),
            "workers": train_cfg.get("workers", 8),
            "optimizer": train_cfg.get("optimizer", "SGD"),
            "lr0": train_cfg.get("lr0", 0.01),
            "lrf": train_cfg.get("lrf", 0.01),
            "momentum": train_cfg.get("momentum", 0.937),
            "weight_decay": train_cfg.get("weight_decay", 0.0005),
            "warmup_epochs": train_cfg.get("warmup_epochs", 3.0),
            "warmup_momentum": train_cfg.get("warmup_momentum", 0.8),
            "warmup_bias_lr": train_cfg.get("warmup_bias_lr", 0.1),
            "box": train_cfg.get("box", 7.5),
            "cls": train_cfg.get("cls", 0.5),
            "dfl": train_cfg.get("dfl", 1.5),
            "patience": train_cfg.get("patience", 50),
            "save_period": train_cfg.get("save_period", 10),
            "mosaic": train_cfg.get("mosaic", 1.0),
            "mixup": train_cfg.get("mixup", 0.1),
            "copy_paste": train_cfg.get("copy_paste", 0.1),
            "degrees": train_cfg.get("degrees", 5.0),
            "translate": train_cfg.get("translate", 0.1),
            "scale": train_cfg.get("scale", 0.5),
            "shear": train_cfg.get("shear", 2.0),
            "flipud": train_cfg.get("flipud", 0.0),
            "fliplr": train_cfg.get("fliplr", 0.5),
            "hsv_h": train_cfg.get("hsv_h", 0.015),
            "hsv_s": train_cfg.get("hsv_s", 0.7),
            "hsv_v": train_cfg.get("hsv_v", 0.4),
            "erasing": train_cfg.get("erasing", 0.4),
            "project": output_cfg.get("project", "./runs"),
            "name": output_cfg.get("train_name", "pcm_defect_train"),
            "exist_ok": True,
            "resume": resume,
            "pretrained": model_cfg.get("pretrained", True),
            "verbose": True,
        }

        # 设备配置
        device = train_cfg.get("device", "")
        if device:
            train_params["device"] = device

        # 用户自定义参数覆盖
        train_params.update(kwargs)

        print("\n" + "=" * 60)
        print("  PCM 缺陷检测 - YOLO11 训练启动")
        print("=" * 60)
        print(f"  数据集: {train_params['data']}")
        print(f"  轮数:   {train_params['epochs']}")
        print(f"  批次:   {train_params['batch']}")
        print(f"  图像:   {train_params['imgsz']}×{train_params['imgsz']}")
        print(f"  输出:   {train_params['project']}/{train_params['name']}")
        print("=" * 60 + "\n")

        results = self.model.train(**train_params)
        return results

    # ─────────────────────────────────────────────────────────────────────
    # 推理
    # ─────────────────────────────────────────────────────────────────────

    def predict(
        self,
        source: Union[str, List[str]],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        save: bool = True,
        save_txt: bool = False,
        show: bool = False,
        output_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        对图像/目录/视频进行缺陷检测推理。

        Args:
            source:     图像路径、目录、视频路径或摄像头索引(0)
            conf:       置信度阈值（覆盖配置）
            iou:        NMS IoU 阈值
            save:       是否保存带标注结果图
            save_txt:   是否保存 YOLO 格式预测标注
            show:       是否显示实时预测（需要 GUI）
            output_dir: 结果保存目录
            **kwargs:   其他 ultralytics predict() 参数

        Returns:
            ultralytics Results 列表
        """
        if self.model is None:
            self._init_model()

        infer_cfg = self.config.get("inference", {})
        output_cfg = self.config.get("output", {})

        predict_params = {
            "source": source,
            "conf": conf or infer_cfg.get("conf", 0.25),
            "iou": iou or infer_cfg.get("iou", 0.45),
            "max_det": infer_cfg.get("max_det", 100),
            "save": save,
            "save_txt": save_txt,
            "show": show,
            "project": output_dir or output_cfg.get("project", "./runs"),
            "name": output_cfg.get("predict_name", "pcm_defect_predict"),
            "exist_ok": True,
            "verbose": False,
        }
        predict_params.update(kwargs)

        results = self.model.predict(**predict_params)
        return results

    # ─────────────────────────────────────────────────────────────────────
    # 评估
    # ─────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        data_yaml: str,
        split: str = "test",
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        **kwargs,
    ) -> Dict:
        """
        在指定数据子集上评估模型性能。

        Args:
            data_yaml: dataset.yaml 路径
            split:     评估子集（"train"/"val"/"test"）
            conf:      置信度阈值
            iou:       IoU 阈值
            **kwargs:  其他 ultralytics val() 参数

        Returns:
            指标字典（包含 mAP/Precision/Recall/F1）
        """
        if self.model is None:
            self._init_model()

        infer_cfg = self.config.get("inference", {})
        model_cfg = self.config.get("model", {})
        output_cfg = self.config.get("output", {})

        val_params = {
            "data": str(Path(data_yaml).resolve()),
            "split": split,
            "conf": conf or infer_cfg.get("conf", 0.25),
            "iou": iou or infer_cfg.get("iou", 0.45),
            "imgsz": model_cfg.get("imgsz", 640),
            "project": output_cfg.get("project", "./runs"),
            "name": output_cfg.get("eval_name", "pcm_defect_eval"),
            "exist_ok": True,
            "verbose": True,
            "plots": True,           # 生成混淆矩阵、PR 曲线
            "save_json": True,       # COCO JSON 格式结果
        }
        val_params.update(kwargs)

        print(f"\n[INFO] 开始评估 ({split} 集)...")
        metrics = self.model.val(**val_params)
        self._print_metrics(metrics)
        return metrics

    def _print_metrics(self, metrics) -> None:
        """格式化打印评估指标。"""
        print("\n" + "=" * 60)
        print("  评估结果")
        print("=" * 60)
        try:
            box = metrics.box
            print(f"  mAP@0.50:      {box.map50:.4f}")
            print(f"  mAP@0.50:0.95: {box.map:.4f}")
            print(f"  Precision:     {box.mp:.4f}")
            print(f"  Recall:        {box.mr:.4f}")
            print(f"\n  各类别指标:")
            for i, (p, r, ap50, ap) in enumerate(
                zip(box.p, box.r, box.ap50, box.ap)
            ):
                name = metrics.names.get(i, f"class_{i}")
                name_cn = self.CLASS_NAMES_CN.get(name, name)
                print(f"    {i}: {name_cn}({name})")
                print(f"         P={p:.4f}  R={r:.4f}  AP@50={ap50:.4f}  AP={ap:.4f}")
        except Exception:
            print(f"  原始结果: {metrics}")
        print("=" * 60 + "\n")

    # ─────────────────────────────────────────────────────────────────────
    # 导出
    # ─────────────────────────────────────────────────────────────────────

    def export(
        self,
        format: str = "onnx",
        output_path: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        导出模型用于生产部署。

        Args:
            format:      导出格式（"onnx"/"engine"/"coreml"/"tflite"...）
            output_path: 导出文件路径（默认与权重同目录）
            **kwargs:    其他导出参数（opset/half/dynamic 等）

        Returns:
            导出文件路径
        """
        if self.model is None:
            self._init_model()

        export_params = {"format": format}
        model_cfg = self.config.get("model", {})
        export_params["imgsz"] = model_cfg.get("imgsz", 640)
        export_params.update(kwargs)

        print(f"\n[INFO] 导出模型为 {format.upper()} 格式...")
        path = self.model.export(**export_params)
        print(f"[INFO] 导出完成: {path}")
        return str(path)

    # ─────────────────────────────────────────────────────────────────────
    # 工厂方法
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        model_path: str,
        config_path: Optional[str] = None,
    ) -> "PCMDefectDetector":
        """
        从已训练权重加载检测器。

        Args:
            model_path:  .pt 权重文件路径
            config_path: 可选配置文件（用于推理参数）

        Returns:
            已加载的 PCMDefectDetector 实例
        """
        detector = cls(config_path=config_path, model_path=model_path)
        detector._init_model()
        print(f"[INFO] 模型加载完成: {model_path}")
        return detector

    def get_model_info(self) -> Dict:
        """返回模型基本信息。"""
        if self.model is None:
            return {"status": "未初始化"}
        info = {
            "model_path": self._model_path,
            "config": self.config,
        }
        try:
            info["task"] = self.model.task
            info["names"] = self.model.names
        except Exception:
            pass
        return info
