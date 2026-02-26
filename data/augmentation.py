"""
PCM 彩板缺陷检测 - 金属表面专用数据增强管线

基于 albumentations 实现，针对金属板表面特点设计：
- 生产线光照变化（亮度/对比度）
- 传感器噪声模拟（高斯噪声/模糊）
- 高速运动模糊
- 不做垂直翻转（生产线方向固定）
"""

import cv2
import numpy as np

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


def get_train_transforms(img_size: int = 640):
    """
    训练集增强管线。

    针对 PCM 金属表面缺陷的特点：
    1. 划伤（细长线条）：对旋转、缩放敏感，用轻微旋转
    2. 凸包（局部突起）：CLAHE 增强有助于检测
    3. 脏污（不规则暗斑）：亮度/对比度变化可增强泛化
    4. 色差：HSV 轻微调整

    Args:
        img_size: 目标图像尺寸

    Returns:
        albumentations.Compose 增强管线
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError("请安装 albumentations: pip install albumentations")

    return A.Compose(
        [
            # ── 几何变换 ──────────────────────────────────────────
            A.LongestMaxSize(max_size=img_size, p=1.0),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114),  # YOLO 标准灰色填充
                p=1.0,
            ),
            # 轻微旋转（±5°），金属板在生产线上基本水平
            A.Rotate(limit=5, border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114), p=0.3),
            # 水平翻转（左右对称，凸包/划伤均可翻转）
            A.HorizontalFlip(p=0.5),
            # 随机裁剪后恢复尺寸（模拟局部检测）
            A.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.7, 1.0),
                ratio=(0.9, 1.1),
                p=0.3,
            ),

            # ── 光照与颜色 ────────────────────────────────────────
            # 亮度/对比度变化（模拟生产线光源波动）
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.5,
            ),
            # HSV 调整（细微色调变化，用于色差类缺陷泛化）
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.4,
            ),
            # CLAHE 对比度增强（凸包/压痕检测关键：增强低对比度缺陷）
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            # 伽马校正（模拟不同相机曝光）
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            # 阴影模拟（模拟生产线结构光阴影）
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=0.2,
            ),

            # ── 噪声与模糊 ────────────────────────────────────────
            # 高斯噪声（相机传感器噪声）
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            # 运动模糊（高速传送带运动）
            A.MotionBlur(blur_limit=(3, 7), p=0.2),
            # 高斯模糊（镜头轻微失焦）
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),

            # ── 遮挡模拟 ──────────────────────────────────────────
            # 随机擦除（模拟局部污迹/反光遮挡）
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(10, 40),
                hole_width_range=(10, 40),
                fill_value=114,
                p=0.2,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",          # YOLO 格式：cx cy w h（归一化）
            label_fields=["class_labels"],
            min_visibility=0.3,     # 增强后至少保留 30% 可见面积
            clip=True,
        ),
    )


def get_val_transforms(img_size: int = 640):
    """
    验证/测试集变换（仅 resize + pad，无增强）。

    Args:
        img_size: 目标图像尺寸

    Returns:
        albumentations.Compose 变换管线
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError("请安装 albumentations: pip install albumentations")

    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size, p=1.0),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            clip=True,
        ),
    )


def apply_metal_specific_augmentation(image: np.ndarray) -> np.ndarray:
    """
    金属表面专用的额外增强（离线数据扩增用）。

    模拟常见的金属板表面干扰：
    - 轻微反光（高亮椭圆斑）
    - 轻微油污（半透明深色斑）
    - 传送带辊痕（周期性横纹）

    Args:
        image: 输入图像 (H, W, 3) BGR

    Returns:
        增强后图像
    """
    h, w = image.shape[:2]
    aug = image.copy().astype(np.float32)

    # 随机反光（高斯亮斑）
    if np.random.random() < 0.3:
        cx = np.random.randint(w // 4, 3 * w // 4)
        cy = np.random.randint(h // 4, 3 * h // 4)
        radius = np.random.randint(20, 80)
        intensity = np.random.uniform(0.3, 0.6)
        y_idx, x_idx = np.ogrid[:h, :w]
        mask = np.exp(-((x_idx - cx) ** 2 + (y_idx - cy) ** 2) / (2 * radius ** 2))
        aug += mask[:, :, np.newaxis] * 255 * intensity
        aug = np.clip(aug, 0, 255)

    # 辊印模拟（淡色横纹，间距固定）
    if np.random.random() < 0.15:
        period = np.random.randint(30, 80)
        alpha = np.random.uniform(0.05, 0.15)
        for y in range(0, h, period):
            aug[max(0, y - 1) : y + 2, :] = aug[max(0, y - 1) : y + 2, :] * (1 - alpha)

    return aug.astype(np.uint8)
