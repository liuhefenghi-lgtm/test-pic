==================================================================
  PCM 彩板缺陷检测 - 打标记说明
==================================================================

【目录结构】
  raw_data/
  ├── classes.txt          ← LabelImg 读取的类别文件（勿修改顺序！）
  ├── images/              ← 将 PCM 图像放入此目录
  │   ├── pcm_001.jpg
  │   └── ...
  ├── labels/              ← LabelImg 自动保存标注到此目录
  │   ├── pcm_001.txt
  │   └── ...
  └── README_打标记说明.txt

【类别 ID 对照】（ID 顺序与 classes.txt 一致）
  0 = scratch   (划伤)
  1 = bump      (凸包)
  2 = stain     (脏污)
  3 = indentation (压痕)
  4 = color_diff  (色差)
  5 = blister   (起泡)

【LabelImg 安装】
  pip install labelImg
  labelImg

【LabelImg 操作步骤】
  1. 菜单 → Change Save Dir → 选择 raw_data/labels/
  2. 菜单 → Open Dir         → 选择 raw_data/images/
  3. 菜单 → View → YOLO format（切换为 YOLO 输出格式）
  4. 点击左侧 "YOLO" 按钮确认格式
  5. 对每张图：
     - 按 W 键开始画框（鼠标拖动）
     - 弹出类别选择窗口 → 选择缺陷类型
     - Ctrl+S 保存（或勾选 Auto Save）
  6. D 键切换下一张，A 键切换上一张

【打标记技巧】
  划伤 (scratch):
    - 框要包住整条划痕，含两端
    - 细短划伤框可以稍微宽松一些（加 5~10px 余量）

  凸包 (bump):
    - 框住整个凸起区域（通常是圆形/椭圆）
    - 看反光斑点，框住亮斑及其周围轻微变形区

  脏污 (stain):
    - 框住整块污迹，包括浅色边缘区域
    - 多块分散污迹 → 每块单独画框

  压痕 (indentation):
    - 与凸包类似，但是向内凹陷
    - 通常看阴影边缘确定范围

  色差 (color_diff):
    - 框住颜色明显不同于周围的区域
    - 整块色差区域框住

  起泡 (blister):
    - 框住气泡隆起部分及周边变形区

【打完标记后】
  python prepare_dataset.py --source raw_data/ --output dataset/
  python visualize_labels.py --data dataset/ --split train --n 16 --save
  python train.py --data dataset/dataset.yaml

==================================================================
