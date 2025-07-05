import time

from ultralytics import YOLO
import cv2
import os
from pathlib import Path

# 初始化模型
model = YOLO("/content/drive/MyDrive/YoloTorch/runs/detect/train2/weights/last.pt")

# 输入和输出路径
input_dir = Path("/content/drive/MyDrive/YoloTorch/dataset/train")
output_dir = Path("/content/drive/MyDrive/YoloTorch/dataset/train_chars")
output_dir.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在
idx = 0

# 处理所有验证码图片
for img_path in input_dir.glob("*.png"):
    # 解析文件名 (083o_20250627211057059.png -> ["083o", "20250627211057059"])
    parts = img_path.stem.split('_')
    chars = parts[0]
    if len(parts) < 2:
        continue  # 跳过不匹配格式的文件

    timestamp = int(time.time())  # 获取时间戳部分
    result = model(img_path)[0]  # 获取第一个预测结果

    # 只处理检测到4个字符的图片
    if len(result.boxes) == 4:
        print(f"处理图片: {img_path.name} - 检测到4个字符")

        # 读取原始图片
        img = cv2.imread(str(img_path))

        # 按从左到右顺序排序边界框 (基于x坐标)
        boxes = sorted(result.boxes.xyxy.tolist(), key=lambda box: box[0])
        classes = result.boxes.cls.tolist()
        class_names = [result.names[int(cls)] for cls in classes]

        # 裁剪和保存每个字符
        for i, (box, char) in enumerate(zip(boxes, chars)):
            # 裁剪字符区域 (增加5像素的边界)
            x1, y1, x2, y2 = map(int, box)
            margin = 5
            char_img = img[max(0, y1 - margin):min(img.shape[0], y2 + margin),
                       max(0, x1 - margin):min(img.shape[1], x2 + margin)]

            # 按正确顺序保存字符
            save_path = output_dir / f"{char}_{timestamp}{idx}.png"
            idx += 1
            cv2.imwrite(str(save_path), char_img)
            print(f"保存字符: {save_path.name}")
    else:
        print(f"跳过图片: {img_path.name} - 检测到{len(result.boxes)}个字符")

print("处理完成! 所有符合条件的结果已保存到", output_dir)