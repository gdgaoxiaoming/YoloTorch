# import torch
# print(torch.cuda.is_available())  # 返回 True 表示支持 GPU
#
# from ultralytics import YOLO
#
# model = YOLO("yolov8n.pt")  # 自动下载模型
# results = model.predict(source="https://ultralytics.com/images/bus.jpg", save=True)

import os
from PIL import Image
import numpy as np


def calculate_average_image_size(directory_path):
    # 存储所有图片的尺寸
    sizes = []

    # 遍历目录中的所有文件
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.png'):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        sizes.append((width, height))
                        print(f"已处理: {file} - 尺寸: {width}x{height}")
                except Exception as e:
                    print(f"无法处理文件 {file_path}: {str(e)}")

    if not sizes:
        print("未找到PNG文件")
        return None

    # 计算平均尺寸
    widths, heights = zip(*sizes)
    avg_width = np.mean(widths)
    avg_height = np.mean(heights)

    return avg_width, avg_height, len(sizes)


# 指定目录路径
directory = r"D:\下载\train_chars-20250628T091716Z-1-001\train_chars"

# 计算平均尺寸
result = calculate_average_image_size(directory)

if result:
    avg_width, avg_height, count = result
    print("\n" + "=" * 60)
    print(f"处理完成! 共找到 {count} 个PNG文件")
    print(f"平均宽度: {avg_width:.2f} 像素")
    print(f"平均高度: {avg_height:.2f} 像素")
    print("=" * 60)
