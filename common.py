import os
import random
import string
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from datetime import datetime

# 创建存储目录
os.makedirs('./dataset/train', exist_ok=True)


def get_clear_font(size=36):
    """获取清晰字体（优先尝试系统字体）"""
    try:
        return ImageFont.truetype("arialbd.ttf", size)
    except:
        try:
            return ImageFont.truetype("Arial Bold.ttf", size)
        except:
            return ImageFont.load_default().font


def get_full_bg_color():
    """生成浅色背景"""
    return (random.randint(220, 255),
            random.randint(220, 255),
            random.randint(220, 255))


def get_text_color():
    """生成深色文字（增强对比度）"""
    return (random.randint(0, 50),  # 更暗的色调
            random.randint(0, 50),
            random.randint(0, 50))


def apply_mild_warp(image, bg_color):
    """安全的小幅扭曲（旋转+平滑）"""
    return image.filter(ImageFilter.SMOOTH).rotate(
        random.uniform(-3, 3),  # 旋转角度±3度
        fillcolor=bg_color
    )


# 字符集：仅数字和小写字母
chars = string.digits + string.ascii_lowercase  # 移除大写字母[1,3](@ref)

for _ in range(3000):
    # 生成4位验证码
    captcha_text = ''.join(random.sample(chars, 4))
    bg_color = get_full_bg_color()

    # 创建新尺寸图像 (宽度180, 高度60)
    img = Image.new('RGB', (180, 60), bg_color)
    draw = ImageDraw.Draw(img)

    # 干扰线（适配新尺寸）
    for i in range(4):
        start = (random.randint(0, 180), random.randint(0, 60))
        end = (random.randint(0, 180), random.randint(0, 60))
        draw.line([start, end],
                  fill=(random.randint(180, 230), random.randint(180, 230), random.randint(180, 230)),
                  width=1)

    # 文字绘制（适配新尺寸）
    font = get_clear_font(32)  # 稍小字号适应新尺寸
    text_color = get_text_color()

    # 计算总宽度并居中
    char_widths = [font.getbbox(char)[2] - font.getbbox(char)[0] for char in captcha_text]
    total_width = sum(char_widths) + 3 * (len(captcha_text) - 1)  # 含间距
    start_x = (180 - total_width) // 2

    # 逐个绘制字符
    for i, char in enumerate(captcha_text):
        y_offset = random.randint(-5, 5)
        draw.text((start_x, 15 + y_offset), char, font=font, fill=text_color)
        start_x += char_widths[i] + 3  # 字符间距

    # 噪点（适配新尺寸）
    for _ in range(30):
        draw.point((random.randint(0, 180), random.randint(0, 60)),
                   fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    # 应用扭曲
    img = apply_mild_warp(img, bg_color)

    # 边缘增强
    img = img.filter(ImageFilter.EDGE_ENHANCE)

    # 保存
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
    filename = f"{captcha_text}_{timestamp}.png"
    img.save(f'./dataset/train/{filename}', quality=95)

print("验证码生成完成！保存在 ./dataset/train 目录")