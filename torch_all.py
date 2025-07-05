import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import random


# 设置随机种子确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()


# 数据集类
class CharDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        初始化字符数据集

        参数:
            root_dir: 图片根目录
            transform: 数据增强变换
        """
        self.root_dir = root_dir
        self.transform = transform

        # 获取所有PNG文件
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]

        # 创建字符到标签的映射
        self.char_to_idx = {}
        self.idx_to_char = {}

        # 36个字符：数字0-9和小写字母a-z
        chars = [str(i) for i in range(10)] + [chr(ord('a') + i) for i in range(26)]
        for idx, char in enumerate(chars):
            self.char_to_idx[char] = idx
            self.idx_to_char[idx] = char

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 获取文件名
        file_name = self.image_files[idx]

        # 从文件名中提取标签（第一个字符，下划线前的内容）
        label_char = file_name.split('_')[0]
        label = self.char_to_idx[label_char]

        # 加载图像
        img_path = os.path.join(self.root_dir, file_name)
        image = Image.open(img_path)

        # 转换格式
        if image.mode == 'RGB':
            # 转换RGB为灰度图并归一化到单通道
            image = image.convert('L')
        elif image.mode == 'L':
            # 灰度图，保持原样
            pass
        else:
            # 其他格式转换为灰度
            image = image.convert('L')

        # 转换为张量
        if self.transform:
            image = self.transform(image)
        else:
            # 基本转换
            image = transforms.ToTensor()(image)

        # 调整为统一尺寸 (高度, 宽度) = (36, 30)
        if image.shape[1] != 36 or image.shape[2] != 30:
            image = transforms.functional.resize(image, (36, 30))

        return image, label


# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 单通道归一化
])


# 定义模型
class CharRecognitionModel(nn.Module):
    def __init__(self):
        super(CharRecognitionModel, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 展平后的特征图大小: (32, 9, 7) 因为 (36/2/2=9, 30/2/2=7.5 -> 取整为7)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 9 * 7, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 36)  # 36个类别

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20, patience=3):
    """
    训练模型并验证

    参数:
        model: 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 训练设备 (CPU/GPU)
        num_epochs: 训练轮数
        patience: 提前停止的耐心值
    """
    best_val_acc = 0.0
    no_improve = 0  # 没有提升的轮次计数

    # 记录训练过程
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for images, labels in train_progress:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计信息
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # 更新进度条
            train_acc = correct_train / total_train
            train_progress.set_postfix(loss=loss.item(), acc=train_acc)

        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Validation]'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct_val / total_val
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_char_recognition_model.pth')
            print(f'New best model saved with val acc: {val_acc:.4f}')
            no_improve = 0
        else:
            no_improve += 1
            print(f'No improvement for {no_improve} epochs')

        # 提前停止
        if no_improve >= patience:
            print(f'Early stopping after {epoch + 1} epochs')
            break

    # 绘制训练历史
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    return history


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 数据集路径
    train_dir = '/content/drive/MyDrive/YoloTorch/dataset/train_chars'
    val_dir = '/content/drive/MyDrive/YoloTorch/dataset/val_chars'

    # 创建数据集实例
    train_dataset = CharDataset(train_dir, transform=transform)
    val_dataset = CharDataset(val_dir, transform=transform)

    # 创建数据加载器
    batch_size = 64
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    print(f'Train samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')

    # 创建模型
    model = CharRecognitionModel().to(device)
    print(model)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # 监控验证准确率
        factor=0.5,
        patience=2,
        verbose=True
    )

    # 训练模型
    history = train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=60
    )

    # 最终保存模型
    torch.save(model.state_dict(), 'final_char_recognition_model.pth')
    print('Final model saved')


if __name__ == '__main__':
    main()