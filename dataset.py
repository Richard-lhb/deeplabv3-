import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import glob  # 添加对 glob 模块的导入
from torchvision import transforms
import matplotlib.pyplot as plt

class MoNuSegDataset(Dataset):
    def __init__(self, data_path, mode='train', target_size=(512, 512), augment=True):
        """
        MoNuSeg 数据集加载器
        :param data_path: 数据集根目录
        :param mode: 数据集模式（'train' 或 'test'）
        :param target_size: 图像统一调整的目标尺寸
        :param augment: 是否进行数据增强
        """
        self.data_path = data_path
        self.mode = mode
        self.target_size = target_size
        self.augment = augment

        # 根据模式选择数据路径
        if mode == 'train':
            self.image_dir = os.path.join(data_path, 'kmms_training', 'images')
            self.mask_dir = os.path.join(data_path, 'kmms_training', 'masks')
        elif mode == 'test':
            self.image_dir = os.path.join(data_path, 'kmms_test', 'images')
            self.mask_dir = os.path.join(data_path, 'kmms_test', 'masks')
        else:
            raise ValueError("Mode must be 'train' or 'test'")

        # 获取所有图像路径
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, '*')))
        self.mask_paths = sorted(glob.glob(os.path.join(self.mask_dir, '*')))

        # 确保图像和掩码数量一致
        assert len(self.image_paths) == len(self.mask_paths), "Number of images and masks do not match"

    def augment_data(self, image, mask):
        """
        数据增强
        :param image: 输入图像
        :param mask: 对应掩码
        :return: 增强后的图像和掩码
        """
        if random.random() > 0.5:  # 随机水平翻转
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        if random.random() > 0.5:  # 随机垂直翻转
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        if random.random() > 0.5:  # 随机添加高斯噪声
            noise = np.random.normal(0, 0.1, image.shape)
            image = image + noise
            image = np.clip(image, 0, 1)  # 限制在 [0, 1] 范围内
        return image, mask

    def __getitem__(self, index):
        # 读取图像和掩码
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        # 不进行灰度化处理，读取彩色图像
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # 掩码仍然以灰度模式读取
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 调整图像大小
        image = cv2.resize(image, self.target_size)
        mask = cv2.resize(mask, self.target_size)

        # 归一化处理
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # 数据增强（仅在训练模式下）
        if self.mode == 'train' and self.augment:
            image, mask = self.augment_data(image, mask)

        # 转换为 PyTorch 张量
        # 注意：这里需要调整通道维度，因为彩色图像有 3 个通道
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask

    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    # 测试代码
    data_path = r"C:\ZHyl\Dzy\data"
    train_dataset = MoNuSegDataset(data_path, mode='train')
    test_dataset = MoNuSegDataset(data_path, mode='test', augment=False)

    print("训练集数据个数：", len(train_dataset))
    print("测试集数据个数：", len(test_dataset))

    # 测试数据加载器
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)
    for images, masks in train_loader:
        print("训练图像尺寸：", images.shape)
        print("训练掩码尺寸：", masks.shape)
        break

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=2, shuffle=False)
    for images, masks in test_loader:
        print("测试图像尺寸：", images.shape)
        print("测试掩码尺寸：", masks.shape)
        break

    # 展示训练集的一张图像及其掩码
    train_image, train_mask = train_dataset[0]
    train_image = train_image.permute(1, 2, 0).numpy()
    train_mask = train_mask.squeeze(0).numpy()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(train_image)
    plt.title('Train Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(train_mask, cmap='gray')
    plt.title('Train Mask')
    plt.axis('off')
    plt.show()

    # 展示测试集的一张图像及其掩码
    test_image, test_mask = test_dataset[0]
    test_image = test_image.permute(1, 2, 0).numpy()
    test_mask = test_mask.squeeze(0).numpy()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(test_image)
    plt.title('Test Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(test_mask, cmap='gray')
    plt.title('Test Mask')
    plt.axis('off')
    plt.show()