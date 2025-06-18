import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.DeepLabv3model import DeepLabV3PlusForNucleiSegmentation
from utils.dataset import MoNuSegDataset
import matplotlib.pyplot as plt
import os

def train_net(net, device, data_path, epochs=25, batch_size=4, lr=0.0001):
    # 加载训练集和验证集
    train_dataset = MoNuSegDataset(data_path, mode='train', augment=True)
    val_dataset = MoNuSegDataset(data_path, mode='test', augment=False)  # 验证集不进行数据增强

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # 定义优化器和学习率调度器
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # 定义损失函数
    criterion = nn.BCEWithLogitsLoss()

    # 记录训练和验证的损失
    train_losses = []
    val_losses = []

    # 训练过程
    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 验证过程
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = net(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # 更新学习率
        scheduler.step(val_loss)

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # 保存训练后的模型
    torch.save(net.state_dict(), 'deeplabv3plus_model.pth')

    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve_deeplabv3plus.png')
    plt.show()

if __name__ == "__main__":
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 初始化模型
    model = DeepLabV3PlusForNucleiSegmentation(num_classes=1)  # 假设是单类别分割
    model.to(device)

    # 数据路径
    data_path = r"C:\ZHyl\Dzy\data"

    # 开始训练
    train_net(model, device, data_path)